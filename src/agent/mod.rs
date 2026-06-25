//! Core agent implementation with multi-provider LLM support and agentic capabilities.
//!
//! This module provides the [`Agent`] struct that orchestrates conversations between
//! users, multiple LLM providers (Bedrock, LM Studio, Anthropic, OpenAI), and tool systems.
//! You'll get streaming responses, 5-phase agentic execution, and robust error handling for production deployments.
//!
//! # Quick Start
//!
//! Create an agent with multi-provider support:
//!
//! ```no_run
//! use stood::agent::Agent;
//! use stood::llm::models::{Bedrock, LMStudio};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Simplest usage - defaults to Claude 3.5 Haiku via Bedrock
//!     let mut agent = Agent::builder().build().await?;
//!
//!     // Use Bedrock (cloud)
//!     let mut agent = Agent::builder()
//!         .model(Bedrock::ClaudeHaiku45)
//!         .system_prompt("You are a helpful assistant")
//!         .build().await?;
//!
//!     // Use LM Studio (local)
//!     let mut agent = Agent::builder()
//!         .model(LMStudio::Gemma3_12B)
//!         .system_prompt("You are a helpful assistant")
//!         .build().await?;
//!
//!     let result = agent.execute("Hello, how are you?").await?;
//!     println!("Agent: {}", result.response);
//!     Ok(())
//! }
//! ```
//!
//! # 5-Phase Agentic Execution
//!
//! All agents use 5-phase agentic execution automatically with a single `execute()` method:
//!
//! ```no_run
//! use stood::{Agent, tool};
//! use stood::llm::models::Bedrock;
//!
//! #[tool]
//! async fn search_web(query: String) -> Result<String, String> {
//!     // Your search implementation here
//!     Ok(format!("Search results for: {}", query))
//! }
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut agent = Agent::builder()
//!         .model(Bedrock::ClaudeHaiku45)
//!         .tool(search_web())
//!         .with_builtin_tools()
//!         .build().await?;
//!
//!     // Single method for all tasks - automatically uses 5-phase agentic execution
//!     let result = agent.execute(
//!         "Research the latest developments in quantum computing"
//!     ).await?;
//!
//!     println!("Final result: {}", result.response);
//!     println!("Tool calls made: {}", result.tools_called.len());
//!     println!("Execution cycles: {}", result.execution.cycles);
//!     Ok(())
//! }
//! ```
//!
//! # Parallel Tool Execution
//!
//! Configure parallel tool execution to improve performance when using multiple tools:
//!
//! ```no_run
//! use stood::agent::Agent;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Sequential execution (default) - tools run one at a time
//!     let mut sequential_agent = Agent::builder()
//!         .max_parallel_tools(1)
//!         .build().await?;
//!
//!     // Parallel execution - up to 4 tools run concurrently
//!     let mut parallel_agent = Agent::builder()
//!         .max_parallel_tools(4)
//!         .build().await?;
//!
//!     // Auto-detect CPU count for optimal parallelism
//!     let mut auto_agent = Agent::builder()
//!         .max_parallel_tools_auto()
//!         .build().await?;
//!
//!     let result = parallel_agent.execute(
//!         "Process these files, send emails, and generate reports simultaneously"
//!     ).await?;
//!
//!     println!("Completed {} tools in parallel", result.tools_called);
//!     Ok(())
//! }
//! ```
//!
//! # Architecture
//!
//! The agent coordinates four main components:
//!
//! - **Multi-Provider LLM Integration** - Supports Bedrock, LM Studio, Anthropic, OpenAI providers
//! - **Conversation Management** - Handles message history and context across providers
//! - **Tool Orchestration** - Executes tools with parallel execution support
//! - **5-Phase EventLoop** - Reasoning → Tool Selection → Tool Execution → Reflection → Response Generation
//!
//! The [`EventLoop`] orchestrates the 5-phase agentic execution system with automatic task evaluation
//! to determine when the user's request has been fully satisfied.
//!
//! # Performance
//!
//! - Maintains conversation context efficiently across multiple turns
//! - Supports streaming responses for real-time interactions
//! - Parallel tool execution where possible
//! - Configurable retry logic and error recovery
//!
//! # Key Types
//!
//! - [`Agent`] - Main agent implementation with conversation management
//! - [`AgentBuilder`] - Builder pattern for agent configuration
//! - [`AgentConfig`] - Configuration for model parameters and behavior
//! - [`ConversationManager`] - Handles message history and context
//! - [`EventLoop`] - Orchestrates agentic execution workflows

// BedrockClient now in llm::providers::bedrock
use crate::tools::{Tool, ToolMiddleware, ToolRegistry};
use crate::types::Message;
use crate::{Result, StoodError};
use std::sync::Arc;
use std::time::Duration;
#[allow(unused_imports)] // Used in future features
use uuid::Uuid;

// LLM provider system imports
use crate::llm::providers::retry::RetryConfig;
use crate::llm::registry::PROVIDER_REGISTRY;
use crate::llm::traits::{CacheStrategy, LlmModel, LlmProvider, ProviderType};

use crate::telemetry::{StoodTracer, TelemetryConfig};

pub mod callbacks;
pub mod config;
pub mod conversation;
pub mod evaluation;
pub mod event_loop;
pub mod result;

pub use callbacks::{
    CallbackHandler, CallbackHandlerConfig, CompositeCallbackHandler, NullCallbackHandler,
    PerformanceCallbackHandler, PrintingCallbackHandler, PrintingConfig,
};
pub use config::{ExecutionConfig, LogLevel};
pub use conversation::ConversationManager;
pub use evaluation::{EvaluationStrategy, PerspectiveConfig};
pub use event_loop::{EventLoop, EventLoopConfig, EventLoopResult};
pub use result::{AgentResult, ExecutionDetails, PerformanceMetrics, TokenUsage};

#[cfg(test)]
mod integration_tests;

#[cfg(test)]
mod tools_integration_tests;

#[cfg(test)]
mod llm_integration_test;

/// Configuration for agent behavior and model parameters.
///
/// This struct controls how your agent interacts with multiple LLM providers,
/// including temperature settings, token limits, provider selection, and system prompts that
/// shape the agent's behavior.
///
/// # Examples
///
/// Create a conservative configuration for code analysis:
/// ```
/// use stood::agent::AgentConfig;
/// use stood::llm::traits::ProviderType;
///
/// let config = AgentConfig {
///     provider: ProviderType::Bedrock,
///     model_id: "us.anthropic.claude-3-5-sonnet-20241022-v2:0".to_string(),
///     temperature: Some(0.1),  // More deterministic
///     max_tokens: Some(8192),  // Longer responses
///     system_prompt: Some(
///         "You are a code analysis expert. Provide detailed,
///          accurate technical explanations.".to_string()
///     ),
///     agent_id: None,
///     agent_name: None,
///     telemetry_config: None,
///     retry_config: None,
/// };
/// ```
///
/// # Model Selection Guide
///
/// **Bedrock Provider:**
/// - **Claude Haiku 3.5** - Fast, cost-effective for simple tasks
/// - **Claude Sonnet 3.5** - Balanced performance for most applications
/// - **Claude Opus 3** - Maximum capability for complex reasoning
/// - **Nova Pro/Lite/Micro** - Amazon's models for specific use cases
///
/// **LM Studio Provider (Local):**
/// - **Gemma 3 12B/27B** - Google's open models for local development
/// - **Llama 3 70B** - Meta's large model for complex tasks
/// - **Mistral 7B** - Efficient model for general tasks
/// - **Tessa Rust T1 7B** - Specialized Rust code model
#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub provider: ProviderType,
    pub model_id: String,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub system_prompt: Option<String>,
    pub agent_id: Option<String>,
    pub agent_name: Option<String>,
    /// Prompt caching strategy for reducing latency and costs
    pub cache_strategy: CacheStrategy,

    pub telemetry_config: Option<TelemetryConfig>,
    pub retry_config: Option<RetryConfig>,
}

/// Context passed through tools for parent-child tracking and telemetry
#[derive(Debug, Clone)]
pub struct AgentContext {
    pub agent_id: String,
    pub agent_name: Option<String>,
    pub agent_type: String,
    pub span_context: Option<opentelemetry::Context>,
}

impl AgentContext {
    /// Create a new AgentContext from an Agent instance
    pub fn from_agent(agent: &Agent, agent_type: impl Into<String>) -> Self {
        Self {
            agent_id: agent.agent_id.clone(),
            agent_name: agent.agent_name.clone(),
            agent_type: agent_type.into(),
            span_context: None, // Will be set by telemetry system
        }
    }

    /// Create a new AgentContext with specific parameters
    pub fn new(
        agent_id: impl Into<String>,
        agent_name: Option<String>,
        agent_type: impl Into<String>,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            agent_name,
            agent_type: agent_type.into(),
            span_context: None,
        }
    }

    /// Set the span context for telemetry tracking
    pub fn with_span_context(mut self, span_context: opentelemetry::Context) -> Self {
        self.span_context = Some(span_context);
        self
    }
}

/// Performance metrics and operational summary for an agent instance.
///
/// This struct provides insights into your agent's conversation state,
/// token usage patterns, and configuration details that help with
/// optimization and monitoring.
///
/// # Use Cases
///
/// - Monitor conversation length for context management
/// - Track message counts for usage billing
/// - Verify system prompt configuration
/// - Performance optimization and debugging
///
/// # Examples
///
/// ```no_run
/// # use stood::agent::Agent;
/// # async fn example(mut agent: Agent) {
/// let summary = agent.get_performance_summary();
///
/// println!("Messages exchanged: {}", summary.total_messages);
/// println!("Using model: {:?}", summary.model);
///
/// if summary.conversation_length > 50 {
///     println!("Consider conversation cleanup for performance");
/// }
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct AgentPerformanceSummary {
    pub total_messages: usize,
    pub conversation_length: usize,
    pub has_system_prompt: bool,
    pub provider: ProviderType,
    pub model_id: String,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            provider: ProviderType::Bedrock,
            model_id: "us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string(),
            temperature: Some(0.7),
            max_tokens: Some(4096),
            system_prompt: None,
            agent_id: None,
            agent_name: None,
            cache_strategy: CacheStrategy::default(),

            telemetry_config: None,
            retry_config: None,
        }
    }
}

/// Core agent implementation providing conversational AI with multi-provider tool capabilities.
///
/// The `Agent` orchestrates interactions between users, multiple LLM providers,
/// and external tools to create intelligent, agentic workflows. You'll get
/// conversation management, streaming responses, and robust error handling.
///
/// # Architecture
///
/// ```text
/// User Input → Agent → ProviderRegistry → LLM Provider → Provider API
///      ↓           ↓                                        (Bedrock, LM Studio, etc.)
/// Tool System ← EventLoop (5-phase agentic execution)
/// ```
///
/// # Key Capabilities
///
/// - **Unified Execution** - Single `execute()` method for all tasks (simple to complex)
/// - **5-Phase Agentic System** - Reasoning → Tool Selection → Tool Execution → Reflection → Response Generation
/// - **Multi-Provider Support** - Bedrock, LM Studio, Anthropic, OpenAI providers
/// - **Conversation Memory** - Maintains context across interactions and providers
/// - **Tool Integration** - Parallel tool execution with compile-time validation
/// - **Streaming Support** - Real-time response delivery
/// - **Task Evaluation** - Automatic assessment of user intent satisfaction
/// - **Error Recovery** - Robust handling of API and tool failures with provider failover
///
/// # Examples
///
/// Basic usage with different providers:
/// ```no_run
/// # use stood::agent::Agent;
/// # use stood::llm::models::{Bedrock, LMStudio};
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Use Bedrock (cloud)
/// let mut bedrock_agent = Agent::builder()
///     .model(Bedrock::ClaudeHaiku45)
///     .build().await?;
/// let result = bedrock_agent.execute("Explain quantum computing").await?;
/// println!("Bedrock: {}", result.response);
///
/// // Use LM Studio (local)
/// let mut local_agent = Agent::builder()
///     .model(LMStudio::Gemma3_12B)
///     .build().await?;
/// let result = local_agent.execute("Explain quantum computing").await?;
/// println!("Local: {}", result.response);
/// # Ok(())
/// # }
/// ```
///
/// Agentic execution with tools (5-phase system):
/// ```no_run
/// # use stood::agent::Agent;
/// # use stood::llm::models::Bedrock;
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let mut agent = Agent::builder()
///     .model(Bedrock::ClaudeHaiku45)
///     .with_builtin_tools()
///     .build().await?;
///
/// // Single execute() method - automatically uses 5-phase agentic execution
/// let result = agent.execute(
///     "Research and summarize the latest AI developments"
/// ).await?;
///
/// println!("Final response: {}", result.response);
/// println!("Tools used: {}", result.tools_called.len());
/// println!("Execution cycles: {}", result.execution.cycles);
/// println!("Task completed: {}", result.success);
/// # Ok(())
/// # }
/// ```
///
/// # Performance Characteristics
///
/// - Conversation context: O(n) memory where n = message count
/// - Tool execution: Parallel where dependencies allow
/// - Model calls: Batched for efficiency, streamed for responsiveness
/// - Error recovery: Automatic retries with exponential backoff
pub struct Agent {
    agent_id: String,
    agent_name: Option<String>,
    provider: Arc<dyn LlmProvider>,
    model: Box<dyn LlmModel>,
    config: AgentConfig,
    conversation: ConversationManager,
    tool_registry: ToolRegistry,
    execution_config: ExecutionConfig, // Pre-configured execution settings

    tracer: Option<StoodTracer>,
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("agent_id", &self.agent_id)
            .field("agent_name", &self.agent_name)
            .field("provider_type", &self.model.provider())
            .field("model_id", &self.model.model_id())
            .field("config", &self.config)
            .field("conversation", &self.conversation)
            .field("tool_registry", &self.tool_registry)
            .field("execution_config", &self.execution_config)
            .finish()
    }
}

impl Clone for Agent {
    fn clone(&self) -> Self {
        // Create a new model instance based on config
        let model = create_model_from_config(&self.config.provider, &self.config.model_id);

        Self {
            agent_id: self.agent_id.clone(),
            agent_name: self.agent_name.clone(),
            provider: Arc::clone(&self.provider),
            model,
            config: self.config.clone(),
            conversation: self.conversation.clone(),
            tool_registry: self.tool_registry.clone(),
            execution_config: self.execution_config.clone(),
            tracer: self.tracer.clone(),
        }
    }
}

/// Utility function to create model instances from provider and model_id
#[allow(deprecated)]
fn create_model_from_config(provider: &ProviderType, model_id: &str) -> Box<dyn LlmModel> {
    tracing::info!(
        target: "stood::agent::create_model_from_config",
        provider = ?provider,
        model_id = model_id,
        "Creating model from config"
    );
    let result: Box<dyn LlmModel> = match provider {
        ProviderType::Bedrock => {
            match model_id {
                // Claude 4.6 / 4.8 models (latest, recommended)
                "us.anthropic.claude-opus-4-8" => {
                    Box::new(crate::llm::models::Bedrock::ClaudeOpus48)
                }
                "us.anthropic.claude-sonnet-4-6" => {
                    Box::new(crate::llm::models::Bedrock::ClaudeSonnet46)
                }
                // Claude 4.5 models
                "us.anthropic.claude-sonnet-4-5-20250929-v1:0" => {
                    Box::new(crate::llm::models::Bedrock::ClaudeSonnet45)
                }
                "us.anthropic.claude-haiku-4-5-20251001-v1:0" => {
                    Box::new(crate::llm::models::Bedrock::ClaudeHaiku45)
                }
                "us.anthropic.claude-opus-4-5-20251101-v1:0" => {
                    Box::new(crate::llm::models::Bedrock::ClaudeOpus45)
                }
                // Legacy Claude models (deprecated but still supported)
                "us.anthropic.claude-3-5-sonnet-20241022-v2:0" => {
                    Box::new(crate::llm::models::Bedrock::Claude35Sonnet)
                }
                "us.anthropic.claude-3-5-haiku-20241022-v1:0" => {
                    Box::new(crate::llm::models::Bedrock::Claude35Haiku)
                }
                "us.anthropic.claude-3-haiku-20240307-v1:0" => {
                    Box::new(crate::llm::models::Bedrock::ClaudeHaiku3)
                }
                "us.anthropic.claude-3-opus-20240229-v1:0" => {
                    Box::new(crate::llm::models::Bedrock::ClaudeOpus3)
                }
                // Nova models (legacy Nova 1)
                "us.amazon.nova-lite-v1:0" => Box::new(crate::llm::models::Bedrock::NovaLite),
                "us.amazon.nova-pro-v1:0" => Box::new(crate::llm::models::Bedrock::NovaPro),
                "us.amazon.nova-micro-v1:0" => Box::new(crate::llm::models::Bedrock::NovaMicro),
                // Nova 2 models (current generation)
                "us.amazon.nova-2-lite-v1:0" => Box::new(crate::llm::models::Bedrock::Nova2Lite),
                "us.amazon.nova-2-pro-v1:0" => Box::new(crate::llm::models::Bedrock::Nova2Pro),
                "us.amazon.nova-premier-v1:0" | "amazon.nova-premier-v1:0" => {
                    Box::new(crate::llm::models::Bedrock::NovaPremier)
                }
                _ => Box::new(crate::llm::models::Bedrock::ClaudeHaiku45), // Default fallback
            }
        }
        ProviderType::LmStudio => {
            match model_id {
                "google/gemma-3-12b" => Box::new(crate::llm::models::LMStudio::Gemma3_12B),
                "google/gemma-3-27b" => Box::new(crate::llm::models::LMStudio::Gemma3_27B),
                "llama-3-70b" => Box::new(crate::llm::models::LMStudio::Llama3_70B),
                "mistral-7b" => Box::new(crate::llm::models::LMStudio::Mistral7B),
                "tessa-rust-t1-7b" => Box::new(crate::llm::models::LMStudio::TessaRust7B),
                _ => Box::new(crate::llm::models::LMStudio::Gemma3_12B), // Default fallback
            }
        }
        _ => Box::new(crate::llm::models::Bedrock::ClaudeHaiku45), // Default fallback for other providers
    };
    tracing::info!(
        target: "stood::agent::create_model_from_config",
        resulting_model_id = result.model_id(),
        "Model created from config"
    );
    result
}

impl Agent {
    pub fn builder() -> AgentBuilder {
        AgentBuilder::new()
    }

    /// Internal constructor - only called by AgentBuilder
    async fn build_internal(
        provider: Arc<dyn LlmProvider>,
        model: Box<dyn LlmModel>,
        config: AgentConfig,
        tools: Vec<Box<dyn Tool>>,
        middlewares: Vec<Arc<dyn ToolMiddleware>>,
        execution_config: ExecutionConfig,
        agent_id: Option<String>,
        agent_name: Option<String>,
    ) -> Result<Self> {
        crate::perf_checkpoint!("stood.build_internal.start");
        let _build_internal_guard = crate::perf_guard!("stood.build_internal");

        let mut conversation = crate::perf_timed!("stood.build_internal.conversation_manager", {
            ConversationManager::new()
        });

        // Set system prompt from config if provided
        conversation.set_system_prompt(config.system_prompt.clone());

        // Initialize tool registry and register tools
        let tool_registry = crate::perf_timed!("stood.build_internal.tool_registry_new", {
            ToolRegistry::new()
        });
        #[cfg(feature = "perf-timing")]
        let tool_count = tools.len();
        crate::perf_timed!("stood.build_internal.register_tools", {
            for tool in tools {
                tool_registry.register_tool(tool).await.map_err(|e| {
                    StoodError::configuration_error(format!("Failed to register tool: {}", e))
                })?;
            }
            Ok::<(), StoodError>(())
        })?;
        #[cfg(feature = "perf-timing")]
        crate::perf_checkpoint!("stood.build_internal.tools_registered", &format!("count={}", tool_count));

        // Register middleware
        #[cfg(feature = "perf-timing")]
        let middleware_count = middlewares.len();
        crate::perf_timed!("stood.build_internal.register_middleware", {
            for middleware in middlewares {
                tool_registry.add_middleware(middleware).await;
            }
        });
        #[cfg(feature = "perf-timing")]
        crate::perf_checkpoint!("stood.build_internal.middleware_registered", &format!("count={}", middleware_count));

        // Initialize smart telemetry with auto-detection
        crate::perf_checkpoint!("stood.build_internal.telemetry_init.start");
        let tracer = crate::perf_timed!("stood.build_internal.telemetry_init", {
            // Use provided config or auto-detect with proper log level
            let mut tel_config = config.telemetry_config.clone().unwrap_or_default();

            // Apply execution_config log level to telemetry
            let telemetry_log_level = match execution_config.log_level {
                crate::agent::config::LogLevel::Off => crate::telemetry::LogLevel::OFF,
                crate::agent::config::LogLevel::Info => crate::telemetry::LogLevel::INFO,
                crate::agent::config::LogLevel::Debug => crate::telemetry::LogLevel::DEBUG,
                crate::agent::config::LogLevel::Trace => crate::telemetry::LogLevel::TRACE,
            };
            tel_config.set_log_level(telemetry_log_level);

            // Use init_async to automatically create the CloudWatch log group
            // This is required for spans to appear in the GenAI Dashboard
            match StoodTracer::init_async(tel_config).await {
                Ok(tracer_opt) => {
                    // Initialize tracing subscriber with OpenTelemetry layer for automatic context propagation
                    if tracer_opt.is_some() {
                        if let Err(e) = crate::telemetry::StoodTracer::init_tracing_subscriber() {
                            tracing::warn!("Failed to initialize tracing subscriber: {}", e);
                        }
                    }
                    tracer_opt
                }
                Err(e) => {
                    tracing::warn!("Failed to initialize telemetry: {}", e);
                    None
                }
            }
        });
        crate::perf_checkpoint!("stood.build_internal.telemetry_init.end");

        crate::perf_checkpoint!("stood.build_internal.end");
        Ok(Self {
            agent_id: agent_id
                .or_else(|| config.agent_id.clone())
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
            agent_name: agent_name.or_else(|| config.agent_name.clone()),
            provider,
            model,
            config,
            conversation,
            tool_registry,
            execution_config,

            tracer,
        })
    }

    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    pub fn agent_name(&self) -> Option<&str> {
        self.agent_name.as_deref()
    }

    /// Get the telemetry configuration for this agent
    pub fn telemetry_config(&self) -> Option<&TelemetryConfig> {
        self.config.telemetry_config.as_ref()
    }

    /// Get the cancellation token if cancellation was enabled during building
    ///
    /// Returns the cancellation token that can be used to cancel agent execution
    /// from other tasks, threads, or event handlers.
    ///
    /// # Returns
    /// * `Some(CancellationToken)` if `.with_cancellation()` was called during building
    /// * `None` if cancellation was not enabled
    ///
    /// # Examples
    /// ```no_run
    /// use stood::agent::Agent;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let agent = Agent::builder()
    ///     .with_cancellation()
    ///     .build().await?;
    ///
    /// // Get cancellation token
    /// if let Some(cancel_token) = agent.cancellation_token() {
    ///     // Use in ESC handler, timeout, etc.
    ///     tokio::spawn(async move {
    ///         // Cancel after 10 seconds
    ///         tokio::time::sleep(std::time::Duration::from_secs(10)).await;
    ///         cancel_token.cancel();
    ///     });
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn cancellation_token(&self) -> Option<tokio_util::sync::CancellationToken> {
        self.execution_config.event_loop.cancellation_token.clone()
    }

    /// Reset the cancellation token with a fresh one
    ///
    /// This is useful when recovering from a cancelled state. The agent's
    /// conversation history is preserved, only the cancellation token is replaced.
    /// Returns the new token for external tracking.
    pub fn reset_cancellation_token(&mut self) -> tokio_util::sync::CancellationToken {
        let new_token = tokio_util::sync::CancellationToken::new();
        self.execution_config.event_loop.cancellation_token = Some(new_token.clone());
        new_token
    }

    /// Create an AgentContext from this agent for parent-child tracking
    pub fn create_context(&self, agent_type: impl Into<String>) -> AgentContext {
        AgentContext::from_agent(self, agent_type)
    }

    pub fn provider(&self) -> &Arc<dyn LlmProvider> {
        &self.provider
    }

    pub fn model(&self) -> &dyn LlmModel {
        self.model.as_ref()
    }

    pub fn conversation(&self) -> &ConversationManager {
        &self.conversation
    }

    pub fn conversation_mut(&mut self) -> &mut ConversationManager {
        &mut self.conversation
    }

    pub fn tool_registry(&self) -> &ToolRegistry {
        &self.tool_registry
    }

    pub fn conversation_history(&self) -> &[Message] {
        &self.conversation.messages().messages
    }

    pub fn clear_history(&mut self) {
        self.conversation.clear();
    }

    pub fn add_user_message<S: Into<String>>(&mut self, text: S) {
        self.conversation.add_user_message(text);
    }

    pub fn add_assistant_message<S: Into<String>>(&mut self, text: S) {
        self.conversation.add_assistant_message(text);
    }

    /// Execute a task using the unified agent interface.
    ///
    /// This is the ONLY execution method - always agentic, always powerful, with Python-like simplicity.
    /// All configuration (callbacks, streaming, timeouts) is set during Agent construction via the builder.
    ///
    /// # Arguments
    ///
    /// * `prompt` - Your task, question, or instruction for the agent
    ///
    /// # Returns
    ///
    /// An [`AgentResult`] containing:
    /// - `response` - The final answer or result (accessible via Display trait)
    /// - `execution` - Detailed metrics and execution information
    /// - `tools_called` - List of tools that were used
    /// - `duration` - Total execution time
    /// - `success` - Whether execution completed successfully
    ///
    /// # Examples
    ///
    /// Simple usage (Python-like):
    /// ```no_run
    /// # use stood::agent::Agent;
    /// # async fn example(mut agent: Agent) -> Result<(), Box<dyn std::error::Error>> {
    /// let result = agent.execute("Tell me a joke").await?;
    /// println!("{}", result); // Prints just the response text
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Detailed analysis:
    /// ```no_run
    /// # use stood::agent::Agent;
    /// # async fn example(mut agent: Agent) -> Result<(), Box<dyn std::error::Error>> {
    /// let result = agent.execute("Analyze this complex data").await?;
    ///
    /// println!("Response: {}", result.response);
    /// println!("Used {} tools in {} cycles", result.tools_called.len(), result.execution.cycles);
    /// println!("Execution took {:?}", result.duration);
    ///
    /// if result.used_tools {
    ///     println!("Tools used: {}", result.tools_called.join(", "));
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// With callbacks configured at build time:
    /// ```no_run
    /// # use stood::agent::Agent;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut agent = Agent::builder()
    ///     .with_printing_callbacks()  // Callbacks configured here
    ///     .build().await?;
    ///
    /// let result = agent.execute("Complex task").await?; // Uses configured callbacks
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Architecture
    ///
    /// This method always uses EventLoop orchestration for consistent, powerful execution:
    /// 1. **Analysis Phase** - Agent analyzes the task and creates a plan
    /// 2. **Tool Selection** - Identifies relevant tools for each step
    /// 3. **Execution Cycles** - Iteratively executes tools and reasons about results
    /// 4. **Synthesis** - Combines results into a coherent final response
    /// 5. **Callback Integration** - Real-time updates via pre-configured handlers
    ///
    /// # Performance Characteristics
    ///
    /// - **Simple tasks** - 1-5 seconds, single model interaction, minimal tool use
    /// - **Complex tasks** - 10 seconds to several minutes, multiple model interactions, extensive tool use
    /// - **Token Usage** - Optimized based on task complexity
    /// - **Memory** - Conversation context grows linearly with message count
    ///
    /// # Configuration
    ///
    /// All execution behavior is configured during Agent construction:
    /// - Callbacks via `Agent::builder().with_printing_callbacks().build()`
    /// - Timeouts via execution_config
    /// - Streaming via execution_config
    /// - EventLoop settings via execution_config
    ///
    /// # Errors
    ///
    /// Returns [`StoodError`] variants:
    /// - `ToolError` - Tool execution failures or timeouts
    /// - `ModelError` - AWS Bedrock API failures during reasoning
    /// - `ConversationError` - Context management issues
    /// - `InvalidInput` - Empty prompts or invalid parameters
    pub async fn execute<S: Into<String>>(&mut self, prompt: S) -> Result<AgentResult> {
        let prompt = prompt.into();
        let start_time = std::time::Instant::now();

        // Use pre-configured ExecutionConfig from Agent construction
        let config = &self.execution_config;

        // Create EventLoop with pre-configured settings - EventLoop OWNS a copy of the Agent
        // Create a new model instance based on the configuration
        tracing::debug!(
            target: "stood::agent",
            "DEBUG [EXECUTE-1]: About to call create_model_from_config(provider={:?}, model_id={})",
            &self.config.provider,
            &self.config.model_id
        );

        let model = create_model_from_config(&self.config.provider, &self.config.model_id);

        tracing::debug!(
            target: "stood::agent",
            "DEBUG [EXECUTE-2]: create_model_from_config returned model with id: {}",
            model.model_id()
        );

        let agent_copy = Agent::build_internal(
            Arc::clone(&self.provider),
            model,
            self.config.clone(),
            vec![], // Tools are already in the main agent's registry
            vec![], // Middleware is already in the main agent's registry
            config.clone(),
            Some(self.agent_id.clone()),
            self.agent_name.clone(),
        )
        .await?;

        // Copy current conversation state to the EventLoop agent
        let mut event_loop_agent = agent_copy;
        for message in self.conversation.messages().messages.iter() {
            match message.role {
                crate::types::MessageRole::User => {
                    if let Some(text) = message.text() {
                        event_loop_agent.add_user_message(text);
                    }
                }
                crate::types::MessageRole::Assistant => {
                    if let Some(text) = message.text() {
                        event_loop_agent.add_assistant_message(text);
                    }
                }
                _ => {} // Skip system messages as they're handled separately
            }
        }

        // Copy system prompt
        if let Some(system_prompt) = self.conversation.system_prompt() {
            event_loop_agent
                .conversation_mut()
                .set_system_prompt(Some(system_prompt.to_string()));
        }

        // Create callback handler from configuration
        let callback_handler = match &config.callback_handler {
            CallbackHandlerConfig::None => None,
            other_config => Some(Self::create_callback_handler(other_config)?),
        };

        // EventLoop orchestrates everything using pre-configured settings with callbacks
        // Apply ExecutionConfig.streaming to EventLoopConfig.enable_streaming
        let mut event_loop_config = config.event_loop.clone();
        event_loop_config.enable_streaming = config.streaming;

        tracing::info!(
            "🔧 Agent streaming config: {}, EventLoop streaming config: {}",
            config.streaming,
            event_loop_config.enable_streaming
        );

        // Enable telemetry in EventLoop if configured at agent level

        {
            event_loop_config.enable_telemetry = self.config.telemetry_config.is_some();
        }

        let mut event_loop = event_loop::EventLoop::new_with_callbacks(
            event_loop_agent,
            self.tool_registry.clone(),
            event_loop_config,
            callback_handler,
        )?;

        let event_loop_result = event_loop.execute(prompt).await?;

        // Convert to unified result type
        let agent_result = AgentResult::from(event_loop_result, start_time.elapsed());

        // Sync conversation state from EventLoop result
        self.sync_conversation_from_eventloop(event_loop.agent());

        Ok(agent_result)
    }

    /// Check if the agent has access to tools for agentic execution
    pub fn supports_agentic_execution(&self) -> bool {
        // For now, all agents support agentic execution
        // In the future, this might depend on model capabilities
        true
    }

    /// Get agent performance metrics summary
    pub fn get_performance_summary(&self) -> AgentPerformanceSummary {
        AgentPerformanceSummary {
            total_messages: self.conversation.message_count(),
            conversation_length: self.conversation.messages().messages.len(),
            has_system_prompt: self.conversation.system_prompt().is_some(),
            provider: self.config.provider,
            model_id: self.config.model_id.clone(),
        }
    }

    /// Sync conversation state from EventLoop agent after execution
    fn sync_conversation_from_eventloop(&mut self, eventloop_agent: &Agent) {
        // Replace our conversation with the EventLoop agent's final conversation state
        self.conversation = eventloop_agent.conversation.clone();
    }

    /// Create callback handler from configuration
    fn create_callback_handler(config: &CallbackHandlerConfig) -> Result<Arc<dyn CallbackHandler>> {
        match config {
            CallbackHandlerConfig::None => Ok(Arc::new(NullCallbackHandler)),
            CallbackHandlerConfig::Printing(print_config) => {
                Ok(Arc::new(PrintingCallbackHandler::new(print_config.clone())))
            }
            CallbackHandlerConfig::Custom(handler) => Ok(handler.clone()),
            CallbackHandlerConfig::Composite(handlers) => {
                let mut handler_arcs = Vec::new();
                for handler_config in handlers {
                    let handler = Self::create_callback_handler(handler_config)?;
                    handler_arcs.push(handler);
                }
                Ok(Arc::new(CompositeCallbackHandler::with_handlers(
                    handler_arcs,
                )))
            }
            CallbackHandlerConfig::Performance(level) => {
                Ok(Arc::new(PerformanceCallbackHandler::new(*level)))
            }
            CallbackHandlerConfig::Batching {
                inner,
                batch_config,
            } => {
                let inner_handler = Self::create_callback_handler(inner)?;
                Ok(Arc::new(
                    crate::agent::callbacks::BatchingCallbackHandler::new(
                        inner_handler,
                        batch_config.clone(),
                    ),
                ))
            }
        }
    }
}

/// Builder for configuring and creating [`Agent`] instances.
///
/// This builder provides a fluent interface for setting up agents with
/// custom models, parameters, and configurations. You'll get validation
/// of settings and helpful error messages for common misconfigurations.
///
/// # Examples
///
/// Build a basic agent:
/// ```no_run
/// # use stood::agent::Agent;
/// # use stood::llm::models::Bedrock;
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let agent = Agent::builder()
///     .model(Bedrock::ClaudeHaiku45)
///     .temperature(0.7)
///     .build().await?;
/// # Ok(())
/// # }
/// ```
///
/// Build an agent optimized for code generation:
/// ```no_run
/// # use stood::agent::Agent;
/// # use stood::llm::models::Bedrock;
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let agent = Agent::builder()
///     .model(Bedrock::ClaudeSonnet45)     // Better for code
///     .temperature(0.1)                   // More deterministic
///     .max_tokens(8192)                   // Longer code responses
///     .system_prompt(
///         "You are an expert software engineer. Write clean,
///          well-documented code with comprehensive error handling."
///     )
///     .build().await?;
/// # Ok(())
/// # }
/// ```
///
/// # Validation
///
/// The builder validates parameters during construction:
/// - Temperature must be between 0.0 and 1.0
/// - Max tokens must be greater than 0
/// - Model must be supported by the selected provider
/// - Provider must be properly configured
pub struct AgentBuilder {
    config: AgentConfig,
    tools: Vec<Box<dyn Tool>>,
    execution_config: ExecutionConfig,
    model: Option<Box<dyn LlmModel>>,
    agent_id: Option<String>,
    agent_name: Option<String>,
    aws_credentials: Option<AwsCredentials>,
    middlewares: Vec<Arc<dyn ToolMiddleware>>,
}

/// AWS credentials for programmatic authentication
#[derive(Debug, Clone)]
pub struct AwsCredentials {
    pub access_key: String,
    pub secret_key: String,
    pub session_token: Option<String>,
    pub region: Option<String>,
}

impl AgentBuilder {
    pub fn new() -> Self {
        Self {
            config: AgentConfig::default(),
            tools: Vec::new(),
            execution_config: ExecutionConfig::default(),
            model: None,
            agent_id: None,
            agent_name: None,
            aws_credentials: None,
            middlewares: Vec::new(),
        }
    }

    /// Set model using LLM provider system
    ///
    /// # Examples
    /// ```no_run
    /// use stood::agent::Agent;
    /// use stood::llm::models::{Bedrock, LMStudio};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// // Use Bedrock (costs money)
    /// let agent = Agent::builder()
    ///     .model(Bedrock::ClaudeSonnet45)
    ///     .build()
    ///     .await?;
    ///
    /// // Use LM Studio (local)
    /// let agent = Agent::builder()
    ///     .model(LMStudio::Gemma3_12B)
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn model<M: LlmModel + 'static>(mut self, model: M) -> Self {
        // Store the model info in the config
        self.config.provider = model.provider();
        self.config.model_id = model.model_id().to_string();

        // Store the model instance
        self.model = Some(Box::new(model));

        // DEBUG: Log that model was set
        tracing::debug!(
            target: "stood::agent::builder",
            "DEBUG [STOOD-1]: model() called, set model_id={}, provider={:?}",
            self.config.model_id,
            self.config.provider
        );

        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        if !(0.0..=1.0).contains(&temperature) {
            panic!("Temperature must be between 0.0 and 1.0");
        }
        self.config.temperature = Some(temperature);
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        if max_tokens == 0 {
            panic!("Max tokens must be greater than 0");
        }
        self.config.max_tokens = Some(max_tokens);
        self
    }

    pub fn system_prompt<S: Into<String>>(mut self, prompt: S) -> Self {
        self.config.system_prompt = Some(prompt.into());
        self
    }

    pub fn name<S: Into<String>>(mut self, name: S) -> Self {
        self.agent_name = Some(name.into());
        self
    }

    pub fn with_id<S: Into<String>>(mut self, id: S) -> Self {
        self.agent_id = Some(id.into());
        self
    }

    /// Configure retry behavior for LM Studio provider
    ///
    /// This setting applies to LM Studio providers to handle model loading delays.
    /// Other providers use their own retry strategies.
    ///
    /// # Examples
    /// ```no_run
    /// use stood::agent::Agent;
    /// use stood::llm::models::LMStudio;
    /// use stood::llm::providers::retry::RetryConfig;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let agent = Agent::builder()
    ///     .model(LMStudio::TessaRust7B)
    ///     .with_retry_config(RetryConfig::lm_studio_aggressive())
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_retry_config(mut self, config: RetryConfig) -> Self {
        self.config.retry_config = Some(config);
        self
    }

    /// Enable conservative retry behavior (2 attempts max)
    pub fn with_conservative_retry(mut self) -> Self {
        self.config.retry_config = Some(RetryConfig::lm_studio_conservative());
        self
    }

    /// Enable aggressive retry behavior (5 attempts max)
    pub fn with_aggressive_retry(mut self) -> Self {
        self.config.retry_config = Some(RetryConfig::lm_studio_aggressive());
        self
    }

    /// Disable retry behavior entirely
    pub fn without_retry(mut self) -> Self {
        self.config.retry_config = Some(RetryConfig::disabled());
        self
    }

    /// Enable prompt caching to reduce latency and costs
    ///
    /// Prompt caching can reduce latency by up to 85% and costs by up to 90%
    /// for repeated prompts by caching frequently used content across API calls.
    ///
    /// # Supported Models
    /// - **Claude models**: Full support (system prompt + tool definitions)
    /// - **Nova models**: Partial support (system prompt only, no tool caching)
    /// - **Mistral models**: Not supported (caching will be ignored)
    ///
    /// # Cache TTL
    /// The cache has a 5-minute TTL that resets with each successful cache hit.
    ///
    /// # Examples
    /// ```no_run
    /// use stood::agent::Agent;
    /// use stood::llm::models::Bedrock;
    /// use stood::CacheStrategy;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// // Cache system prompt only (most common use case)
    /// let agent = Agent::builder()
    ///     .model(Bedrock::ClaudeSonnet45)
    ///     .system_prompt("You are a helpful coding assistant...")
    ///     .with_prompt_caching(CacheStrategy::SystemOnly)
    ///     .build()
    ///     .await?;
    ///
    /// // Cache system prompt and tool definitions
    /// let agent = Agent::builder()
    ///     .model(Bedrock::ClaudeSonnet45)
    ///     .with_builtin_tools()
    ///     .with_prompt_caching(CacheStrategy::SystemAndTools)
    ///     .build()
    ///     .await?;
    ///
    /// // Automatic cache placement
    /// let agent = Agent::builder()
    ///     .model(Bedrock::ClaudeHaiku45)
    ///     .with_prompt_caching(CacheStrategy::Auto)
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_prompt_caching(mut self, strategy: CacheStrategy) -> Self {
        self.config.cache_strategy = strategy;
        self
    }

    /// Enable system-only prompt caching (convenience method)
    ///
    /// This is equivalent to `.with_prompt_caching(CacheStrategy::SystemOnly)`.
    /// Caches the system prompt to reduce costs on repeated queries with the
    /// same instructions.
    pub fn with_system_caching(mut self) -> Self {
        self.config.cache_strategy = CacheStrategy::SystemOnly;
        self
    }

    /// Enable full prompt caching including tools (convenience method)
    ///
    /// This is equivalent to `.with_prompt_caching(CacheStrategy::SystemAndTools)`.
    /// Caches both the system prompt and tool definitions.
    ///
    /// **Note**: Tool caching is only supported by Claude models. Nova models
    /// will automatically fall back to system-only caching.
    pub fn with_full_caching(mut self) -> Self {
        self.config.cache_strategy = CacheStrategy::SystemAndTools;
        self
    }

    // Client method removed - now using LLM provider system

    /// Add a single tool to the agent
    pub fn tool(mut self, tool: Box<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Add multiple tools to the agent
    pub fn tools(mut self, mut tools: Vec<Box<dyn Tool>>) -> Self {
        self.tools.append(&mut tools);
        self
    }

    /// Add all built-in tools to the agent
    pub fn with_builtin_tools(mut self) -> Self {
        // We'll implement this by creating builtin tools
        let builtin_tools = vec![
            Box::new(crate::tools::builtin::CalculatorTool::new()) as Box<dyn Tool>,
            Box::new(crate::tools::builtin::FileReadTool::new()) as Box<dyn Tool>,
            Box::new(crate::tools::builtin::FileWriteTool::new()) as Box<dyn Tool>,
            Box::new(crate::tools::builtin::FileListTool::new()) as Box<dyn Tool>,
            Box::new(crate::tools::builtin::HttpRequestTool::new()) as Box<dyn Tool>,
            Box::new(crate::tools::builtin::CurrentTimeTool::new()) as Box<dyn Tool>,
            Box::new(crate::tools::builtin::EnvVarTool::new()) as Box<dyn Tool>,
        ];
        self.tools.extend(builtin_tools);
        self
    }

    /// Add tool middleware to the agent.
    ///
    /// Middleware intercepts tool execution, allowing you to:
    /// - Modify tool parameters before execution
    /// - Abort or skip tool calls
    /// - Modify or augment tool results
    /// - Inject additional context after tool execution
    ///
    /// Middleware is executed in registration order for `before_tool`
    /// and reverse order for `after_tool`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use stood::agent::Agent;
    /// use stood::tools::{ToolMiddleware, ToolMiddlewareAction, AfterToolAction, ToolContext, ToolResult};
    /// use async_trait::async_trait;
    /// use serde_json::Value;
    /// use std::sync::Arc;
    ///
    /// #[derive(Debug)]
    /// struct LoggingMiddleware;
    ///
    /// #[async_trait]
    /// impl ToolMiddleware for LoggingMiddleware {
    ///     async fn before_tool(&self, tool_name: &str, params: &Value, _ctx: &ToolContext) -> ToolMiddlewareAction {
    ///         println!("Executing tool: {}", tool_name);
    ///         ToolMiddlewareAction::Continue
    ///     }
    ///
    ///     async fn after_tool(&self, tool_name: &str, result: &ToolResult, _ctx: &ToolContext) -> AfterToolAction {
    ///         println!("Tool {} completed", tool_name);
    ///         AfterToolAction::PassThrough
    ///     }
    /// }
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let agent = Agent::builder()
    ///         .with_middleware(Arc::new(LoggingMiddleware))
    ///         .build()
    ///         .await?;
    ///     Ok(())
    /// }
    /// ```
    pub fn with_middleware(mut self, middleware: Arc<dyn ToolMiddleware>) -> Self {
        self.middlewares.push(middleware);
        self
    }

    /// Add a think tool with custom prompt for structured problem-solving
    ///
    /// The think tool provides structured thinking guidance based on Anthropic's research.
    /// It helps the model work through complex problems step by step.
    ///
    /// # Arguments
    /// * `prompt` - Custom thinking prompt for domain-specific reasoning
    ///
    /// # Examples
    /// ```rust
    /// use stood::agent::Agent;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// // Use custom prompt for legal analysis
    /// let agent = Agent::builder()
    ///     .with_think_tool("Think through this legal case step by step, considering precedent and applicable law.")
    ///     .build().await?;
    ///
    /// // Use custom prompt for software architecture
    /// let agent = Agent::builder()
    ///     .with_think_tool("Analyze this software architecture problem considering scalability, maintainability, and performance.")
    ///     .build().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_think_tool<S: Into<String>>(mut self, custom_prompt: S) -> Self {
        let think_tool =
            Box::new(crate::tools::builtin::ThinkTool::new(custom_prompt.into())) as Box<dyn Tool>;
        self.tools.push(think_tool);
        self
    }

    /// Add tools from an MCP client (matches Python's simple approach)
    ///
    /// This method automatically connects to the MCP server, lists available tools,
    /// and creates tool adapters with the specified namespace prefix.
    ///
    /// # Arguments
    /// * `mcp_client` - An MCPClient instance configured with transport
    /// * `namespace` - Optional namespace prefix for tool names (e.g., "mcp_")
    ///
    /// # Examples
    /// ```rust
    /// use stood::agent::Agent;
    /// use stood::mcp::{MCPClient, MCPClientConfig};
    /// use stood::mcp::transport::{TransportFactory, StdioConfig};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// // Create MCP client
    /// let config = StdioConfig {
    ///     command: "python".to_string(),
    ///     args: vec!["-m", "my_mcp_server"].iter().map(|s| s.to_string()).collect(),
    ///     working_dir: None,
    ///     env_vars: std::collections::HashMap::new(),
    ///     startup_timeout_ms: 30000,
    ///     max_message_size: Some(1024 * 1024),
    /// };
    /// let transport = TransportFactory::stdio(config);
    /// let mut mcp_client = MCPClient::new(MCPClientConfig::default(), transport);
    ///
    /// // Connect and create agent with MCP tools
    /// mcp_client.connect().await?;
    /// let agent = Agent::builder()
    ///     .with_mcp_client(mcp_client, Some("mcp_".to_string()))
    ///     .build().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn with_mcp_client(
        mut self,
        mcp_client: crate::mcp::client::MCPClient,
        namespace: Option<String>,
    ) -> Result<Self> {
        use crate::tools::mcp_adapter::MCPAgentTool;
        use std::sync::Arc;
        use tokio::sync::RwLock;

        // Ensure client is connected
        if mcp_client.session_info().await.is_err() {
            return Err(crate::StoodError::configuration_error(
                "MCP client must be connected before adding to agent. Call mcp_client.connect().await first."
            ));
        }

        // List tools from the server
        let tools = mcp_client.list_tools().await.map_err(|e| {
            crate::StoodError::configuration_error(format!("Failed to list MCP tools: {}", e))
        })?;

        // Create tool adapters
        let mcp_client_arc = Arc::new(RwLock::new(mcp_client));

        for tool in tools {
            let mcp_tool = MCPAgentTool::new(tool, mcp_client_arc.clone(), namespace.clone());
            self.tools.push(Box::new(mcp_tool));
        }

        Ok(self)
    }

    /// Add tools from multiple MCP clients (matches Python's multi-client approach)
    ///
    /// This method allows connecting to multiple MCP servers with different namespace
    /// prefixes to avoid tool name conflicts.
    ///
    /// # Arguments
    /// * `mcp_clients` - Vector of (MCPClient, namespace) pairs
    ///
    /// # Examples
    /// ```rust
    /// use stood::agent::Agent;
    /// use stood::mcp::{MCPClient, MCPClientConfig};
    /// use stood::mcp::transport::{TransportFactory, StdioConfig, WebSocketConfig};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// // Create multiple MCP clients
    /// let stdio_config = StdioConfig { /* ... */ };
    /// let mut stdio_client = MCPClient::new(MCPClientConfig::default(),
    ///     TransportFactory::stdio(stdio_config));
    /// stdio_client.connect().await?;
    ///
    /// let ws_config = WebSocketConfig {
    ///     url: "ws://localhost:8080/mcp".to_string(),
    ///     /* ... */
    /// };
    /// let mut ws_client = MCPClient::new(MCPClientConfig::default(),
    ///     TransportFactory::websocket(ws_config));
    /// ws_client.connect().await?;
    ///
    /// // Create agent with tools from both servers
    /// let agent = Agent::builder()
    ///     .with_mcp_clients(vec![
    ///         (stdio_client, Some("local_".to_string())),
    ///         (ws_client, Some("remote_".to_string())),
    ///     ])
    ///     .await?
    ///     .build().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn with_mcp_clients(
        mut self,
        mcp_clients: Vec<(crate::mcp::client::MCPClient, Option<String>)>,
    ) -> Result<Self> {
        use crate::tools::mcp_adapter::MCPAgentTool;
        use std::sync::Arc;
        use tokio::sync::RwLock;

        for (mcp_client, namespace) in mcp_clients {
            // Ensure client is connected
            if mcp_client.session_info().await.is_err() {
                return Err(crate::StoodError::configuration_error(
                    "All MCP clients must be connected before adding to agent. Call mcp_client.connect().await first."
                ));
            }

            // List tools from the server
            let tools = mcp_client.list_tools().await.map_err(|e| {
                crate::StoodError::configuration_error(format!("Failed to list MCP tools: {}", e))
            })?;

            // Create tool adapters
            let mcp_client_arc = Arc::new(RwLock::new(mcp_client));

            for tool in tools {
                let mcp_tool = MCPAgentTool::new(tool, mcp_client_arc.clone(), namespace.clone());
                self.tools.push(Box::new(mcp_tool));
            }
        }

        Ok(self)
    }

    /// Enable telemetry with explicit configuration
    ///
    /// Telemetry is **disabled by default**. Use this method to enable it with custom settings.
    ///
    /// # Example
    /// ```rust
    /// # use stood::agent::Agent;
    /// # use stood::telemetry::TelemetryConfig;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let config = TelemetryConfig::default()
    ///     .with_otlp_endpoint("http://localhost:4318")
    ///     .enable();
    ///
    /// let agent = Agent::builder()
    ///     .with_telemetry(config)
    ///     .build().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_telemetry(mut self, telemetry_config: TelemetryConfig) -> Self {
        self.config.telemetry_config = Some(telemetry_config);
        self
    }

    /// Enable telemetry with environment-based configuration
    ///
    /// This enables telemetry with smart auto-detection of OTLP endpoints and reads configuration
    /// from environment variables. **Telemetry is disabled by default** - this method explicitly enables it.
    ///
    /// Environment variables:
    /// - `OTEL_ENABLED=false` - Disable telemetry entirely
    /// - `OTEL_EXPORTER_OTLP_ENDPOINT` - Custom OTLP endpoint
    /// - `OTEL_SERVICE_NAME` - Service name for traces
    ///
    /// # Example
    /// ```bash
    /// export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
    /// export OTEL_SERVICE_NAME=my-stood-agent
    /// ```
    ///
    /// ```rust
    /// # use stood::agent::Agent;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let agent = Agent::builder()
    ///     .with_telemetry_from_env()  // Explicitly enable with env config
    ///     .build().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_telemetry_from_env(mut self) -> Self {
        let mut telemetry_config = TelemetryConfig::from_env();
        // If log level is already set in execution config, apply it to telemetry
        let telemetry_log_level = match self.execution_config.log_level {
            crate::agent::config::LogLevel::Off => crate::telemetry::LogLevel::OFF,
            crate::agent::config::LogLevel::Info => crate::telemetry::LogLevel::INFO,
            crate::agent::config::LogLevel::Debug => crate::telemetry::LogLevel::DEBUG,
            crate::agent::config::LogLevel::Trace => crate::telemetry::LogLevel::TRACE,
        };
        telemetry_config.set_log_level(telemetry_log_level);
        self.config.telemetry_config = Some(telemetry_config);
        self
    }

    /// Enable comprehensive metrics collection with smart telemetry auto-detection
    pub fn with_metrics(mut self) -> Self {
        let mut telemetry_config = TelemetryConfig::from_env();
        // If log level is already set in execution config, apply it to telemetry
        let telemetry_log_level = match self.execution_config.log_level {
            crate::agent::config::LogLevel::Off => crate::telemetry::LogLevel::OFF,
            crate::agent::config::LogLevel::Info => crate::telemetry::LogLevel::INFO,
            crate::agent::config::LogLevel::Debug => crate::telemetry::LogLevel::DEBUG,
            crate::agent::config::LogLevel::Trace => crate::telemetry::LogLevel::TRACE,
        };
        telemetry_config.set_log_level(telemetry_log_level);
        self.config.telemetry_config = Some(telemetry_config);
        self
    }

    /// Enable metrics collection with custom telemetry configuration
    pub fn with_metrics_config(mut self, telemetry_config: TelemetryConfig) -> Self {
        self.config.telemetry_config = Some(telemetry_config);
        self
    }

    /// Enable printing callbacks with default settings (matches Python's PrintingCallbackHandler)
    pub fn with_printing_callbacks(mut self) -> Self {
        self.execution_config.callback_handler =
            CallbackHandlerConfig::Printing(PrintingConfig::default());
        self
    }

    /// Enable printing callbacks with custom settings
    pub fn with_printing_callbacks_config(mut self, config: PrintingConfig) -> Self {
        self.execution_config.callback_handler = CallbackHandlerConfig::Printing(config);
        self
    }

    /// Set custom callback handler
    pub fn with_callback_handler<H: CallbackHandler + 'static>(mut self, handler: H) -> Self {
        use std::sync::Arc;
        self.execution_config.callback_handler = CallbackHandlerConfig::Custom(Arc::new(handler));
        self
    }

    /// Enable verbose printing (development mode)
    pub fn with_verbose_callbacks(mut self) -> Self {
        self.execution_config.callback_handler =
            CallbackHandlerConfig::Printing(PrintingConfig::verbose());
        self
    }

    /// Enable performance logging callbacks
    pub fn with_performance_callbacks(mut self, level: tracing::Level) -> Self {
        self.execution_config.callback_handler = CallbackHandlerConfig::Performance(level);
        self
    }

    /// Enable batching wrapper around printing callbacks for better performance
    pub fn with_batched_printing_callbacks(mut self) -> Self {
        self.execution_config.callback_handler = CallbackHandlerConfig::Batching {
            inner: Box::new(CallbackHandlerConfig::Printing(PrintingConfig::default())),
            batch_config: crate::agent::callbacks::BatchConfig::default(),
        };
        self
    }

    /// Enable batching wrapper with custom configuration
    pub fn with_batched_callbacks(
        mut self,
        inner_config: CallbackHandlerConfig,
        batch_config: crate::agent::callbacks::BatchConfig,
    ) -> Self {
        self.execution_config.callback_handler = CallbackHandlerConfig::Batching {
            inner: Box::new(inner_config),
            batch_config,
        };
        self
    }

    /// Add multiple callback handlers (matches Python's CompositeCallbackHandler)
    pub fn with_composite_callbacks(mut self, configs: Vec<CallbackHandlerConfig>) -> Self {
        self.execution_config.callback_handler = CallbackHandlerConfig::Composite(configs);
        self
    }

    /// Enable streaming by default
    pub fn with_streaming(mut self, enabled: bool) -> Self {
        self.execution_config.streaming = enabled;
        self
    }

    /// Set default timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.execution_config.timeout = Some(timeout);
        self
    }

    /// Configure EventLoop settings
    pub fn with_event_loop_config(mut self, config: EventLoopConfig) -> Self {
        self.execution_config.event_loop = config;
        self
    }

    /// Configure execution settings directly
    pub fn with_execution_config(mut self, config: ExecutionConfig) -> Self {
        self.execution_config = config;
        self
    }

    /// Set the log level for debug output from the agent
    pub fn with_log_level(mut self, level: LogLevel) -> Self {
        self.execution_config.log_level = level;
        self
    }

    /// Configure parallel tool execution (matches reference-python max_parallel_tools)
    ///
    /// This is the primary method for configuring parallel execution, following
    /// the same pattern as the reference Python implementation.
    ///
    /// # Arguments
    /// * `max_parallel_tools` - Maximum number of tools to execute concurrently
    ///   - `1` = Sequential execution (tools run one at a time)
    ///   - `> 1` = Parallel execution (up to N tools run concurrently)
    ///
    /// # Examples
    /// ```rust
    /// // Sequential execution (like Python's max_parallel_tools=1)
    /// let agent = Agent::builder().max_parallel_tools(1).build().await?;
    ///
    /// // Parallel execution (like Python's max_parallel_tools=4)
    /// let agent = Agent::builder().max_parallel_tools(4).build().await?;
    /// ```
    pub fn max_parallel_tools(mut self, max_parallel: usize) -> Self {
        // Create ExecutorConfig with the specified parallelism
        let mut tool_config = match crate::tools::executor::ExecutorConfig::new(max_parallel) {
            Ok(config) => config,
            Err(_) => {
                // Fallback to sequential if invalid value
                crate::tools::executor::ExecutorConfig::sequential()
            }
        };

        // Enable parallel execution strategy when max_parallel > 1
        if max_parallel > 1 {
            tool_config.execution_strategy = crate::tools::executor::ExecutionStrategy::Parallel;
        }

        // Update the EventLoopConfig with the new tool configuration
        self.execution_config.event_loop.tool_config = tool_config;

        self
    }

    /// Use CPU count for parallel execution (matches reference-python max_parallel_tools=None)
    ///
    /// This sets max_parallel_tools to the number of CPU cores, providing
    /// optimal parallelism for CPU-bound tasks.
    pub fn max_parallel_tools_auto(self) -> Self {
        let cpu_count = num_cpus::get();
        self.max_parallel_tools(cpu_count)
    }

    /// Configure sequential execution (shorthand for max_parallel_tools(1))
    ///
    /// This is equivalent to `max_parallel_tools(1)` but more explicit about
    /// the intent to run tools sequentially.
    pub fn sequential_execution(self) -> Self {
        self.max_parallel_tools(1)
    }

    /// Enable task evaluation strategy with custom prompt
    ///
    /// The agent will evaluate whether it has fully satisfied the user's request and intent
    /// after each cycle to determine if it should continue working. This allows for
    /// thoughtful, iterative problem-solving focused on user satisfaction.
    ///
    /// # Arguments
    /// * `prompt` - The prompt to use for task evaluation (e.g., "Have I fully satisfied the user's request?")
    ///
    /// # Examples
    /// ```rust
    /// use stood::agent::Agent;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let agent = Agent::builder()
    ///     .with_task_evaluation("Have I fully addressed all aspects of the user's request? Is there anything important missing?")
    ///     .build().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_task_evaluation(mut self, prompt: impl Into<String>) -> Self {
        self.execution_config.event_loop.evaluation_strategy =
            EvaluationStrategy::task_evaluation(prompt);
        self
    }

    /// Enable multi-perspective evaluation strategy
    ///
    /// The agent will evaluate task completion from multiple perspectives,
    /// combining different viewpoints to make a more informed continuation decision.
    ///
    /// # Arguments
    /// * `perspectives` - List of perspectives to evaluate, each with a name, prompt, and weight
    ///
    /// # Examples
    /// ```rust
    /// use stood::agent::{Agent, PerspectiveConfig};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let perspectives = vec![
    ///     PerspectiveConfig {
    ///         name: "quality_check".to_string(),
    ///         prompt: "As a quality analyst, is the work complete and high-quality?".to_string(),
    ///         weight: 0.6,
    ///     },
    ///     PerspectiveConfig {
    ///         name: "user_satisfaction".to_string(),
    ///         prompt: "From a user's perspective, does this fully address their needs?".to_string(),
    ///         weight: 0.4,
    ///     },
    /// ];
    ///
    /// let agent = Agent::builder()
    ///     .with_multi_perspective_evaluation(perspectives)
    ///     .build().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_multi_perspective_evaluation(
        mut self,
        perspectives: Vec<PerspectiveConfig>,
    ) -> Self {
        self.execution_config.event_loop.evaluation_strategy =
            EvaluationStrategy::multi_perspective(perspectives);
        self
    }

    /// Enable agent-based evaluation strategy
    ///
    /// A separate evaluator agent will assess the main agent's work to determine
    /// if the task is complete. This allows for specialized evaluation with
    /// different models, prompts, or configurations.
    ///
    /// # Arguments
    /// * `evaluator_agent` - The agent instance to use for evaluation
    ///
    /// # Examples
    /// ```rust
    /// use stood::agent::Agent;
    /// use stood::llm::models::Bedrock;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let evaluator = Agent::builder()
    ///     .model(Bedrock::ClaudeHaiku45)
    ///     .system_prompt("You are a critical evaluator. Assess task completion quality.")
    ///     .build().await?;
    ///
    /// let main_agent = Agent::builder()
    ///     .with_agent_based_evaluation(evaluator)
    ///     .build().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_agent_based_evaluation(mut self, evaluator_agent: Agent) -> Self {
        let evaluation_prompt =
            "Please evaluate if the previous agent's work fully completes the user's request."
                .to_string();
        self.execution_config.event_loop.evaluation_strategy =
            EvaluationStrategy::agent_based(evaluator_agent, evaluation_prompt);
        self
    }

    /// Set a high limit for tool iterations to enable more autonomous behavior
    ///
    /// This increases the maximum number of tool execution rounds per cycle,
    /// allowing agents to perform more complex, multi-step tasks without
    /// hitting artificial limits.
    ///
    /// # Arguments
    /// * `limit` - Maximum number of tool iterations (default: 7)
    ///
    /// # Examples
    /// ```rust
    /// use stood::agent::Agent;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let agent = Agent::builder()
    ///     .with_self_reflection_evaluation("Have I completed the task?")
    ///     .with_high_tool_limit(50)  // Allow up to 50 tool iterations
    ///     .build().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_high_tool_limit(mut self, limit: u32) -> Self {
        self.execution_config.event_loop.max_tool_iterations = limit;
        self
    }

    /// Enable cancellation support for the agent execution
    ///
    /// Creates an internal cancellation token that can be retrieved after building
    /// the agent. When cancelled, the agent will stop execution immediately,
    /// bypassing any task evaluation and returning with a cancellation error.
    ///
    /// # Examples
    /// ```no_run
    /// use stood::agent::Agent;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let agent = Agent::builder()
    ///     .with_cancellation()
    ///     .build().await?;
    ///
    /// // Get the cancellation token from the agent
    /// let cancel_token = agent.cancellation_token().unwrap();
    ///
    /// // In another task, cancel execution
    /// tokio::spawn(async move {
    ///     tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    ///     cancel_token.cancel();
    /// });
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_cancellation(mut self) -> Self {
        let token = tokio_util::sync::CancellationToken::new();
        self.execution_config.event_loop.cancellation_token = Some(token);
        self
    }

    /// Configure AWS credentials for Bedrock provider
    ///
    /// This allows programmatic authentication instead of relying on the default
    /// AWS credential chain. Useful for applications that manage credentials
    /// centrally or use temporary credentials from STS.
    ///
    /// # Arguments
    /// * `access_key` - AWS access key ID
    /// * `secret_key` - AWS secret access key
    /// * `session_token` - Optional session token for temporary credentials
    /// * `region` - AWS region for service endpoints
    ///
    /// # Examples
    /// ```no_run
    /// use stood::agent::Agent;
    /// use stood::llm::models::Bedrock;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let agent = Agent::builder()
    ///     .model(Bedrock::ClaudeHaiku45)
    ///     .with_credentials(
    ///         "AKIA...".to_string(),
    ///         "secret".to_string(),
    ///         Some("token".to_string()),
    ///         "us-east-1".to_string()
    ///     )
    ///     .build().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_credentials(
        mut self,
        access_key: String,
        secret_key: String,
        session_token: Option<String>,
        region: String,
    ) -> Self {
        self.aws_credentials = Some(AwsCredentials {
            access_key,
            secret_key,
            session_token,
            region: Some(region),
        });
        self
    }

    /// Configure AWS credentials with a specific region
    ///
    /// **Deprecated:** Use `with_credentials` instead, which now includes region as a required parameter.
    /// This method is kept for backward compatibility.
    ///
    /// # Examples
    /// ```no_run
    /// use stood::agent::Agent;
    /// use stood::llm::models::Bedrock;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// // Deprecated - use with_credentials instead
    /// let agent = Agent::builder()
    ///     .model(Bedrock::ClaudeHaiku45)
    ///     .with_credentials_and_region(
    ///         "AKIA...".to_string(),
    ///         "secret".to_string(),
    ///         Some("token".to_string()),
    ///         "us-west-2".to_string()
    ///     )
    ///     .build().await?;
    /// # Ok(())
    /// # }
    /// ```
    #[deprecated(
        since = "1.0.0",
        note = "Use `with_credentials` instead, which now includes region as a required parameter"
    )]
    pub fn with_credentials_and_region(
        mut self,
        access_key: String,
        secret_key: String,
        session_token: Option<String>,
        region: String,
    ) -> Self {
        self.aws_credentials = Some(AwsCredentials {
            access_key,
            secret_key,
            session_token,
            region: Some(region),
        });
        self
    }

    /// Build the configured agent instance with smart defaults.
    ///
    /// Automatically creates a BedrockClient if none was provided, enabling
    /// the simplest possible usage while supporting full customization.
    ///
    /// # Returns
    ///
    /// A configured [`Agent`] instance ready for chat and agentic execution.
    ///
    /// # Examples
    ///
    /// Simplest usage (auto-creates client):
    /// ```no_run
    /// # use stood::agent::Agent;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let agent = Agent::builder().build().await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// With tools:
    /// ```no_run
    /// # use stood::agent::Agent;
    /// # use stood::tools::builtin::CalculatorTool;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let agent = Agent::builder()
    ///     .tool(Box::new(CalculatorTool::new()))
    ///     .with_builtin_tools()
    ///     .build().await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Full configuration:
    /// ```no_run
    /// # use stood::agent::Agent;
    /// # use stood::bedrock::BedrockClient;
    /// # use stood::types::ModelReference;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = BedrockClient::new().await?;
    ///
    /// let agent = Agent::builder()
    ///     .client(client)
    ///     .model(ModelReference::claude_sonnet_35())
    ///     .temperature(0.8)
    ///     .build().await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`StoodError::ConfigurationError`] when:
    /// - AWS credentials are missing or invalid
    /// - Tool registration fails
    /// - Bedrock client creation fails
    pub async fn build(mut self) -> Result<Agent> {
        crate::perf_checkpoint!("stood.agent_builder.build.start");
        let _build_guard = crate::perf_guard!("stood.agent_builder.build");

        // Use provided model or create default
        let model = if let Some(m) = self.model {
            // DEBUG: Log that model was found
            tracing::debug!(
                target: "stood::agent::builder",
                "DEBUG [STOOD-2]: build() found model, model_id={}",
                self.config.model_id
            );
            m
        } else {
            // DEBUG: Log that model is None, falling back to default
            tracing::warn!(
                target: "stood::agent::builder",
                "DEBUG [STOOD-2]: build() found NO MODEL (self.model is None), falling back to Claude Haiku 4.5"
            );
            // Default to Claude Haiku 4.5
            Box::new(crate::llm::models::Bedrock::ClaudeHaiku45)
        };

        // Update config with model info if not already set
        if self.config.model_id.is_empty() {
            self.config.provider = model.provider();
            self.config.model_id = model.model_id().to_string();
        }

        // Use model's default max_tokens if not explicitly set by user
        // This ensures each model gets its optimal output limit (e.g., 8192 for Sonnet 4.5)
        if self.config.max_tokens.is_none() || self.config.max_tokens == Some(4096) {
            let model_max = model.default_max_tokens();
            tracing::debug!(
                target: "stood::agent::builder",
                "Setting max_tokens={} from model default (was {:?})",
                model_max,
                self.config.max_tokens
            );
            self.config.max_tokens = Some(model_max);
        }

        // CRITICAL FIX: Auto-configure provider registry with timeout and error handling
        let provider_type = model.provider();

        // Configure custom credentials for Bedrock if provided
        if provider_type == ProviderType::Bedrock && self.aws_credentials.is_some() {
            crate::perf_checkpoint!("stood.agent_builder.configure_bedrock_creds.start");
            use crate::llm::registry::{BedrockCredentials, ProviderConfig};

            let aws_creds = self.aws_credentials.as_ref().unwrap();
            let bedrock_creds = BedrockCredentials {
                access_key: aws_creds.access_key.clone(),
                secret_key: aws_creds.secret_key.clone(),
                session_token: aws_creds.session_token.clone(),
            };

            let bedrock_config = ProviderConfig::Bedrock {
                region: aws_creds.region.clone(),
                credentials: Some(bedrock_creds),
            };

            crate::perf_timed!("stood.agent_builder.add_bedrock_config", {
                PROVIDER_REGISTRY
                    .add_config(ProviderType::Bedrock, bedrock_config)
                    .await
            });
        }

        // For LM Studio providers, configure retry settings if specified
        if provider_type == ProviderType::LmStudio && self.config.retry_config.is_some() {
            use crate::llm::registry::ProviderConfig;

            // Get base URL from existing config or use default
            let base_url = std::env::var("LM_STUDIO_BASE_URL")
                .unwrap_or_else(|_| "http://localhost:1234".to_string());

            // Configure LM Studio with custom retry settings
            let lm_studio_config = ProviderConfig::LMStudio {
                base_url,
                retry_config: self.config.retry_config.clone(),
            };

            PROVIDER_REGISTRY
                .add_config(ProviderType::LmStudio, lm_studio_config)
                .await;
        }

        // Check if provider is configured, with timeout
        let is_configured = crate::perf_timed!("stood.agent_builder.is_configured_check", {
            tokio::time::timeout(
                std::time::Duration::from_secs(5),
                crate::llm::registry::PROVIDER_REGISTRY.is_configured(provider_type),
            )
            .await
            .unwrap_or(false)
        });

        if !is_configured {
            // Auto-configure with timeout
            crate::perf_timed!("stood.agent_builder.auto_configure", {
                tokio::time::timeout(
                    std::time::Duration::from_secs(10),
                    crate::llm::registry::ProviderRegistry::configure(),
                )
                .await
                .map_err(|_| crate::StoodError::ConfigurationError {
                    message: "Provider registry configuration timed out".to_string(),
                })?
                .map_err(|e| crate::StoodError::ConfigurationError {
                    message: format!("Failed to auto-configure provider registry: {}", e),
                })
            })?;
        }

        // Get provider from registry with timeout (THIS IS THE MAIN BOTTLENECK)
        let provider = crate::perf_timed!("stood.agent_builder.get_provider", {
            tokio::time::timeout(
                std::time::Duration::from_secs(30),
                PROVIDER_REGISTRY.get_provider(provider_type),
            )
            .await
            .map_err(|_| crate::StoodError::ConfigurationError {
                message: "Provider creation timed out".to_string(),
            })?
            .map_err(|e| crate::StoodError::ConfigurationError {
                message: format!("Failed to get provider: {}", e),
            })
        })?;

        // Build internal agent
        let agent = crate::perf_timed!("stood.agent_builder.build_internal", {
            Agent::build_internal(
                provider,
                model,
                self.config,
                self.tools,
                self.middlewares,
                self.execution_config,
                self.agent_id,
                self.agent_name,
            )
            .await
        });

        crate::perf_checkpoint!("stood.agent_builder.build.end");
        agent
    }
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_agent_builder_default() {
        let agent = Agent::builder().build().await.unwrap();

        assert_eq!(
            agent.config().provider,
            crate::llm::traits::ProviderType::Bedrock
        );
        assert_eq!(
            agent.config().model_id,
            "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        );
        assert_eq!(agent.config().temperature, Some(0.7));
        assert_eq!(agent.config().max_tokens, Some(4096));
        assert!(agent.config().system_prompt.is_none());
        assert!(agent.conversation_history().is_empty());
    }

    #[tokio::test]
    async fn test_agent_builder_custom() {
        let agent = Agent::builder()
            .model(crate::llm::models::Bedrock::ClaudeHaiku45)
            .temperature(0.5)
            .max_tokens(2048)
            .system_prompt("You are a helpful assistant")
            .build()
            .await
            .unwrap();

        assert_eq!(
            agent.config().provider,
            crate::llm::traits::ProviderType::Bedrock
        );
        assert_eq!(
            agent.config().model_id,
            "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        );
        assert_eq!(agent.config().temperature, Some(0.5));
        assert_eq!(agent.config().max_tokens, Some(2048));
        assert_eq!(
            agent.config().system_prompt,
            Some("You are a helpful assistant".to_string())
        );
    }

    #[tokio::test]
    async fn test_agent_builder_with_tools() {
        let agent = Agent::builder().with_builtin_tools().build().await.unwrap();

        // Should have builtin tools
        assert!(agent.tool_registry.has_tool("calculator").await);
        assert!(agent.tool_registry.has_tool("file_read").await);
    }

    #[test]
    #[should_panic(expected = "Temperature must be between 0.0 and 1.0")]
    fn test_agent_builder_invalid_temperature() {
        Agent::builder().temperature(1.5);
    }

    #[test]
    #[should_panic(expected = "Max tokens must be greater than 0")]
    fn test_agent_builder_invalid_max_tokens() {
        Agent::builder().max_tokens(0);
    }

    #[tokio::test]
    async fn test_agent_history_management() {
        let mut agent = Agent::builder().build().await.unwrap();

        assert!(agent.conversation_history().is_empty());
        assert!(agent.conversation().is_empty());

        agent.add_user_message("Hello");
        assert_eq!(agent.conversation().message_count(), 1);
        assert!(!agent.conversation_history().is_empty());

        agent.add_assistant_message("Hi there!");
        assert_eq!(agent.conversation().message_count(), 2);

        agent.clear_history();
        assert!(agent.conversation_history().is_empty());
        assert!(agent.conversation().is_empty());
    }

    #[tokio::test]
    async fn test_agent_system_prompt() {
        let agent = Agent::builder()
            .system_prompt("You are a helpful assistant")
            .build()
            .await
            .unwrap();

        // System prompt should be set in conversation manager
        assert_eq!(
            agent.conversation().system_prompt(),
            Some("You are a helpful assistant")
        );
    }

    #[tokio::test]
    async fn test_agent_conversation_integration() {
        let mut agent = Agent::builder()
            .system_prompt("You are helpful")
            .build()
            .await
            .unwrap();

        // Add messages through agent methods
        agent.add_user_message("What is 2+2?");
        agent.add_assistant_message("2+2 equals 4");

        // Verify conversation state
        assert_eq!(agent.conversation().message_count(), 2);

        let last_message = agent.conversation().last_assistant_message().unwrap();
        assert_eq!(last_message.text(), Some("2+2 equals 4".to_string()));

        // Verify system prompt is set
        assert_eq!(
            agent.conversation().system_prompt(),
            Some("You are helpful")
        );
    }

    #[tokio::test]
    async fn test_agent_chat_functionality() {
        let mut agent = Agent::builder()
            .system_prompt("You are a helpful assistant. Keep responses brief.")
            .build()
            .await
            .unwrap();

        // Test chat functionality - handle potential API errors gracefully
        match agent.execute("What is 2+2?").await {
            Ok(response) => {
                // Verify conversation history was updated (even with empty response)
                assert_eq!(agent.conversation().message_count(), 2); // user + assistant

                let last_user_msg = &agent.conversation().messages().messages[0];
                assert_eq!(last_user_msg.text(), Some("What is 2+2?".to_string()));

                let last_assistant_msg = agent.conversation().last_assistant_message().unwrap();
                assert_eq!(last_assistant_msg.text(), Some(response.response.clone()));

                if response.response.is_empty() {
                    println!("Response is empty - likely due to missing AWS credentials in test environment");
                } else {
                    println!("Received valid response: {}", response.response);
                }
            }
            Err(e) => {
                // If API call fails (expected in test environment), verify error handling
                println!(
                    "Chat API call failed as expected in test environment: {}",
                    e
                );

                // Conversation should remain empty on failure since EventLoop manages it
                assert_eq!(agent.conversation().message_count(), 0);
            }
        }
    }

    #[tokio::test]
    async fn test_agent_multi_turn_chat() {
        let mut agent = Agent::builder()
            .system_prompt("You are helpful. Keep responses very brief.")
            .build()
            .await
            .unwrap();

        // Test multi-turn capability with error handling
        match agent.execute("Hello").await {
            Ok(response1) => {
                assert_eq!(agent.conversation().message_count(), 2);

                // Second turn - should have context
                if let Ok(response2) = agent.execute("What did I just say?").await {
                    assert_eq!(agent.conversation().message_count(), 4);

                    if !response1.response.is_empty() && !response2.response.is_empty() {
                        // Should contain reference to previous message
                        assert!(
                            response2.response.to_lowercase().contains("hello")
                                || response2.response.to_lowercase().contains("said")
                                || response2.response.to_lowercase().contains("just")
                        );
                    } else {
                        println!("Multi-turn test completed with empty responses (expected in test environment)");
                    }
                }
            }
            Err(_) => {
                // API calls may fail in test environment - verify structure
                assert_eq!(agent.conversation().message_count(), 0);
                println!("Multi-turn chat test skipped due to API unavailability");
            }
        }
    }

    #[tokio::test]
    async fn test_agent_identification() {
        // Test agent creation with custom ID and name
        let custom_id = "test-agent-123";
        let custom_name = "TestAgent";

        let agent = Agent::builder()
            .with_id(custom_id)
            .name(custom_name)
            .build()
            .await
            .unwrap();

        assert_eq!(agent.agent_id(), custom_id);
        assert_eq!(agent.agent_name(), Some(custom_name));
    }

    #[tokio::test]
    async fn test_agent_auto_id_generation() {
        // Test agent creation with auto-generated ID
        let agent = Agent::builder().name("AutoIdAgent").build().await.unwrap();

        // Should have auto-generated UUID
        assert!(!agent.agent_id().is_empty());
        assert_eq!(agent.agent_name(), Some("AutoIdAgent"));

        // ID should be a valid UUID format (36 characters with hyphens)
        assert_eq!(agent.agent_id().len(), 36);
        assert!(agent.agent_id().contains('-'));
    }

    #[tokio::test]
    async fn test_agent_context_creation() {
        let agent = Agent::builder()
            .with_id("parent-agent")
            .name("ParentAgent")
            .build()
            .await
            .unwrap();

        // Test context creation
        let context = agent.create_context("planning");

        assert_eq!(context.agent_id, "parent-agent");
        assert_eq!(context.agent_name, Some("ParentAgent".to_string()));
        assert_eq!(context.agent_type, "planning");
        assert!(context.span_context.is_none()); // Should be None initially
    }

    #[tokio::test]
    async fn test_agent_context_from_agent() {
        let agent = Agent::builder()
            .with_id("test-agent")
            .name("TestAgent")
            .build()
            .await
            .unwrap();

        // Test AgentContext::from_agent
        let context = AgentContext::from_agent(&agent, "researcher");

        assert_eq!(context.agent_id, agent.agent_id());
        assert_eq!(
            context.agent_name,
            agent.agent_name().map(|s| s.to_string())
        );
        assert_eq!(context.agent_type, "researcher");
    }

    #[tokio::test]
    async fn test_agent_context_manual_creation() {
        // Test manual context creation
        let context = AgentContext::new("manual-agent", Some("ManualAgent".to_string()), "analyst");

        assert_eq!(context.agent_id, "manual-agent");
        assert_eq!(context.agent_name, Some("ManualAgent".to_string()));
        assert_eq!(context.agent_type, "analyst");
        assert!(context.span_context.is_none());
    }

    #[tokio::test]
    async fn test_agent_config_identification() {
        // Test that agent config fields are used
        let config = AgentConfig {
            agent_id: Some("config-agent".to_string()),
            agent_name: Some("ConfigAgent".to_string()),
            ..AgentConfig::default()
        };

        let agent = Agent::build_internal(
            Arc::new(
                crate::llm::providers::bedrock::BedrockProvider::new(None)
                    .await
                    .unwrap(),
            ),
            Box::new(crate::llm::models::Bedrock::ClaudeHaiku45),
            config,
            vec![],
            vec![], // No middlewares
            crate::agent::config::ExecutionConfig::default(),
            None, // No override agent_id
            None, // No override agent_name
        )
        .await
        .unwrap();

        assert_eq!(agent.agent_id(), "config-agent");
        assert_eq!(agent.agent_name(), Some("ConfigAgent"));
    }

    #[tokio::test]
    async fn test_agent_builder_override_config() {
        // Test that builder overrides config values
        let agent = Agent::builder()
            .with_id("builder-agent")
            .name("BuilderAgent")
            .build()
            .await
            .unwrap();

        // Builder values should take precedence
        assert_eq!(agent.agent_id(), "builder-agent");
        assert_eq!(agent.agent_name(), Some("BuilderAgent"));
    }

    #[tokio::test]
    async fn test_agent_builder_with_custom_tools() {
        use crate::tools::Tool;
        use serde_json::json;

        // Create a mock unified tool for testing
        #[derive(Debug)]
        struct TestTool;

        #[async_trait::async_trait]
        impl Tool for TestTool {
            fn name(&self) -> &str {
                "test_tool"
            }
            fn description(&self) -> &str {
                "A test tool"
            }
            fn parameters_schema(&self) -> serde_json::Value {
                json!({
                    "type": "object",
                    "properties": {
                        "input": { "type": "string" }
                    }
                })
            }
            async fn execute(
                &self,
                _parameters: Option<serde_json::Value>,
                _agent_context: Option<&crate::agent::AgentContext>,
            ) -> std::result::Result<crate::tools::ToolResult, crate::tools::ToolError>
            {
                Ok(crate::tools::ToolResult::success(
                    json!({"result": "test completed"}),
                ))
            }
        }

        let tools: Vec<Box<dyn Tool>> = vec![Box::new(TestTool)];

        match Agent::builder().tools(tools).build().await {
            Ok(agent) => {
                assert_eq!(
                    agent.config().model_id,
                    "us.anthropic.claude-haiku-4-5-20251001-v1:0"
                );
                assert!(agent.conversation_history().is_empty());
                assert!(agent.tool_registry.has_tool("test_tool").await);
            }
            Err(e) => {
                // Constructor may fail in test environment due to AWS setup
                println!("Builder constructor test skipped due to setup: {}", e);
            }
        }
    }
}
