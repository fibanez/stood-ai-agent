//! Agent-related type definitions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use super::{content::ThinkingSummary, Messages, ToolExecutionResult};
use crate::llm::traits::CacheStrategy;
use crate::types::tools::ToolChoice;

/// Configuration for an agent instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Provider type for the model
    pub provider: crate::llm::traits::ProviderType,
    /// Model identifier within the provider
    pub model_id: String,
    /// Optional system prompt
    pub system_prompt: Option<String>,
    /// Maximum number of tools to execute in parallel
    pub max_parallel_tools: usize,
    /// Temperature for model responses (0.0 to 1.0)
    pub temperature: Option<f32>,
    /// Maximum tokens in model response
    pub max_tokens: Option<u32>,
    /// Whether to enable Claude 4's think tool
    pub enable_thinking: bool,
    /// Prompt caching strategy for reducing latency and costs
    ///
    /// When enabled, frequently used content (system prompts, tool definitions)
    /// can be cached by AWS Bedrock to reduce latency by up to 85% and costs by up to 90%
    /// on subsequent requests within the 5-minute cache TTL.
    ///
    /// See [`CacheStrategy`] for available options.
    #[serde(default)]
    pub cache_strategy: CacheStrategy,
    /// How the LLM should choose which tools to use
    ///
    /// Controls the tool-selection behavior sent to the provider:
    /// - `ToolChoice::Auto` (default) — the model may or may not call a tool
    /// - `ToolChoice::Any` — the model must call at least one tool
    /// - `ToolChoice::Tool { name }` — the model must call the named tool
    /// - `ToolChoice::None` — no tools are sent; the model cannot call any tool
    #[serde(default = "default_tool_choice")]
    pub tool_choice: ToolChoice,
    /// Additional model-specific parameters
    #[serde(default)]
    pub additional_params: HashMap<String, serde_json::Value>,
}

fn default_tool_choice() -> ToolChoice {
    ToolChoice::Auto
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            provider: crate::llm::traits::ProviderType::Bedrock,
            model_id: "us.anthropic.claude-3-5-haiku-20241022-v1:0".to_string(),
            system_prompt: None,
            max_parallel_tools: num_cpus::get(),
            temperature: None,
            max_tokens: None,
            enable_thinking: false, // Haiku doesn't support thinking
            cache_strategy: CacheStrategy::default(),
            tool_choice: ToolChoice::Auto,
            additional_params: HashMap::new(),
        }
    }
}

/// Model family classification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFamily {
    Claude,
    Nova,
}

/// Response from an agent interaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResponse {
    /// Unique identifier for this response
    pub id: Uuid,
    /// The response message from the agent
    pub message: super::Message,
    /// Tool executions that occurred during this interaction
    pub tool_executions: Vec<ToolExecutionResult>,
    /// Thinking summary if think tool was used
    pub thinking_summary: Option<ThinkingSummary>,
    /// Total duration of the interaction in milliseconds
    pub duration_ms: u64,
    /// Token usage information
    pub token_usage: Option<TokenUsage>,
    /// Whether this response required multiple reasoning iterations
    pub reasoning_iterations: u32,
    /// Additional metadata
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl AgentResponse {
    /// Create a new agent response
    pub fn new(
        message: super::Message,
        tool_executions: Vec<ToolExecutionResult>,
        duration_ms: u64,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            message,
            tool_executions,
            thinking_summary: None,
            duration_ms,
            token_usage: None,
            reasoning_iterations: 1,
            metadata: HashMap::new(),
        }
    }

    /// Add thinking summary to the response
    pub fn with_thinking_summary(mut self, summary: ThinkingSummary) -> Self {
        self.thinking_summary = Some(summary);
        self
    }

    /// Add token usage information
    pub fn with_token_usage(mut self, usage: TokenUsage) -> Self {
        self.token_usage = Some(usage);
        self
    }

    /// Set the number of reasoning iterations
    pub fn with_reasoning_iterations(mut self, iterations: u32) -> Self {
        self.reasoning_iterations = iterations;
        self
    }

    /// Get the text content of the response
    pub fn text(&self) -> Option<String> {
        self.message.text()
    }

    /// Check if this response used tools
    pub fn used_tools(&self) -> bool {
        !self.tool_executions.is_empty() || self.message.has_tool_use()
    }

    /// Check if this response included thinking
    pub fn used_thinking(&self) -> bool {
        self.thinking_summary.is_some()
    }
}

/// Token usage information for a model interaction
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct TokenUsage {
    /// Number of input tokens
    pub input_tokens: u32,
    /// Number of output tokens
    pub output_tokens: u32,
    /// Total number of tokens
    pub total_tokens: u32,
}

impl TokenUsage {
    /// Create new token usage
    pub fn new(input_tokens: u32, output_tokens: u32) -> Self {
        Self {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
        }
    }
}

/// Current state of an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    /// Unique identifier for the agent
    pub agent_id: String,
    /// Current conversation messages
    pub messages: Messages,
    /// Agent configuration
    pub config: AgentConfig,
    /// Whether the agent is currently processing
    pub is_processing: bool,
    /// Last interaction timestamp
    pub last_interaction: Option<chrono::DateTime<chrono::Utc>>,
    /// Total number of interactions
    pub interaction_count: u64,
    /// Cumulative token usage
    pub total_token_usage: Option<TokenUsage>,
}

impl AgentState {
    /// Create a new agent state
    pub fn new<S: Into<String>>(agent_id: S, config: AgentConfig) -> Self {
        Self {
            agent_id: agent_id.into(),
            messages: Messages::new(),
            config,
            is_processing: false,
            last_interaction: None,
            interaction_count: 0,
            total_token_usage: None,
        }
    }

    /// Update the state after an interaction
    pub fn update_after_interaction(&mut self, response: &AgentResponse) {
        self.last_interaction = Some(chrono::Utc::now());
        self.interaction_count += 1;

        if let Some(usage) = &response.token_usage {
            if let Some(total) = &mut self.total_token_usage {
                total.input_tokens += usage.input_tokens;
                total.output_tokens += usage.output_tokens;
                total.total_tokens += usage.total_tokens;
            } else {
                self.total_token_usage = Some(usage.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::messages::Message;

    #[test]
    fn test_agent_config_default() {
        let config = AgentConfig::default();
        assert_eq!(config.provider, crate::llm::traits::ProviderType::Bedrock);
        assert_eq!(
            config.model_id,
            "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        );
        assert!(!config.enable_thinking); // Haiku doesn't support thinking
        assert!(config.max_parallel_tools > 0);
    }

    #[test]
    fn test_token_usage() {
        let usage = TokenUsage::new(100, 50);
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
    }

    #[test]
    fn test_agent_response() {
        let message = Message::assistant("Hello!");
        let response = AgentResponse::new(message, vec![], 1000);

        assert_eq!(response.text(), Some("Hello!".to_string()));
        assert!(!response.used_tools());
        assert!(!response.used_thinking());
        assert_eq!(response.duration_ms, 1000);
    }
}
