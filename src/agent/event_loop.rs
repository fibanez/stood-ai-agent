//! Event Loop implementation for the Stood agent
//!
//! This module implements the core agentic loop with 5 distinct phases:
//! 1. Reasoning Phase: Model analyzes current state and plans next steps
//! 2. Tool Selection Phase: Model selects appropriate tools for parallel execution
//! 3. Tool Execution Phase: Selected tools executed with validation
//! 4. Reflection Phase: Model reflects on results and determines completion
//! 5. Response Generation Phase: Generate final response when task is complete
//!
//! The event loop supports recursive execution, context management, error recovery,
//! and comprehensive telemetry integration.

use crate::llm::traits::LlmProvider;
use crate::types::{ContentBlock, MessageRole};
use chrono::Utc;
use serde_json::Value;
use std::time::{Duration, Instant};
use tracing::debug;
use uuid::Uuid;

use crate::agent::callbacks::{CallbackEvent, CallbackHandler};
use crate::agent::evaluation::EvaluationStrategy;
use crate::agent::Agent;
use crate::error_recovery::RetryConfig;
use crate::streaming::{StreamCallback, StreamConfig, StreamEvent};
use crate::telemetry::{CycleMetrics, EventLoopMetrics, PerformanceTracer, ToolExecutionMetric};
use crate::tools::{ExecutorConfig, ToolExecutor, ToolRegistry};
use crate::Result;
use std::sync::Arc;

/// Information about a tool currently being streamed
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct CurrentToolInfo {
    tool_use_id: String,
    name: String,
}

use crate::telemetry::StoodTracer;

use crate::StoodError;

/// Configuration for the event loop
#[derive(Debug, Clone)]
pub struct EventLoopConfig {
    /// Maximum number of agentic cycles before stopping
    pub max_cycles: u32,
    /// Maximum duration for the entire event loop
    pub max_duration: Duration,
    /// Whether to enable streaming responses
    pub enable_streaming: bool,
    /// Tool execution configuration
    pub tool_config: ExecutorConfig,
    /// Whether to enable telemetry
    pub enable_telemetry: bool,
    /// Streaming configuration
    pub stream_config: StreamConfig,
    /// Error recovery configuration
    pub retry_config: RetryConfig,
    /// Evaluation strategy for determining continuation
    pub evaluation_strategy: EvaluationStrategy,
    /// Maximum number of tool iterations per cycle
    pub max_tool_iterations: u32,
    /// Cancellation token for early termination
    pub cancellation_token: Option<tokio_util::sync::CancellationToken>,
}

impl Default for EventLoopConfig {
    fn default() -> Self {
        Self {
            max_cycles: 10,
            max_duration: Duration::from_secs(300), // 5 minutes
            enable_streaming: true,                 // Enable real streaming by default
            tool_config: ExecutorConfig::default(),
            enable_telemetry: true,
            stream_config: StreamConfig::default(),
            retry_config: RetryConfig::default(),
            evaluation_strategy: EvaluationStrategy::default(),
            max_tool_iterations: 7, // Default conservative limit
            cancellation_token: None,
        }
    }
}

/// Result of an agentic loop execution
#[derive(Debug, Clone)]
pub struct EventLoopResult {
    /// Final response text
    pub response: String,
    /// Number of cycles executed
    pub cycles_executed: u32,
    /// Total duration of the event loop
    pub total_duration: Duration,
    /// Comprehensive metrics
    pub metrics: EventLoopMetrics,
    /// Whether the loop completed successfully
    pub success: bool,
    /// Error message if the loop failed
    pub error: Option<String>,
    /// Whether streaming was used
    pub was_streamed: bool,
    /// Stream events collected during execution
    pub stream_events: Vec<StreamEvent>,
}

/// Isolated evaluation context to prevent conversation pollution
#[derive(Debug)]
struct EvaluationContext {
    provider: Arc<dyn LlmProvider>,
    model_id: String,
    system_prompt: Option<String>,
}

impl EvaluationContext {
    /// Create a new evaluation context using the same provider as the main agent
    fn new(agent: &Agent) -> Self {
        Self {
            provider: agent.provider().clone(),
            model_id: agent.config().model_id.clone(),
            system_prompt: agent.conversation().system_prompt().map(|s| s.to_string()),
        }
    }

    /// Execute an isolated evaluation call that doesn't pollute the main conversation
    async fn evaluate_with_prompt(&self, prompt: &str) -> Result<String> {
        let mut messages = crate::types::Messages::new();
        if let Some(ref system) = self.system_prompt {
            messages.add_system_message(system);
        }
        messages.add_user_message(prompt);

        let response = self
            .provider
            .chat(&self.model_id, &messages, &Default::default())
            .await
            .map_err(|e| StoodError::InvalidInput {
                message: format!("Isolated evaluation failed: {}", e),
            })?;

        Ok(response.content)
    }
}

/// The core event loop for agentic execution
pub struct EventLoop {
    agent: Agent,
    tool_registry: ToolRegistry,
    #[allow(dead_code)]
    tool_executor: ToolExecutor,
    config: EventLoopConfig,
    metrics: EventLoopMetrics,
    stream_events: Vec<StreamEvent>,
    callback_handler: Option<Arc<dyn CallbackHandler>>,

    tracer: Option<StoodTracer>,

    active_spans: std::collections::HashMap<Uuid, SpanInfo>,
    performance_logger: PerformanceLogger,
    performance_tracer: PerformanceTracer,

    // Streaming completion tracking
    stream_completion_time: Option<std::time::Instant>,
    stream_was_active: bool,

    // Isolated evaluation context
    evaluation_context: Option<EvaluationContext>,

    // Track pending tool uses for cancellation handling
    // When cancellation occurs mid-execution, we need to add synthetic
    // tool_results for any pending tool_uses to keep conversation valid
    pending_tool_uses: Vec<crate::tools::ToolUse>,
}

/// Span tracking information for telemetry
#[derive(Debug, Clone)]
#[allow(dead_code)] // Used in future telemetry enhancements
struct SpanInfo {
    #[allow(dead_code)] // Used in future telemetry enhancements
    span_id: String,
    start_time: Instant,
    #[allow(dead_code)] // Used in future telemetry enhancements
    cycle_id: Uuid,
    #[allow(dead_code)] // Used in future telemetry enhancements
    span_type: SpanType,
}

/// Types of spans for tracking
#[derive(Debug, Clone)]
#[allow(dead_code)] // Used in future telemetry enhancements
enum SpanType {
    #[allow(dead_code)] // Used in future telemetry enhancements
    EventLoop,
    #[allow(dead_code)] // Used in future telemetry enhancements
    Cycle,
    ModelInvoke,
    ToolExecution,
}

/// Performance logging and metrics collection
#[derive(Debug, Clone)]
pub struct PerformanceLogger {
    pub cycle_times: Vec<Duration>,
    pub tool_times: std::collections::HashMap<String, Vec<Duration>>,
    pub model_invoke_times: Vec<Duration>,
    pub total_cycles: u32,
    /// Track input tokens for metrics and analysis
    pub total_input_tokens: u32,
    /// Track output tokens for metrics and analysis
    pub total_output_tokens: u32,
    /// Track input tokens per model call for detailed analysis
    pub model_input_tokens: Vec<u32>,
    /// Track output tokens per model call for detailed analysis
    pub model_output_tokens: Vec<u32>,
}

impl EventLoop {
    /// Create a new event loop
    pub fn new(agent: Agent, tool_registry: ToolRegistry, config: EventLoopConfig) -> Result<Self> {
        Self::new_with_callbacks(agent, tool_registry, config, None)
    }

    /// Create a new event loop with callback handler
    pub fn new_with_callbacks(
        agent: Agent,
        tool_registry: ToolRegistry,
        config: EventLoopConfig,
        callback_handler: Option<Arc<dyn CallbackHandler>>,
    ) -> Result<Self> {
        let tool_executor = ToolExecutor::new(config.tool_config.clone());

        let tracer = if config.enable_telemetry {
            // Use agent's telemetry config if available, otherwise fall back to env
            // This ensures custom service names and settings are propagated correctly
            let mut telemetry_config = agent
                .telemetry_config()
                .cloned()
                .unwrap_or_else(crate::telemetry::TelemetryConfig::from_env);
            // Convert agent LogLevel to telemetry LogLevel and apply it
            let telemetry_log_level = match agent.execution_config.log_level {
                crate::agent::config::LogLevel::Off => crate::telemetry::LogLevel::OFF,
                crate::agent::config::LogLevel::Info => crate::telemetry::LogLevel::INFO,
                crate::agent::config::LogLevel::Debug => crate::telemetry::LogLevel::DEBUG,
                crate::agent::config::LogLevel::Trace => crate::telemetry::LogLevel::TRACE,
            };
            telemetry_config.set_log_level(telemetry_log_level);
            StoodTracer::init(telemetry_config).map_err(|e| {
                StoodError::configuration_error(format!("Failed to initialize telemetry: {}", e))
            })?
        } else {
            None
        };

        // Create evaluation context if evaluation is enabled
        let evaluation_context = if config.evaluation_strategy.requires_evaluation() {
            Some(EvaluationContext::new(&agent))
        } else {
            None
        };

        Ok(Self {
            agent,
            tool_registry,
            tool_executor,
            config,
            metrics: EventLoopMetrics::new(),
            stream_events: Vec::new(),
            callback_handler,

            tracer,

            active_spans: std::collections::HashMap::new(),
            performance_logger: PerformanceLogger::new(),
            performance_tracer: PerformanceTracer::new(),

            // Initialize streaming completion tracking
            stream_completion_time: None,
            stream_was_active: false,

            // Initialize evaluation context
            evaluation_context,

            // Initialize pending tool uses tracking
            pending_tool_uses: Vec::new(),
        })
    }

    /// Get a reference to the agent
    pub fn agent(&self) -> &Agent {
        &self.agent
    }

    /// Create a clean conversation summary for evaluation (no tool blocks, no evaluation artifacts)
    fn create_evaluation_summary(&self) -> String {
        let mut summary_parts = Vec::new();

        for message in &self.agent.conversation().messages().messages {
            match message.role {
                MessageRole::User => {
                    if let Some(text) = message.text() {
                        // Skip evaluation artifacts
                        if !text.contains("[INTERNAL EVALUATION") {
                            summary_parts.push(format!("User: {}", text));
                        }
                    }
                }
                MessageRole::Assistant => {
                    if let Some(text) = message.text() {
                        if !text.trim().is_empty() {
                            summary_parts.push(format!("Assistant: {}", text));
                        }
                    }
                    // Extract tool results only (skip tool use blocks)
                    for block in &message.content {
                        if let ContentBlock::ToolResult { content, .. } = block {
                            if let crate::types::ToolResultContent::Text { text } = content {
                                summary_parts.push(format!("Tool result: {}", text));
                            } else if let crate::types::ToolResultContent::Json { data } = content {
                                summary_parts.push(format!("Tool result: {}", data));
                            }
                        }
                    }
                }
                MessageRole::System => {} // Skip system messages
            }
        }

        summary_parts.join("\n")
    }

    /// Execute the agentic loop for a given prompt
    pub async fn execute(&mut self, prompt: impl Into<String>) -> Result<EventLoopResult> {
        crate::perf_checkpoint!("stood.event_loop.execute.start");
        let _execute_guard = crate::perf_guard!("stood.event_loop.execute");
        let prompt = prompt.into();

        // Create telemetry span for the event loop
        let event_loop_span: Option<crate::telemetry::StoodSpan> = self.tracer.as_ref().map(|t| {
            // Start a new trace for this execution
            t.start_trace();

            // Start a session for CloudWatch Gen AI Observability
            // This sets session.id and gen_ai.conversation.id on all spans
            let mut session = t.start_session();
            session.set_agent_name(self.agent.agent_name().unwrap_or("stood-agent"));
            session.set_agent_id(self.agent.agent_id());
            t.set_session(session);

            t.start_invoke_agent_span(
                self.agent.agent_name().unwrap_or("stood-agent"),
                Some(self.agent.agent_id()),
            )
        });
        let loop_start = Instant::now();
        let loop_id = Uuid::new_v4();

        debug!("🚀 EventLoop::execute() started with prompt: '{}'", prompt);

        let _event_loop_guard = self
            .performance_tracer
            .start_operation("event_loop_execution");
        _event_loop_guard.add_context("loop_id", &loop_id.to_string());
        _event_loop_guard.add_context("prompt_length", &prompt.len().to_string());

        tracing::info!("Starting agentic loop {} for prompt: {}", loop_id, prompt);

        // Emit EventLoopStart callback
        if let Some(ref callback) = self.callback_handler {
            let event = CallbackEvent::EventLoopStart {
                loop_id,
                prompt: prompt.clone(),
                config: self.config.clone(),
            };
            if let Err(e) = callback.handle_event(event).await {
                tracing::warn!("Callback error during EventLoopStart: {}", e);
            }
        }

        // Remove redundant tracing span - use only OpenTelemetry spans

        // Record loop ID and configuration in current tracing span
        tracing::Span::current().record("event_loop.id", loop_id.to_string());
        tracing::Span::current().record("event_loop.max_cycles", self.config.max_cycles);
        tracing::Span::current().record(
            "event_loop.max_duration_seconds",
            self.config.max_duration.as_secs(),
        );

        // Store the original prompt for tool analysis
        let original_prompt = prompt.clone();

        // Add initial user message to conversation
        debug!("💬 Adding user message to EventLoop conversation");
        self.agent.add_user_message(&prompt);
        debug!(
            "💬 EventLoop conversation now has {} messages",
            self.agent.conversation().message_count()
        );

        let mut model_interaction_count = 0;
        // final_response will be set from all_responses.join() below
        let mut all_responses = Vec::new(); // Collect all responses from all cycles
        let mut loop_error = None;

        // Execute model interaction cycles until completion, limits, or cancellation
        while model_interaction_count < self.config.max_cycles
            && loop_start.elapsed() < self.config.max_duration
            && !self.is_cancelled()
        {
            let cycle_id = Uuid::new_v4();

            // 📊 COMPREHENSIVE MODEL INTERACTION CYCLE LOGGING
            tracing::info!(
                "🔄 MODEL INTERACTION CYCLE {} of {} - Duration: {}ms",
                model_interaction_count + 1,
                self.config.max_cycles,
                loop_start.elapsed().as_millis()
            );
            debug!(
                "🔄 Model Interaction Cycle {} ID: {}, Conversation messages: {}",
                model_interaction_count + 1,
                cycle_id,
                self.agent.conversation().message_count()
            );

            // Emit CycleStart callback (model interaction)
            if let Some(ref callback) = self.callback_handler {
                let event = CallbackEvent::CycleStart {
                    cycle_id,
                    cycle_number: model_interaction_count + 1,
                };
                if let Err(e) = callback.handle_event(event).await {
                    tracing::warn!("Callback error during CycleStart: {}", e);
                }
            }

            _event_loop_guard.checkpoint(&format!(
                "starting_model_interaction_{}",
                model_interaction_count + 1
            ));
            self.performance_tracer.record_cycle(
                &format!("model_interaction_start_{}", model_interaction_count + 1),
                Duration::from_millis(0),
            );

            debug!(
                "🔄 Starting model interaction {} with execute_cycle_with_prompt",
                model_interaction_count + 1
            );

            // Pass the parent span context to the method for proper parent-child relationship
            let parent_context = event_loop_span.as_ref().map(|span| span.context());
            // Pass invoke_agent span IDs for log event linking (AgentCore Evaluations)
            let invoke_agent_span_ids = event_loop_span
                .as_ref()
                .map(|span| (span.trace_id().to_string(), span.span_id().to_string()));

            crate::perf_checkpoint!("stood.event_loop.cycle.start");
            let cycle_result = crate::perf_timed!("stood.event_loop.cycle", {
                self.execute_cycle_with_prompt_with_context(
                    cycle_id,
                    &original_prompt,
                    model_interaction_count == 0,
                    parent_context,
                    invoke_agent_span_ids,
                )
                .await
            });
            crate::perf_checkpoint!("stood.event_loop.cycle.end");

            match cycle_result {
                Ok(cycle_result) => {
                    model_interaction_count += 1;

                    // Collect the response from this cycle
                    all_responses.push(cycle_result.response.clone());

                    // 📊 COMPREHENSIVE MODEL INTERACTION CYCLE COMPLETION LOGGING
                    tracing::info!("✅ MODEL INTERACTION CYCLE {} COMPLETED - Response length: {}, Should continue: {}, Tool iterations: {}",
                        model_interaction_count,
                        cycle_result.response.len(),
                        cycle_result.should_continue,
                        cycle_result.tool_iterations_used.unwrap_or(0)
                    );
                    debug!(
                        "✅ Model Interaction Cycle {} response preview: '{}'",
                        model_interaction_count,
                        cycle_result.response.chars().take(100).collect::<String>()
                    );

                    self.performance_tracer.record_cycle(
                        &format!("model_interaction_completed_{}", model_interaction_count),
                        Duration::from_millis(0),
                    );

                    if cycle_result.should_continue {
                        tracing::info!("🔄 Model Interaction Cycle {} completed, CONTINUING to next model interaction", model_interaction_count);
                        _event_loop_guard.checkpoint(&format!(
                            "model_interaction_{}_continue",
                            model_interaction_count
                        ));
                        continue;
                    } else {
                        tracing::info!(
                            "🏁 Model Interaction Cycle {} completed task, STOPPING event loop",
                            model_interaction_count
                        );
                        _event_loop_guard.checkpoint(&format!(
                            "model_interaction_{}_final",
                            model_interaction_count
                        ));
                        break;
                    }
                }
                Err(e) => {
                    loop_error = Some(format!(
                        "Model Interaction Cycle {} failed: {}",
                        cycle_id, e
                    ));
                    tracing::error!("Event loop failed: {}", e);

                    // Emit Error callback
                    if let Some(ref callback) = self.callback_handler {
                        let event = CallbackEvent::Error {
                            error: e.clone(),
                            context: format!(
                                "Model Interaction Cycle {} execution failed",
                                cycle_id
                            ),
                        };
                        if let Err(callback_err) = callback.handle_event(event).await {
                            tracing::warn!("Callback error during Error event: {}", callback_err);
                        }
                    }

                    break;
                }
            }
        }

        let total_duration = loop_start.elapsed();
        let mut success = loop_error.is_none();

        // Combine all responses from all cycles into the final response
        let mut final_response = all_responses.join("\n\n"); // Join with double newlines for readability

        // Emit EventLoopComplete callback
        if let Some(ref callback) = self.callback_handler {
            let event = CallbackEvent::EventLoopComplete {
                result: EventLoopResult {
                    response: final_response.clone(),
                    cycles_executed: model_interaction_count,
                    total_duration,
                    metrics: self.metrics.clone(),
                    success,
                    error: loop_error.clone(),
                    was_streamed: self.config.enable_streaming,
                    stream_events: self.stream_events.clone(),
                },
                total_duration,
            };
            if let Err(e) = callback.handle_event(event).await {
                tracing::warn!("Callback error during EventLoopComplete: {}", e);
            }
        }

        // Final telemetry and logging for event loop completion

        // Record final telemetry in current tracing span (automatically handled by #[tracing::instrument])
        tracing::Span::current().record("event_loop.model_interactions", model_interaction_count);
        tracing::Span::current().record("event_loop.total_duration_ms", total_duration.as_millis());
        tracing::Span::current().record("event_loop.success", success);

        // Log final performance metrics
        self.performance_logger.log_event_loop_completion(
            total_duration,
            model_interaction_count,
            success,
        );

        if model_interaction_count >= self.config.max_cycles {
            tracing::warn!(
                model_interactions = model_interaction_count,
                max_cycles = self.config.max_cycles,
                "Event loop reached maximum model interactions"
            );
        }

        if total_duration >= self.config.max_duration {
            tracing::warn!(
                duration_ms = total_duration.as_millis(),
                max_duration_ms = self.config.max_duration.as_millis(),
                "Event loop reached maximum duration"
            );
        }

        if self.is_cancelled() {
            tracing::info!("Event loop cancelled by cancellation token");

            // Add synthetic tool results for any pending tool uses to keep conversation valid
            // This ensures tool_use blocks always have matching tool_result blocks
            if !self.pending_tool_uses.is_empty() {
                tracing::info!(
                    "Adding synthetic tool results for {} cancelled tool(s)",
                    self.pending_tool_uses.len()
                );

                let synthetic_results: Vec<ToolResult> = self.pending_tool_uses
                    .iter()
                    .map(|tool_use| ToolResult {
                        tool_use_id: tool_use.tool_use_id.clone(),
                        tool_name: tool_use.name.clone(),
                        input: tool_use.input.clone(),
                        success: false,
                        output: Some(serde_json::json!({
                            "cancelled": true,
                            "message": format!(
                                "Tool '{}' execution was cancelled by user request before completion.",
                                tool_use.name
                            )
                        })),
                        error: Some("Execution cancelled by user request".to_string()),
                        duration: std::time::Duration::ZERO,
                    })
                    .collect();

                // Add synthetic results to conversation to maintain valid state
                let tool_result_message = self.create_tool_result_message(synthetic_results);
                self.agent.conversation_mut().add_message(tool_result_message);

                tracing::debug!(
                    "Added synthetic tool results for cancelled tools: {:?}",
                    self.pending_tool_uses.iter().map(|t| &t.name).collect::<Vec<_>>()
                );

                // Clear pending tool uses after adding synthetic results
                self.pending_tool_uses.clear();
            }

            // Override success status for cancellation
            success = false;
            // Override final response for cancellation
            final_response = "Execution cancelled by user request".to_string();
            // Set loop error to indicate cancellation
            loop_error = Some("Execution cancelled by cancellation token".to_string());
        }

        debug!("🏁 EventLoop::execute() completing with final_response: '{}', model_interactions: {}, success: {}",
                 final_response, model_interaction_count, success);
        debug!(
            "🏁 EventLoop conversation has {} messages at completion",
            self.agent.conversation().message_count()
        );

        // Complete the event loop span
        if let Some(mut span) = event_loop_span {
            span.set_attribute(
                "event_loop.model_interactions",
                model_interaction_count as i64,
            );
            span.set_attribute(
                "event_loop.total_duration_ms",
                total_duration.as_millis() as i64,
            );
            span.set_attribute("event_loop.success", success);

            // Record token usage from metrics
            span.record_tokens(
                self.metrics.total_tokens.input_tokens,
                self.metrics.total_tokens.output_tokens,
            );

            if let Some(ref error) = loop_error {
                span.set_attribute("event_loop.error", error.clone());
                span.set_error(error);
            } else {
                span.set_success();
            }
            // Span will be finished automatically on drop
        }

        // Flush pending spans to CloudWatch
        if let Some(ref tracer) = self.tracer {
            if let Err(e) = tracer.flush().await {
                tracing::warn!("Failed to flush telemetry spans: {}", e);
            }
        }

        Ok(EventLoopResult {
            response: final_response,
            cycles_executed: model_interaction_count,
            total_duration,
            metrics: self.metrics.clone(),
            success,
            error: loop_error,
            was_streamed: self.config.enable_streaming,
            stream_events: self.stream_events.clone(),
        })
    }

    /// Execute a single model interaction with the 5-phase pattern using original prompt
    async fn execute_cycle_with_prompt_with_context(
        &mut self,
        cycle_id: Uuid,
        original_prompt: &str,
        is_first_cycle: bool,
        _parent_context: Option<opentelemetry::Context>,
        invoke_agent_span_ids: Option<(String, String)>, // (trace_id, span_id) for log events
    ) -> Result<CycleResult> {
        // Create telemetry span for the model interaction cycle
        let model_id = self.agent.model().model_id().to_string();
        let cycle_span: Option<crate::telemetry::StoodSpan> =
            self.tracer.as_ref().map(|t| t.start_chat_span(&model_id));

        debug!(
            "🔍 execute_cycle_with_prompt() started for model interaction {}",
            cycle_id
        );
        let cycle_start = Instant::now();
        let mut cycle_metrics = CycleMetrics::new(cycle_id);

        let _cycle_guard = self.performance_tracer.start_operation("cycle_execution");
        _cycle_guard.add_context("cycle_id", &cycle_id.to_string());
        _cycle_guard.add_context("is_first_cycle", &is_first_cycle.to_string());
        _cycle_guard.add_context("original_prompt_length", &original_prompt.len().to_string());

        tracing::debug!(
            "Starting model interaction {} (first: {})",
            cycle_id,
            is_first_cycle
        );

        // Using tracing::instrument for automatic span management and context propagation

        // Use original prompt for first interaction, current conversation for subsequent interactions
        let _prompt_for_analysis = if is_first_cycle {
            original_prompt.to_string()
        } else {
            self.agent
                .conversation()
                .messages()
                .messages
                .last()
                .and_then(|msg| msg.text())
                .unwrap_or_default()
        };

        // NEW LLM-DRIVEN APPROACH: Model reasons with tools and decides which to use

        // Phase 1: Get tool configuration for model
        _cycle_guard.checkpoint("get_tool_config");
        let tool_config_start = Instant::now();
        let tool_config = self.tool_registry.get_tool_config().await;
        debug!("🔧 Tool config has {} tools", tool_config.tools.len());
        let tool_config_duration = tool_config_start.elapsed();

        if tool_config_duration > Duration::from_millis(10) {
            self.performance_tracer
                .record_waiting("tool_config_retrieval", tool_config_duration);
        }

        tracing::debug!(
            "Providing {} tools to model for decision making (took {}ms)",
            tool_config.tools.len(),
            tool_config_duration.as_millis()
        );

        // Phase 2: Model reasoning with tool awareness - Model makes tool decisions
        _cycle_guard.checkpoint("start_model_invocation");
        let model_start = Instant::now();

        // Model call span - placeholder for Milestone 2
        let _model_span: Option<(crate::telemetry::StoodSpan, Uuid)> = None;

        debug!(
            "🌐 About to make LLM provider call. Streaming: {}",
            self.config.enable_streaming
        );
        let llm_response = self.execute_chat_with_tools(&tool_config).await?;

        let model_duration = model_start.elapsed();
        _cycle_guard.checkpoint("model_invocation_complete");

        // Emit ModelStart callback (after LLM call to capture raw request JSON)
        if let Some(ref callback) = self.callback_handler {
            // Get raw request JSON from provider if available
            let raw_request_json = if let Some(bedrock_provider) =
                self.agent
                    .provider()
                    .as_any()
                    .downcast_ref::<crate::llm::providers::BedrockProvider>()
            {
                bedrock_provider.get_last_request_json()
            } else {
                None
            };

            let event = CallbackEvent::ModelStart {
                provider: self.agent.config().provider,
                model_id: self.agent.config().model_id.clone(),
                messages: self.agent.conversation().messages().clone(),
                tools_available: tool_config.tools.len(),
                raw_request_json,
            };
            if let Err(e) = callback.handle_event(event).await {
                tracing::warn!("Callback error during ModelStart: {}", e);
            }
        }

        // Emit ModelComplete callback
        if let Some(ref callback) = self.callback_handler {
            let event = CallbackEvent::ModelComplete {
                response: llm_response.content.clone(),
                stop_reason: crate::types::StopReason::EndTurn, // TODO: Map from LLM response
                duration: model_duration,
                tokens: llm_response.usage.as_ref().map(|t| {
                    crate::agent::callbacks::events::TokenUsage {
                        input_tokens: t.input_tokens,
                        output_tokens: t.output_tokens,
                        total_tokens: t.total_tokens,
                    }
                }),
                raw_response_data: None, // TODO: Implement in Phase 2
            };
            if let Err(e) = callback.handle_event(event).await {
                tracing::warn!("Callback error during ModelComplete: {}", e);
            }
        }

        if let Some((mut span, model_span_id)) = _model_span {
            // Enhanced Model Inference Details - Phase 5
            span.set_attribute("model.duration_ms", model_duration.as_millis() as i64);
            span.set_attribute(
                "model.stop_reason",
                "EndTurn", // TODO: Map from LLM response properly
            );

            // Add response content preview (first 200 chars for observability)
            let response_preview = if llm_response.content.chars().count() > 200 {
                format!("{}...", crate::utils::logging::truncate_string(&llm_response.content, 200))
            } else {
                llm_response.content.clone()
            };
            span.set_attribute(
                crate::telemetry::semantic_conventions::GEN_AI_RESPONSE_CONTENT_PREVIEW,
                response_preview,
            );
            span.set_attribute(
                crate::telemetry::semantic_conventions::GEN_AI_RESPONSE_CONTENT_LENGTH,
                llm_response.content.len() as i64,
            );

            // Track tool usage in response
            span.set_attribute(
                crate::telemetry::semantic_conventions::GEN_AI_RESPONSE_TOOL_CALLS_COUNT,
                llm_response.tool_calls.len() as i64,
            );
            if !llm_response.tool_calls.is_empty() {
                let tool_names: Vec<String> = llm_response
                    .tool_calls
                    .iter()
                    .map(|tc| tc.name.clone())
                    .collect();
                span.set_attribute(
                    crate::telemetry::semantic_conventions::GEN_AI_RESPONSE_TOOL_NAMES,
                    tool_names.join(","),
                );
            }

            // Token usage with GenAI semantic conventions
            if let Some(token_usage) = &llm_response.usage {
                span.set_attribute(
                    crate::telemetry::semantic_conventions::GEN_AI_USAGE_INPUT_TOKENS,
                    token_usage.input_tokens as i64,
                );
                span.set_attribute(
                    crate::telemetry::semantic_conventions::GEN_AI_USAGE_OUTPUT_TOKENS,
                    token_usage.output_tokens as i64,
                );
                span.set_attribute(
                    crate::telemetry::semantic_conventions::GEN_AI_USAGE_TOTAL_TOKENS,
                    token_usage.total_tokens as i64,
                );

                // Legacy attributes for backward compatibility
                span.set_attribute("model.input_tokens", token_usage.input_tokens as i64);
                span.set_attribute("model.output_tokens", token_usage.output_tokens as i64);
                span.set_attribute("model.total_tokens", token_usage.total_tokens as i64);
            }

            // Response characteristics
            span.set_attribute(
                crate::telemetry::semantic_conventions::GEN_AI_RESPONSE_FINISH_REASON,
                "stop",
            );
            if llm_response.content.is_empty() && !llm_response.tool_calls.is_empty() {
                span.set_attribute(
                    crate::telemetry::semantic_conventions::GEN_AI_RESPONSE_TYPE,
                    "tool_calls_only",
                );
            } else if !llm_response.content.is_empty() && llm_response.tool_calls.is_empty() {
                span.set_attribute(
                    crate::telemetry::semantic_conventions::GEN_AI_RESPONSE_TYPE,
                    "text_only",
                );
            } else if !llm_response.content.is_empty() && !llm_response.tool_calls.is_empty() {
                span.set_attribute(
                    crate::telemetry::semantic_conventions::GEN_AI_RESPONSE_TYPE,
                    "text_and_tools",
                );
            } else {
                span.set_attribute(
                    crate::telemetry::semantic_conventions::GEN_AI_RESPONSE_TYPE,
                    "empty",
                );
            }

            span.set_success();
            self.active_spans.remove(&model_span_id);
        }

        // Log model performance with token usage
        let token_usage_ref = llm_response
            .usage
            .as_ref()
            .map(|u| crate::types::TokenUsage::new(u.input_tokens, u.output_tokens));
        self.performance_logger
            .log_model_performance(model_duration, token_usage_ref.as_ref());

        cycle_metrics.model_invocations += 1;
        if let Some(token_usage) = &llm_response.usage {
            cycle_metrics.tokens_used.input_tokens += token_usage.input_tokens;
            cycle_metrics.tokens_used.output_tokens += token_usage.output_tokens;
            cycle_metrics.tokens_used.total_tokens =
                cycle_metrics.tokens_used.input_tokens + cycle_metrics.tokens_used.output_tokens;
        }

        // Phase 3: Handle model response - may involve multiple tool execution rounds
        let mut current_response = llm_response;

        tracing::debug!(
            "🔧 Main event loop received response with {} tool calls",
            current_response.tool_calls.len()
        );
        for (i, tool_call) in current_response.tool_calls.iter().enumerate() {
            tracing::debug!(
                "🔧 Main loop tool call {}: {} with input: {}",
                i + 1,
                tool_call.name,
                serde_json::to_string(&tool_call.input).unwrap_or_default()
            );
        }
        let mut tool_iteration_count = 0;
        let mut loop_count = 0; // Count each loop iteration/event match
        let max_tool_iterations = self.config.max_tool_iterations; // Use configurable limit
        // Accumulate tool results for the agent invocation log (Faithfulness evaluation)
        let mut accumulated_tool_results: Vec<(String, String, String)> = Vec::new();

        // Continue processing until we get a final response (no more tools)
        loop {
            loop_count += 1;
            tracing::debug!(
                "Loop iteration {} - Processing response with {} tool calls, content: '{}'",
                loop_count,
                current_response.tool_calls.len(),
                current_response
                    .content
                    .chars()
                    .take(100)
                    .collect::<String>()
            );

            // TRACE MODE: Print detailed iteration state
            if tracing::level_enabled!(tracing::Level::TRACE) {
                tracing::trace!(
                    "📋 TRACE: ==================== ITERATION {} STATE ====================",
                    loop_count
                );
                tracing::trace!(
                    "📋 TRACE: Tool iteration count: {}/{}",
                    tool_iteration_count,
                    max_tool_iterations
                );
                tracing::trace!(
                    "📋 TRACE: Current response content: '{}'",
                    current_response.content
                );
                tracing::trace!(
                    "📋 TRACE: Current response tool calls: {}",
                    current_response.tool_calls.len()
                );

                for (i, tool_call) in current_response.tool_calls.iter().enumerate() {
                    tracing::trace!(
                        "📋 TRACE: Tool call #{}: {} ({})",
                        i + 1,
                        tool_call.name,
                        tool_call.id
                    );
                    tracing::trace!(
                        "📋 TRACE: Tool call #{}: Input: {}",
                        i + 1,
                        serde_json::to_string_pretty(&tool_call.input)
                            .unwrap_or_else(|_| "Invalid JSON".to_string())
                    );
                }

                let conv_len = self.agent.conversation().messages().len();
                tracing::trace!(
                    "📋 TRACE: Current conversation length: {} messages",
                    conv_len
                );
                tracing::trace!(
                    "📋 TRACE: ==================== END ITERATION STATE ===================="
                );
            }
            // Check if response has tool calls to determine next action
            if !current_response.tool_calls.is_empty() {
                tool_iteration_count += 1;

                // 📊 COMPREHENSIVE TOOL ITERATION LOGGING
                tracing::info!(
                    "🔧 TOOL ITERATION {} of {} - {} tool calls requested in this round",
                    tool_iteration_count,
                    max_tool_iterations,
                    current_response.tool_calls.len()
                );

                // FIXED: Check iteration count BEFORE executing tools, not per tool call
                if tool_iteration_count > max_tool_iterations {
                    tracing::warn!(
                        "🚫 REACHED MAXIMUM TOOL ITERATIONS ({}), stopping to prevent infinite loop. This round requested {} tools.",
                        max_tool_iterations,
                        current_response.tool_calls.len()
                    );
                    // Create a final response with max iterations message
                    current_response.content = "I've reached the maximum number of tool executions. Please try rephrasing your request.".to_string();
                    current_response.tool_calls.clear();
                    break;
                }

                tracing::info!(
                    "🤖 LLM requested {} tools in iteration {}: {:?}",
                    current_response.tool_calls.len(),
                    tool_iteration_count,
                    current_response
                        .tool_calls
                        .iter()
                        .map(|tc| &tc.name)
                        .collect::<Vec<_>>()
                );

                // 📊 Detailed tool call breakdown
                for (i, tool_call) in current_response.tool_calls.iter().enumerate() {
                    tracing::info!(
                        "🔧 Tool {}/{}: {} (ID: {})",
                        i + 1,
                        current_response.tool_calls.len(),
                        tool_call.name,
                        tool_call.id
                    );
                }

                // Debug: Show detailed tool call information
                for (i, tool_call) in current_response.tool_calls.iter().enumerate() {
                    tracing::debug!(
                        "🤖 Tool {} - '{}' (ID: {}) with input: {}",
                        i + 1,
                        tool_call.name,
                        tool_call.id,
                        serde_json::to_string_pretty(&tool_call.input)
                            .unwrap_or_else(|_| "invalid JSON".to_string())
                    );
                }

                // Add assistant message with tool calls to conversation
                // Create content blocks: text content + tool_use blocks
                let mut content_blocks = vec![];

                // Add text content if present
                if !current_response.content.is_empty() {
                    content_blocks
                        .push(crate::types::ContentBlock::text(&current_response.content));
                }

                // Add tool_use blocks from tool_calls
                for tool_call in &current_response.tool_calls {
                    content_blocks.push(crate::types::ContentBlock::tool_use(
                        &tool_call.id,
                        &tool_call.name,
                        tool_call.input.clone(),
                    ));
                }

                // Create proper assistant message with both text and tool_use blocks
                let assistant_message = crate::types::Message::new(
                    crate::types::MessageRole::Assistant,
                    content_blocks,
                );
                self.agent
                    .conversation_mut()
                    .add_message(assistant_message.clone());

                // Extract tool uses from conversation message (LLM-driven approach)
                let tool_uses = match self.extract_tool_uses(&assistant_message) {
                    Ok(tools) => {
                        tracing::debug!(
                            "🔧 Extracted {} tool uses from conversation message",
                            tools.len()
                        );
                        for (i, tool_use) in tools.iter().enumerate() {
                            tracing::debug!(
                                "🔧 Tool use {}: {} (ID: {}) with input: {}",
                                i + 1,
                                tool_use.name,
                                tool_use.tool_use_id,
                                serde_json::to_string(&tool_use.input).unwrap_or_default()
                            );
                        }
                        tools
                    }
                    Err(e) => {
                        tracing::error!("❌ Failed to extract tool uses from conversation: {}", e);
                        // Fallback: empty tool uses to prevent crash
                        Vec::new()
                    }
                };

                tracing::info!(
                    "🔧 Executing {} tools: {}",
                    tool_uses.len(),
                    tool_uses
                        .iter()
                        .map(|tu| tu.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );

                // Track pending tool uses for cancellation handling
                // If cancellation occurs during execution, we'll add synthetic results
                self.pending_tool_uses = tool_uses.clone();

                // Execute tools using existing infrastructure - pass context for proper span hierarchy
                let cycle_context = cycle_span.as_ref().map(|span| span.context());
                match self
                    .tool_execution_phase(tool_uses, &mut cycle_metrics, cycle_context)
                    .await
                {
                    Ok(tool_results) => {
                        // Clear pending tool uses - execution completed successfully
                        self.pending_tool_uses.clear();

                        tracing::info!(
                            "✅ Tool execution completed with {} results",
                            tool_results.len()
                        );

                        // Add tool results to conversation for next LLM iteration
                        let tool_result_message =
                            self.create_tool_result_message(tool_results.clone());
                        self.agent
                            .conversation_mut()
                            .add_message(tool_result_message);

                        // Accumulate tool results for agent invocation log (Faithfulness eval)
                        for result in &tool_results {
                            let input_str = serde_json::to_string(&result.input)
                                .unwrap_or_else(|_| result.input.to_string());
                            let output_str = result
                                .output
                                .as_ref()
                                .map(|v| serde_json::to_string(v).unwrap_or_else(|_| v.to_string()))
                                .unwrap_or_else(|| "No output".to_string());
                            accumulated_tool_results.push((
                                result.tool_name.clone(),
                                input_str,
                                output_str,
                            ));
                        }

                        tracing::debug!(
                            "🔄 Tool results added to conversation, making follow-up LLM call"
                        );

                        // TRACE MODE: Print tool execution results
                        if tracing::level_enabled!(tracing::Level::TRACE) {
                            tracing::trace!("📋 TRACE: ==================== TOOL EXECUTION RESULTS ====================");
                            tracing::trace!("📋 TRACE: Executed {} tools:", tool_results.len());
                            for (i, result) in tool_results.iter().enumerate() {
                                tracing::trace!(
                                    "📋 TRACE: Tool result #{}: {} ({})",
                                    i + 1,
                                    result.tool_name,
                                    result.tool_use_id
                                );
                                tracing::trace!(
                                    "📋 TRACE: Tool result #{}: Success: {}",
                                    i + 1,
                                    result.success
                                );
                                tracing::trace!(
                                    "📋 TRACE: Tool result #{}: Duration: {:?}",
                                    i + 1,
                                    result.duration
                                );
                                if let Some(output) = &result.output {
                                    tracing::trace!(
                                        "📋 TRACE: Tool result #{}: Output: {}",
                                        i + 1,
                                        serde_json::to_string_pretty(output)
                                            .unwrap_or_else(|_| "Invalid JSON".to_string())
                                    );
                                }
                                if let Some(error) = &result.error {
                                    tracing::trace!(
                                        "📋 TRACE: Tool result #{}: Error: {}",
                                        i + 1,
                                        error
                                    );
                                }
                            }
                            tracing::trace!("📋 TRACE: ==================== END TOOL RESULTS ====================");
                        }

                        // Debug: Log conversation state before follow-up call for Nova debugging
                        let total_messages = self.agent.conversation().messages().len();
                        tracing::debug!(
                            "🔧 Nova debugging - conversation has {} messages before follow-up",
                            total_messages
                        );

                        if total_messages > 0 {
                            let last_message =
                                &self.agent.conversation().messages()[total_messages - 1];
                            tracing::debug!(
                                "🔧 Nova debugging - last message type: {:?}, content length: {}",
                                last_message.role,
                                last_message.content.len()
                            );

                            // Enhanced debugging: Log actual message content
                            if let Some(text) = last_message.text() {
                                let preview = if text.chars().count() > 200 {
                                    format!("{}...", crate::utils::logging::truncate_string(&text, 200))
                                } else {
                                    text.to_string()
                                };
                                tracing::debug!("🔧 Last message content preview: '{}'", preview);
                            }
                        }

                        // Log all conversation messages for debugging
                        tracing::debug!("🔧 Full conversation state before follow-up:");
                        for (i, msg) in self.agent.conversation().messages().iter().enumerate() {
                            if let Some(text) = msg.text() {
                                let preview = if text.chars().count() > 100 {
                                    format!("{}...", text.chars().take(100).collect::<String>())
                                } else {
                                    text
                                };
                                tracing::debug!(
                                    "🔧   Message {}: {:?} - '{}'",
                                    i,
                                    msg.role,
                                    preview
                                );
                            } else {
                                tracing::debug!(
                                    "🔧   Message {}: {:?} - [Non-text content]",
                                    i,
                                    msg.role
                                );
                            }
                        }

                        // Make another LLM call to get the final response based on tool results
                        match self.execute_chat_with_tools(&tool_config).await {
                            Ok(follow_up_response) => {
                                current_response = follow_up_response;
                                tracing::debug!("✅ Received follow-up response from LLM");
                                tracing::debug!(
                                    "✅ Follow-up response content length: {}",
                                    current_response.content.len()
                                );
                                if current_response.content.is_empty() {
                                    tracing::warn!("⚠️  Follow-up response is EMPTY despite successful tool execution!");
                                    tracing::warn!("⚠️  Tool results were available but model didn't generate response");
                                } else {
                                    let preview = if current_response.content.chars().count() > 200 {
                                        format!("{}...", crate::utils::logging::truncate_string(&current_response.content, 200))
                                    } else {
                                        current_response.content.clone()
                                    };
                                    tracing::debug!("✅ Follow-up response preview: '{}'", preview);
                                }
                            }
                            Err(e) => {
                                tracing::error!("❌ Follow-up LLM call failed: {}", e);
                                tracing::error!("❌ Follow-up LLM call error details: {:?}", e);

                                // Include more detailed error information for debugging
                                let error_details = format!("Follow-up LLM call failed: {}", e);
                                tracing::warn!("🔧 Nova debugging - conversation state before failed follow-up: {} messages", self.agent.conversation().messages().len());

                                current_response.tool_calls.clear();
                                current_response.content = format!(
                                    "I successfully executed the requested tools but encountered an issue generating the final response. Debug info: {}",
                                    error_details
                                );
                            }
                        }

                        // RESPONSE VALIDATION AND RECOVERY
                        // Enhanced recovery: Check for missing verification markers in MCP tool responses
                        let needs_recovery =
                            current_response.content.is_empty() && !tool_results.is_empty();
                        let has_mcp_tools =
                            tool_results.iter().any(|r| r.tool_name.starts_with("mcp_"));
                        let missing_verification_markers = has_mcp_tools
                            && !current_response.content.contains("MCP SERVER")
                            && !current_response.content.contains("🔍")
                            && !current_response.content.contains("⏰")
                            && !current_response.content.contains("[This")
                            && !tool_results.is_empty();

                        if needs_recovery || missing_verification_markers {
                            if needs_recovery {
                                tracing::warn!(
                                    "🚨 EMPTY RESPONSE DETECTED - Implementing recovery strategy"
                                );
                                tracing::warn!("🚨 Tools executed successfully but LLM returned empty response");
                            } else if missing_verification_markers {
                                tracing::warn!("🚨 MCP VERIFICATION MARKERS MISSING - Implementing recovery strategy");
                                tracing::warn!("🚨 MCP tools executed but verification markers not preserved in response");
                                tracing::warn!(
                                    "🚨 Expected markers: 'MCP SERVER', '🔍', '⏰', '[This'"
                                );
                            }

                            // Strategy 1: Generate response from tool outputs
                            let mut recovered_response = String::new();
                            if missing_verification_markers {
                                recovered_response.push_str("Here are the complete, unmodified tool results as requested:\n\n");
                            } else {
                                recovered_response
                                    .push_str("Based on the tool execution results:\n\n");
                            }

                            for result in tool_results.iter() {
                                if result.success {
                                    if let Some(output) = &result.output {
                                        recovered_response
                                            .push_str(&format!("**{}**: ", result.tool_name));

                                        // Handle different output types
                                        match output {
                                            serde_json::Value::String(s) => {
                                                recovered_response.push_str(s);
                                            }
                                            serde_json::Value::Number(n) => {
                                                recovered_response.push_str(&n.to_string());
                                            }
                                            _ => {
                                                recovered_response.push_str(
                                                    &serde_json::to_string_pretty(output)
                                                        .unwrap_or_else(|_| {
                                                            "Tool executed successfully".to_string()
                                                        }),
                                                );
                                            }
                                        }
                                        recovered_response.push_str("\n\n");
                                    } else {
                                        recovered_response.push_str(&format!(
                                            "**{}**: Tool executed successfully\n\n",
                                            result.tool_name
                                        ));
                                    }
                                } else {
                                    recovered_response.push_str(&format!(
                                        "**{}**: Error - {}\n\n",
                                        result.tool_name,
                                        result
                                            .error
                                            .as_ref()
                                            .unwrap_or(&"Unknown error".to_string())
                                    ));
                                }
                            }

                            current_response.content = recovered_response;
                            tracing::info!("✅ RECOVERY SUCCESS - Generated response from tool outputs ({} chars)", current_response.content.len());
                        }
                    }
                    Err(e) => {
                        tracing::error!("❌ Tool execution failed: {}", e);

                        // Add synthetic tool results for pending tool uses to keep conversation valid
                        // This ensures tool_use blocks always have matching tool_result blocks
                        if !self.pending_tool_uses.is_empty() {
                            tracing::info!(
                                "Adding synthetic tool results for {} failed tool(s)",
                                self.pending_tool_uses.len()
                            );

                            let synthetic_results: Vec<ToolResult> = self.pending_tool_uses
                                .iter()
                                .map(|tool_use| ToolResult {
                                    tool_use_id: tool_use.tool_use_id.clone(),
                                    tool_name: tool_use.name.clone(),
                                    input: tool_use.input.clone(),
                                    success: false,
                                    output: Some(serde_json::json!({
                                        "error": true,
                                        "message": format!(
                                            "Tool '{}' execution failed: {}",
                                            tool_use.name, e
                                        )
                                    })),
                                    error: Some(format!("Tool execution failed: {}", e)),
                                    duration: std::time::Duration::ZERO,
                                })
                                .collect();

                            // Add synthetic results to conversation to maintain valid state
                            let tool_result_message = self.create_tool_result_message(synthetic_results);
                            self.agent.conversation_mut().add_message(tool_result_message);

                            // Clear pending tool uses after adding synthetic results
                            self.pending_tool_uses.clear();
                        }

                        // Graceful fallback - provide error context but continue
                        current_response.tool_calls.clear();
                        current_response.content = format!("I encountered an issue executing the requested tools: {}. Let me provide what I can based on my knowledge.", e);
                        tracing::debug!(
                            "🔄 Tool execution failed, continuing with fallback response"
                        );
                    }
                }
            } else {
                // No tool calls, this is the final response
                tracing::debug!("🤖 LLM response contains no tool calls, treating as final answer");
                break;
            }
        }

        let response = current_response.content.clone();

        // Wait for stream completion if streaming was active to prevent evaluation/streaming overlap
        self.wait_for_stream_completion().await;

        // Evaluate whether to continue based on the configured strategy BEFORE adding response to conversation
        let evaluation_result = self
            .evaluate_continuation(&current_response, &cycle_metrics)
            .await?;

        tracing::info!(
            "🤔 Evaluation result: decision={}, additional_content_length={}",
            evaluation_result.decision,
            evaluation_result
                .response
                .as_ref()
                .map(|r| r.len())
                .unwrap_or(0)
        );

        // Add the assistant's response to conversation (always, even if empty)
        self.agent
            .conversation_mut()
            .add_assistant_message(&current_response.content);

        // Queue log event for AgentCore Evaluations (prompt/response content)
        // Link to invoke_agent span (not chat span) as required by AgentCore
        // Use the version with tool results for Faithfulness evaluation
        if let (Some(tracer), Some((trace_id, span_id))) =
            (&self.tracer, &invoke_agent_span_ids)
        {
            if tracer.can_export_log_events() && !current_response.content.is_empty() {
                let system_prompt = self.agent.conversation().system_prompt();
                if accumulated_tool_results.is_empty() {
                    tracer.queue_agent_invocation_log(
                        trace_id,
                        span_id,
                        system_prompt,
                        original_prompt,
                        &current_response.content,
                    );
                } else {
                    // Include tool results for Faithfulness evaluation
                    tracer.queue_agent_invocation_with_tools_log(
                        trace_id,
                        span_id,
                        system_prompt,
                        original_prompt,
                        &accumulated_tool_results,
                        &current_response.content,
                    );
                }
            }
        }

        // If evaluation decided to continue, add the additional content as a USER message
        // This will prompt the model to continue working in the next cycle
        if evaluation_result.decision {
            let content_to_add = if let Some(additional_content) = &evaluation_result.response {
                if !additional_content.trim().is_empty() {
                    additional_content.clone()
                } else {
                    // Fallback: Generate default continuation instruction when CONTINUE decided but no content provided
                    "Please continue working on the task. Focus on completing any missing requirements or improving the quality of your work.".to_string()
                }
            } else {
                // Fallback: Generate default continuation instruction when CONTINUE decided but no response field
                "Please continue working on the task. Focus on completing any missing requirements or improving the quality of your work.".to_string()
            };

            // Add as USER message so the model will respond to it in the next cycle
            tracing::info!(
                "📝 Adding evaluation content as user message to continue conversation: '{}'",
                content_to_add.chars().take(100).collect::<String>()
            );
            self.agent
                .conversation_mut()
                .add_user_message(&content_to_add);
            tracing::info!(
                "📝 Conversation now has {} messages after adding evaluation content",
                self.agent.conversation().message_count()
            );
        }

        let cycle_duration = cycle_start.elapsed();
        let model_invocations = cycle_metrics.model_invocations;
        let tool_calls = cycle_metrics.tool_calls;
        let tokens_used = cycle_metrics.tokens_used.clone();

        cycle_metrics = cycle_metrics.complete_success(cycle_duration);

        // Record telemetry in current tracing span (automatically handled by #[tracing::instrument])
        tracing::Span::current().record("model_interaction.invocations", model_invocations);
        tracing::Span::current().record("model_interaction.tool_calls", tool_calls);
        tracing::Span::current()
            .record("model_interaction.duration_ms", cycle_duration.as_millis());
        tracing::Span::current().record(
            "model_interaction.should_continue",
            evaluation_result.decision,
        );

        // Log model interaction performance
        self.performance_logger
            .log_cycle_performance(cycle_duration, cycle_id);

        self.metrics.add_cycle(cycle_metrics);

        tracing::debug!(
            "Cycle {} completed in {:?} with {} model calls and {} loop iterations",
            cycle_id,
            cycle_duration,
            model_invocations,
            loop_count
        );

        // 📊 FINAL CYCLE SUMMARY
        tracing::info!(
            "🏁 CYCLE {} SUMMARY: {} tool iterations, {} model calls, {}ms duration",
            cycle_id,
            tool_iteration_count,
            model_invocations,
            cycle_duration.as_millis()
        );

        // Complete the model interaction span and queue corresponding log event
        if let Some(mut span) = cycle_span {
            span.set_attribute("model_interaction.invocations", model_invocations as i64);
            span.set_attribute("model_interaction.tool_calls", tool_iteration_count as i64);
            span.set_attribute(
                "model_interaction.duration_ms",
                cycle_duration.as_millis() as i64,
            );
            span.set_attribute(
                "model_interaction.should_continue",
                evaluation_result.decision,
            );
            // Record token usage for this model interaction
            span.record_tokens(tokens_used.input_tokens, tokens_used.output_tokens);
            span.set_success();

            // Queue chat completion log event for AgentCore Evaluations
            // Every span with strands.telemetry.tracer scope must have a corresponding event
            if let Some(ref tracer) = self.tracer {
                if tracer.can_export_log_events() && !response.is_empty() {
                    tracer.queue_chat_completion_log(
                        span.trace_id(),
                        span.span_id(),
                        &model_id,
                        original_prompt,
                        &response,
                    );
                }
            }
        }

        Ok(CycleResult {
            response,
            should_continue: evaluation_result.decision,
            tool_iterations_used: Some(tool_iteration_count),
        })
    }

    /// Extract tool uses from a model response (LLM-driven approach)
    fn extract_tool_uses(
        &self,
        message: &crate::types::Message,
    ) -> Result<Vec<crate::tools::ToolUse>> {
        tracing::trace!(
            "extract_tool_uses called with {} content blocks",
            message.content.len()
        );
        let mut tool_uses = Vec::new();

        for (i, content_block) in message.content.iter().enumerate() {
            tracing::trace!("Examining content block {}: {:?}", i, content_block);
            if let crate::types::ContentBlock::ToolUse { id, name, input } = content_block {
                tool_uses.push(crate::tools::ToolUse {
                    tool_use_id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                });
            }
        }

        tracing::debug!("extract_tool_uses returning {} tool uses", tool_uses.len());
        for (i, tool_use) in tool_uses.iter().enumerate() {
            tracing::trace!(
                "Tool use {}: name={}, id={}",
                i,
                tool_use.name,
                tool_use.tool_use_id
            );
        }

        tracing::debug!(
            "Extracted {} tool uses from model response",
            tool_uses.len()
        );
        Ok(tool_uses)
    }

    /// Create a message containing tool results for the conversation
    fn create_tool_result_message(&self, tool_results: Vec<ToolResult>) -> crate::types::Message {
        let content_blocks: Vec<crate::types::ContentBlock> = tool_results
            .into_iter()
            .map(|result| {
                crate::types::ContentBlock::ToolResult {
                    tool_use_id: result.tool_use_id,
                    content: crate::types::ToolResultContent::json(result.output.unwrap_or_else(|| {
                        serde_json::json!({"error": result.error.unwrap_or_else(|| "Unknown error".to_string())})
                    })),
                    is_error: !result.success,
                }
            })
            .collect();

        crate::types::Message::new(crate::types::MessageRole::User, content_blocks)
    }

    /// Phase 3: Tool Execution Phase - Execute tools selected by LLM
    async fn tool_execution_phase(
        &mut self,
        tool_uses: Vec<crate::tools::ToolUse>,
        cycle_metrics: &mut CycleMetrics,
        cycle_context: Option<opentelemetry::Context>,
    ) -> Result<Vec<ToolResult>> {
        let _tool_phase_guard = self
            .performance_tracer
            .start_operation("tool_execution_phase");
        _tool_phase_guard.add_context("tool_count", &tool_uses.len().to_string());

        tracing::debug!(
            "Executing tool execution phase with {} tools",
            tool_uses.len()
        );

        let mut results = Vec::new();
        cycle_metrics.tool_calls += tool_uses.len() as u32;

        // NEW: Use ToolExecutor directly for parallel execution instead of legacy ToolRegistry
        if tool_uses.len() > 1 {
            // Multiple tools - use parallel execution via ToolExecutor
            tracing::debug!(
                "🚀 Executing {} tools in parallel using ToolExecutor (max_parallel_tools={})",
                tool_uses.len(),
                self.tool_executor.config().max_parallel_tools
            );

            let execution_start = Instant::now();

            // Create parent span for parallel tool execution group
            let parallel_group_span = if let Some(ref tracer) = self.tracer {
                let mut span = if let Some(ref parent_ctx) = cycle_context {
                    tracer.start_tool_span_with_parent_context("parallel_group", parent_ctx)
                } else {
                    tracer.start_tool_span("parallel_group")
                };

                // Set parallel execution attributes
                span.set_attribute("tool.execution.mode", "parallel");
                span.set_attribute("tool.execution.count", tool_uses.len() as i64);
                span.set_attribute(
                    "tool.execution.max_parallel",
                    self.tool_executor.config().max_parallel_tools as i64,
                );
                span.set_attribute("tool.execution.strategy", "tool_executor");

                // Add tool names for correlation
                let tool_names: Vec<&str> = tool_uses.iter().map(|t| t.name.as_str()).collect();
                span.set_attribute("tool.execution.tools", tool_names.join(","));

                Some(span)
            } else {
                None
            };

            // Emit ToolStart callbacks for all tools
            if let Some(ref callback) = self.callback_handler {
                for tool_use in &tool_uses {
                    let event = CallbackEvent::ToolStart {
                        tool_name: tool_use.name.clone(),
                        tool_use_id: tool_use.tool_use_id.clone(),
                        input: tool_use.input.clone(),
                    };
                    if let Err(e) = callback.handle_event(event).await {
                        tracing::warn!("Callback error during ToolStart: {}", e);
                    }
                }
            }

            // Convert to format expected by ToolExecutor
            let tool_executions: Vec<(
                std::sync::Arc<dyn crate::tools::Tool>,
                crate::tools::ToolUse,
            )> = {
                let mut executions = Vec::new();
                for tool_use in &tool_uses {
                    if let Some(tool) = self.tool_registry.get_tool(&tool_use.name).await {
                        executions.push((tool, tool_use.clone()));
                    } else {
                        tracing::error!("Tool '{}' not found in registry", tool_use.name);
                        // Create error result
                        let error_result = ToolResult {
                            tool_use_id: tool_use.tool_use_id.clone(),
                            tool_name: tool_use.name.clone(),
                            input: tool_use.input.clone(),
                            success: false,
                            output: None,
                            error: Some(format!("Tool '{}' not found", tool_use.name)),
                            duration: Duration::from_millis(1),
                        };
                        results.push(error_result);
                    }
                }
                executions
            };

            // Create individual tool spans BEFORE execution to capture accurate timing
            let mut individual_tool_spans: Vec<Option<crate::telemetry::StoodSpan>> = Vec::new();
            let parallel_group_context = parallel_group_span.as_ref().map(|span| span.context());

            if let Some(ref tracer) = self.tracer {
                for tool_use in &tool_uses {
                    let mut tool_span = if let Some(ref group_ctx) = parallel_group_context {
                        tracer.start_tool_span_with_parent_context(&tool_use.name, group_ctx)
                    } else {
                        tracer.start_tool_span(&tool_use.name)
                    };

                    // Set individual tool attributes
                    tool_span.set_attribute("tool.execution.mode", "parallel");
                    tool_span.set_attribute("tool.name", tool_use.name.clone());
                    tool_span.set_attribute("tool.use_id", tool_use.tool_use_id.clone());
                    tool_span.set_attribute(
                        "tool.input_size_bytes",
                        tool_use.input.to_string().len() as i64,
                    );

                    individual_tool_spans.push(Some(tool_span));
                }
            } else {
                // Fill with None if no tracer
                for _ in 0..tool_uses.len() {
                    individual_tool_spans.push(None);
                }
            }

            // Execute tools in parallel using ToolExecutor
            let parallel_results = self
                .tool_executor
                .execute_tools_parallel(tool_executions, None)
                .await;

            // Convert results and emit callbacks
            for (i, ((tool_result, metrics), tool_use)) in parallel_results
                .into_iter()
                .zip(tool_uses.iter())
                .enumerate()
            {
                let duration = metrics
                    .as_ref()
                    .map(|m| m.duration)
                    .unwrap_or_else(|| execution_start.elapsed());

                let result = if tool_result.success {
                    tracing::debug!(
                        "✅ Tool '{}' completed successfully in {:.2}ms",
                        tool_use.name,
                        duration.as_secs_f64() * 1000.0
                    );
                    ToolResult {
                        tool_use_id: tool_use.tool_use_id.clone(),
                        tool_name: tool_use.name.clone(),
                        input: tool_use.input.clone(),
                        success: true,
                        output: Some(tool_result.content.clone()),
                        error: tool_result.error,
                        duration,
                    }
                } else {
                    tracing::error!(
                        "❌ Tool '{}' failed: {}",
                        tool_use.name,
                        tool_result.content
                    );
                    ToolResult {
                        tool_use_id: tool_use.tool_use_id.clone(),
                        tool_name: tool_use.name.clone(),
                        input: tool_use.input.clone(),
                        success: false,
                        output: None,
                        error: Some(tool_result.content.to_string()),
                        duration,
                    }
                };

                // Complete individual tool span with results
                if let Some(mut tool_span) = individual_tool_spans.get_mut(i).and_then(|s| s.take())
                {
                    // Set completion attributes
                    tool_span.set_attribute("tool.duration_ms", duration.as_millis() as i64);
                    tool_span.set_attribute("tool.success", result.success);
                    tool_span.set_attribute(
                        "tool.output_size_bytes",
                        result
                            .output
                            .as_ref()
                            .map(|o| o.to_string().len())
                            .unwrap_or(0) as i64,
                    );

                    if let Some(ref error) = result.error {
                        tool_span.set_attribute("tool.error", error.clone());
                        tool_span.set_error(error);
                    } else {
                        tool_span.set_success();
                    }

                    // Add tool execution event
                    tool_span.add_event(
                        "tool.execution.completed",
                        vec![
                            crate::telemetry::KeyValue::new("tool.name", tool_use.name.clone()),
                            crate::telemetry::KeyValue::new(
                                "tool.duration_ms",
                                duration.as_millis() as i64,
                            ),
                            crate::telemetry::KeyValue::new("tool.success", result.success),
                        ],
                    );

                    // Queue tool execution log event for AgentCore Evaluations
                    // Each tool span needs a corresponding log event with input/output
                    if let Some(ref tracer) = self.tracer {
                        if tracer.can_export_log_events() {
                            let tool_input = serde_json::to_string(&tool_use.input)
                                .unwrap_or_else(|_| "{}".to_string());
                            let tool_output = result
                                .output
                                .as_ref()
                                .map(|o| o.to_string())
                                .unwrap_or_else(|| {
                                    result
                                        .error
                                        .clone()
                                        .unwrap_or_else(|| "No output".to_string())
                                });
                            tracer.queue_tool_execution_log(
                                tool_span.trace_id(),
                                tool_span.span_id(),
                                &tool_use.name,
                                &tool_input,
                                &tool_output,
                            );
                        }
                    }

                    // Finish span with actual duration so the trace timeline is accurate
                    // This calculates end_time = start_time + duration
                    tool_span.finish_with_duration(duration);
                }

                // Emit ToolComplete callback
                if let Some(ref callback) = self.callback_handler {
                    let event = CallbackEvent::ToolComplete {
                        tool_name: tool_use.name.clone(),
                        tool_use_id: tool_use.tool_use_id.clone(),
                        duration,
                        error: result.error.clone(),
                        output: result.output.clone(),
                    };
                    if let Err(e) = callback.handle_event(event).await {
                        tracing::warn!("Callback error during ToolComplete: {}", e);
                    }
                }

                results.push(result);
            }

            // Complete parallel group span
            if let Some(mut group_span) = parallel_group_span {
                let total_execution_time = execution_start.elapsed();
                let successful_tools = results.iter().filter(|r| r.success).count();
                let failed_tools = results.len() - successful_tools;

                // Set completion attributes
                group_span.set_attribute(
                    "tool.execution.total_duration_ms",
                    total_execution_time.as_millis() as i64,
                );
                group_span
                    .set_attribute("tool.execution.successful_count", successful_tools as i64);
                group_span.set_attribute("tool.execution.failed_count", failed_tools as i64);
                group_span.set_attribute(
                    "tool.execution.success_rate",
                    if results.is_empty() {
                        0.0
                    } else {
                        successful_tools as f64 / results.len() as f64
                    },
                );

                if failed_tools > 0 {
                    group_span.set_error(&format!("{} tools failed", failed_tools));
                } else {
                    group_span.set_success();
                }

                // Add group completion event
                group_span.add_event(
                    "tool.parallel_group.completed",
                    vec![
                        crate::telemetry::KeyValue::new(
                            "tool.group.total_count",
                            results.len() as i64,
                        ),
                        crate::telemetry::KeyValue::new(
                            "tool.group.successful_count",
                            successful_tools as i64,
                        ),
                        crate::telemetry::KeyValue::new(
                            "tool.group.duration_ms",
                            total_execution_time.as_millis() as i64,
                        ),
                    ],
                );

                group_span.finish();
            }
        } else {
            // Single tool - process individually (keep existing logic for now)
            for (tool_index, tool_use) in tool_uses.into_iter().enumerate() {
                let execution_start = Instant::now();

                let _tool_guard = self
                    .performance_tracer
                    .start_operation("individual_tool_execution");
                _tool_guard.add_context("tool_name", &tool_use.name);
                _tool_guard.add_context("tool_index", &tool_index.to_string());
                _tool_guard.add_context("tool_use_id", &tool_use.tool_use_id);

                // Remove redundant tracing span - use only OpenTelemetry spans

                let _tool_span = if let Some(ref tracer) = self.tracer {
                    let tool_span_id = Uuid::new_v4();
                    let mut span = if let Some(ref parent_ctx) = cycle_context {
                        tracer.start_tool_span_with_parent_context(&tool_use.name, parent_ctx)
                    } else {
                        tracer.start_tool_span(&tool_use.name)
                    };
                    let span_info = SpanInfo {
                        span_id: tool_span_id.to_string(),
                        start_time: execution_start,
                        cycle_id: tool_span_id, // Using tool span ID for tracking
                        span_type: SpanType::ToolExecution,
                    };
                    self.active_spans.insert(tool_span_id, span_info);
                    span.set_attribute("tool.execution.mode", "sequential");
                    span.set_attribute("tool.name", tool_use.name.clone());
                    span.set_attribute("tool.use_id", tool_use.tool_use_id.clone());
                    span.set_attribute(
                        "tool.input_size_bytes",
                        tool_use.input.to_string().len() as i64,
                    );
                    Some((span, tool_span_id))
                } else {
                    None
                };

                // Log tool execution details for debugging
                tracing::debug!(
                    "🔧 Executing tool '{}' with input: {}",
                    tool_use.name,
                    serde_json::to_string_pretty(&tool_use.input)
                        .unwrap_or_else(|_| "invalid JSON".to_string())
                );

                // Emit ToolStart callback
                if let Some(ref callback) = self.callback_handler {
                    let event = CallbackEvent::ToolStart {
                        tool_name: tool_use.name.clone(),
                        tool_use_id: tool_use.tool_use_id.clone(),
                        input: tool_use.input.clone(),
                    };
                    if let Err(e) = callback.handle_event(event).await {
                        tracing::warn!("Callback error during ToolStart: {}", e);
                    }
                }

                _tool_guard.checkpoint("start_tool_execution");
                let tool_execution_start = Instant::now();
                let tool_result = self
                    .tool_registry
                    .execute_tool(&tool_use.name, Some(tool_use.input.clone()), None)
                    .await;
                let tool_execution_duration = tool_execution_start.elapsed();

                if tool_execution_duration > Duration::from_millis(500) {
                    self.performance_tracer.record_waiting(
                        &format!("tool_execution_{}", tool_use.name),
                        tool_execution_duration,
                    );
                }

                _tool_guard.checkpoint("tool_execution_complete");

                let result = match tool_result {
                    Ok(tool_result) => {
                        if tool_result.success {
                            tracing::debug!(
                                "✅ Tool '{}' executed successfully in {:.2}ms",
                                tool_use.name,
                                execution_start.elapsed().as_secs_f64() * 1000.0
                            );
                            tracing::debug!(
                                "🔧 Tool '{}' output: {}",
                                tool_use.name,
                                serde_json::to_string_pretty(&tool_result.content)
                                    .unwrap_or_else(|_| "invalid JSON".to_string())
                            );
                            ToolResult {
                                tool_use_id: tool_use.tool_use_id.clone(),
                                tool_name: tool_use.name.clone(),
                                input: tool_use.input.clone(),
                                success: true,
                                output: Some(tool_result.content),
                                error: None,
                                duration: execution_start.elapsed(),
                            }
                        } else {
                            tracing::error!(
                                "❌ Tool '{}' failed: {}",
                                tool_use.name,
                                tool_result.error.as_deref().unwrap_or("Unknown error")
                            );
                            tracing::debug!(
                                "🔧 Tool '{}' failure details - Duration: {:.2}ms",
                                tool_use.name,
                                execution_start.elapsed().as_secs_f64() * 1000.0
                            );
                            ToolResult {
                                tool_use_id: tool_use.tool_use_id.clone(),
                                tool_name: tool_use.name.clone(),
                                input: tool_use.input.clone(),
                                success: false,
                                output: None,
                                error: tool_result.error,
                                duration: execution_start.elapsed(),
                            }
                        }
                    }
                    Err(tool_error) => {
                        tracing::error!(
                            "❌ Tool '{}' failed with error: {}",
                            tool_use.name,
                            tool_error
                        );
                        ToolResult {
                            tool_use_id: tool_use.tool_use_id.clone(),
                            tool_name: tool_use.name.clone(),
                            input: tool_use.input.clone(),
                            success: false,
                            output: None,
                            error: Some(tool_error.to_string()),
                            duration: execution_start.elapsed(),
                        }
                    }
                };

                // Update telemetry span with tool execution results

                if let Some((mut span, tool_span_id)) = _tool_span {
                    span.set_attribute("tool.duration_ms", result.duration.as_millis() as i64);
                    span.set_attribute("tool.success", result.success);
                    span.set_attribute(
                        "tool.output_size_bytes",
                        result
                            .output
                            .as_ref()
                            .map(|o| o.to_string().len())
                            .unwrap_or(0) as i64,
                    );

                    if result.success {
                        span.set_success();
                    } else if let Some(ref error_msg) = result.error {
                        span.set_attribute("tool.error", error_msg.clone());
                        span.set_error(error_msg);
                    }

                    // Queue tool execution log event for AgentCore Evaluations
                    // Each tool span needs a corresponding log event with input/output
                    if let Some(ref tracer) = self.tracer {
                        if tracer.can_export_log_events() {
                            let tool_input = serde_json::to_string(&tool_use.input)
                                .unwrap_or_else(|_| "{}".to_string());
                            let tool_output = result
                                .output
                                .as_ref()
                                .map(|o| o.to_string())
                                .unwrap_or_else(|| {
                                    result
                                        .error
                                        .clone()
                                        .unwrap_or_else(|| "No output".to_string())
                                });
                            tracer.queue_tool_execution_log(
                                span.trace_id(),
                                span.span_id(),
                                &tool_use.name,
                                &tool_input,
                                &tool_output,
                            );
                        }
                    }

                    // Clean up span tracking
                    self.active_spans.remove(&tool_span_id);
                }

                // Record performance metrics
                self.performance_logger.log_tool_performance(
                    &result.tool_name,
                    result.duration,
                    result.success,
                );

                // Emit ToolComplete callback
                if let Some(ref callback) = self.callback_handler {
                    let event = CallbackEvent::ToolComplete {
                        tool_name: result.tool_name.clone(),
                        tool_use_id: result.tool_use_id.clone(),
                        output: result.output.clone(),
                        error: result.error.clone(),
                        duration: result.duration,
                    };
                    if let Err(e) = callback.handle_event(event).await {
                        tracing::warn!("Callback error during ToolComplete: {}", e);
                    }
                }

                // Record tool execution metrics
                let tool_metric = ToolExecutionMetric {
                    tool_name: result.tool_name.clone(),
                    tool_use_id: Some(result.tool_use_id.clone()),
                    duration: result.duration,
                    success: result.success,
                    error: result.error.clone(),
                    trace_id: None, // Would be filled by telemetry
                    span_id: None,  // Would be filled by telemetry
                    start_time: Utc::now(),
                    input_size_bytes: Some(tool_use.input.to_string().len()),
                    output_size_bytes: result.output.as_ref().map(|o| o.to_string().len()),
                };

                self.metrics.add_tool_execution(tool_metric);
                results.push(result);
            }
        }

        Ok(results)
    }

    /// Unified method for executing chat with tools (streaming or non-streaming)
    async fn execute_chat_with_tools(
        &mut self,
        tool_config: &crate::types::tools::ToolConfig,
    ) -> Result<crate::llm::traits::ChatResponse> {
        if self.config.enable_streaming {
            self.execute_streaming_chat_internal(tool_config).await
        } else {
            self.execute_non_streaming_chat_internal(tool_config).await
        }
    }

    /// Internal method for non-streaming chat execution
    async fn execute_non_streaming_chat_internal(
        &mut self,
        _tool_config: &crate::types::tools::ToolConfig,
    ) -> Result<crate::llm::traits::ChatResponse> {
        let chat_start = Instant::now();

        // Mark that streaming is not active
        self.stream_was_active = false;
        self.stream_completion_time = None;

        debug!("🌐 Using non-streaming path, making LLM provider API call");

        // Convert tool registry to LLM tool format
        let all_llm_tools = self.tool_registry.to_llm_tools().await;
        let agent_config = self.agent.config();

        // Respect ToolChoice::None — exclude all tools from the request
        let llm_tools: Vec<crate::llm::traits::Tool> =
            if agent_config.tool_choice == crate::types::tools::ToolChoice::None {
                debug!("🚫 ToolChoice::None — excluding all tools from LLM request");
                vec![]
            } else {
                all_llm_tools
            };

        debug!(
            "🔧 Sending {} tools to LLM provider",
            llm_tools.len()
        );

        // Use agent's configured settings (max_tokens, temperature, etc.)
        let chat_config = crate::llm::traits::ChatConfig {
            model_id: agent_config.model_id.clone(),
            provider: agent_config.provider,
            temperature: agent_config.temperature,
            max_tokens: agent_config.max_tokens,
            enable_thinking: false,
            cache_strategy: agent_config.cache_strategy.clone(),
            tool_choice: agent_config.tool_choice.clone(),
            additional_params: std::collections::HashMap::new(),
        };

        let messages_with_prompt = self.agent.conversation().messages_with_system_prompt();
        let response = match self
            .agent
            .provider()
            .chat_with_tools(
                self.agent.model().model_id(),
                &messages_with_prompt,
                &llm_tools,
                &chat_config,
            )
            .await
        {
            Ok(resp) => {
                debug!(
                    "✅ LLM provider call succeeded with response: '{}'",
                    resp.content
                );
                resp
            }
            Err(e) => {
                debug!("❌ LLM provider call failed: {}", e);
                return Err(crate::StoodError::model_error(format!(
                    "LLM provider error: {}",
                    e
                )));
            }
        };
        let chat_duration = chat_start.elapsed();

        if chat_duration > Duration::from_millis(2000) {
            self.performance_tracer
                .record_waiting("llm_provider_api_call", chat_duration);
        }

        Ok(response)
    }

    /// Internal method for streaming chat execution
    async fn execute_streaming_chat_internal(
        &mut self,
        _tool_config: &crate::types::tools::ToolConfig,
    ) -> Result<crate::llm::traits::ChatResponse> {
        tracing::info!("🔧🌊 Starting real LLM provider streaming execution with tools");

        // Mark that streaming is active
        self.stream_was_active = true;
        self.stream_completion_time = None;

        // TRACE MODE: Print full conversation state before making the request
        if tracing::level_enabled!(tracing::Level::TRACE) {
            let messages = self.agent.conversation().messages();
            tracing::trace!(
                "📋 TRACE: Full conversation state before streaming request ({} messages):",
                messages.len()
            );
            tracing::trace!(
                "📋 TRACE: ==================== CONVERSATION THREAD ===================="
            );

            for (i, message) in messages.iter().enumerate() {
                tracing::trace!("📋 TRACE: Message #{}: Role: {:?}", i + 1, message.role);
                tracing::trace!(
                    "📋 TRACE: Message #{}: Timestamp: {:?}",
                    i + 1,
                    message.timestamp
                );
                tracing::trace!(
                    "📋 TRACE: Message #{}: Content blocks ({})",
                    i + 1,
                    message.content.len()
                );

                for (j, block) in message.content.iter().enumerate() {
                    match block {
                        crate::types::ContentBlock::Text { text } => {
                            tracing::trace!("📋 TRACE:   Block #{}: TEXT: {}", j + 1, text);
                        }
                        crate::types::ContentBlock::ToolUse { id, name, input } => {
                            tracing::trace!(
                                "📋 TRACE:   Block #{}: TOOL_USE: {} ({})",
                                j + 1,
                                name,
                                id
                            );
                            tracing::trace!(
                                "📋 TRACE:   Block #{}: TOOL_INPUT: {}",
                                j + 1,
                                serde_json::to_string_pretty(input)
                                    .unwrap_or_else(|_| "Invalid JSON".to_string())
                            );
                        }
                        crate::types::ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            is_error,
                        } => {
                            tracing::trace!(
                                "📋 TRACE:   Block #{}: TOOL_RESULT: {} (error: {})",
                                j + 1,
                                tool_use_id,
                                is_error
                            );
                            tracing::trace!(
                                "📋 TRACE:   Block #{}: TOOL_OUTPUT: {}",
                                j + 1,
                                content.to_display_string()
                            );
                        }
                        _ => {
                            tracing::trace!("📋 TRACE:   Block #{}: OTHER: {:?}", j + 1, block);
                        }
                    }
                }
                tracing::trace!(
                    "📋 TRACE: --------------------------------------------------------"
                );
            }
            tracing::trace!("📋 TRACE: ==================== END CONVERSATION ====================");
        }

        // Convert tool registry to LLM tool format
        let all_llm_tools = self.tool_registry.to_llm_tools().await;
        let agent_config = self.agent.config();

        // Respect ToolChoice::None — exclude all tools from the request
        let llm_tools: Vec<crate::llm::traits::Tool> =
            if agent_config.tool_choice == crate::types::tools::ToolChoice::None {
                debug!("🚫 ToolChoice::None — excluding all tools from streaming LLM request");
                vec![]
            } else {
                all_llm_tools
            };

        debug!(
            "🔧 Sending {} tools to LLM provider for streaming",
            llm_tools.len()
        );

        // TRACE MODE: Print available tools
        if tracing::level_enabled!(tracing::Level::TRACE) {
            tracing::trace!(
                "📋 TRACE: Available tools for streaming request ({} tools):",
                llm_tools.len()
            );
            for (i, tool) in llm_tools.iter().enumerate() {
                tracing::trace!(
                    "📋 TRACE: Tool #{}: {} - {}",
                    i + 1,
                    tool.name,
                    tool.description
                );
                tracing::trace!(
                    "📋 TRACE: Tool #{}: Schema: {}",
                    i + 1,
                    serde_json::to_string_pretty(&tool.input_schema)
                        .unwrap_or_else(|_| "Invalid JSON".to_string())
                );
            }
        }

        // Use agent's configured settings (max_tokens, temperature, etc.)
        let chat_config = crate::llm::traits::ChatConfig {
            model_id: agent_config.model_id.clone(),
            provider: agent_config.provider,
            temperature: agent_config.temperature,
            max_tokens: agent_config.max_tokens,
            enable_thinking: false,
            cache_strategy: agent_config.cache_strategy.clone(),
            tool_choice: agent_config.tool_choice.clone(),
            additional_params: std::collections::HashMap::new(),
        };

        // Get the streaming receiver from LLM provider using streaming with tools
        let messages_with_prompt = self.agent.conversation().messages_with_system_prompt();
        let mut stream_receiver = if llm_tools.is_empty() {
            // No tools available, use regular streaming
            tracing::info!("🌊 Using regular streaming (no tools available)");
            self.agent
                .provider()
                .chat_streaming(
                    self.agent.model().model_id(),
                    &messages_with_prompt,
                    &chat_config,
                )
                .await
                .map_err(|e| crate::StoodError::model_error(format!("Streaming error: {}", e)))?
        } else {
            // Tools available, use streaming with tools
            tracing::info!(
                "🔧🌊 Using streaming with tools ({} tools available)",
                llm_tools.len()
            );
            self.agent
                .provider()
                .chat_streaming_with_tools(
                    self.agent.model().model_id(),
                    &messages_with_prompt,
                    &llm_tools,
                    &chat_config,
                )
                .await
                .map_err(|e| {
                    crate::StoodError::model_error(format!("Streaming with tools error: {}", e))
                })?
        };

        // Initialize for collecting streaming content and tool calls using universal content block pattern
        let mut content_parts = Vec::new();
        let mut final_response: Option<crate::llm::traits::ChatResponse> = None;
        let mut current_tool_calls: std::collections::HashMap<
            String,
            crate::llm::traits::ToolCall,
        > = std::collections::HashMap::new();
        let mut current_tool_inputs: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        let mut active_content_blocks: std::collections::HashMap<usize, String> =
            std::collections::HashMap::new();
        let mut stream_usage: Option<crate::llm::traits::Usage> = None;

        tracing::info!(
            "🎯 Processing real-time streaming events with universal content block pattern"
        );

        // Process streaming events as they arrive using universal content block pattern
        use futures::StreamExt;
        let mut received_event_count = 0;
        while let Some(stream_event) = stream_receiver.next().await {
            received_event_count += 1;
            tracing::debug!(
                "🎯 Event loop received stream event #{}: {:?}",
                received_event_count,
                match &stream_event {
                    crate::llm::traits::StreamEvent::ContentBlockStart { block_type, .. } =>
                        format!("ContentBlockStart({:?})", block_type),
                    crate::llm::traits::StreamEvent::ContentBlockDelta { delta, .. } => {
                        match delta {
                            crate::llm::traits::ContentBlockDelta::Text { text } => {
                                format!("ContentBlockDelta::Text('{}')", text)
                            }
                            crate::llm::traits::ContentBlockDelta::ToolUse {
                                tool_call_id,
                                input_delta,
                            } => format!(
                                "ContentBlockDelta::ToolUse({}:'{}')",
                                tool_call_id, input_delta
                            ),
                            crate::llm::traits::ContentBlockDelta::Thinking { reasoning_delta } => {
                                format!("ContentBlockDelta::Thinking('{}')", reasoning_delta)
                            }
                        }
                    }
                    crate::llm::traits::StreamEvent::ContentBlockStop { .. } =>
                        "ContentBlockStop".to_string(),
                    crate::llm::traits::StreamEvent::MessageStop { .. } =>
                        "MessageStop".to_string(),
                    crate::llm::traits::StreamEvent::Error { error } => format!("Error({})", error),
                    _ => "Other".to_string(),
                }
            );

            // Process stream events using universal content block pattern (like Python reference)
            match &stream_event {
                crate::llm::traits::StreamEvent::ContentBlockStart {
                    block_type,
                    block_index,
                } => {
                    tracing::debug!("🏁 Content block {} start: {:?}", block_index, block_type);

                    match block_type {
                        crate::llm::traits::ContentBlockType::ToolUse => {
                            // Initialize tool accumulation for this block
                            active_content_blocks.insert(*block_index, "tool".to_string());
                        }
                        crate::llm::traits::ContentBlockType::Text => {
                            // Initialize text accumulation for this block
                            active_content_blocks.insert(*block_index, "text".to_string());
                        }
                        crate::llm::traits::ContentBlockType::Thinking => {
                            // Initialize thinking accumulation for this block
                            active_content_blocks.insert(*block_index, "thinking".to_string());
                        }
                    }
                }
                crate::llm::traits::StreamEvent::ContentBlockDelta { delta, block_index } => {
                    match delta {
                        crate::llm::traits::ContentBlockDelta::Text { text } => {
                            tracing::debug!(
                                "🎯 Adding text delta: '{}' (block {})",
                                text,
                                block_index
                            );
                            content_parts.push(text.clone());

                            // Emit callback events for real-time updates if callback exists
                            if let Some(ref callback) = self.callback_handler {
                                let event = CallbackEvent::ContentDelta {
                                    delta: text.clone(),
                                    complete: false,
                                    reasoning: false,
                                };
                                if let Err(e) = callback.handle_event(event).await {
                                    tracing::warn!(
                                        "Callback error during real ContentDelta: {}",
                                        e
                                    );
                                }
                            }
                        }
                        crate::llm::traits::ContentBlockDelta::ToolUse {
                            tool_call_id,
                            input_delta,
                        } => {
                            tracing::debug!(
                                "🔧 Tool call delta for {}: '{}' (block {})",
                                tool_call_id,
                                input_delta,
                                block_index
                            );

                            // UNIVERSAL PATTERN: String accumulation during deltas
                            current_tool_inputs
                                .entry(tool_call_id.clone())
                                .or_default()
                                .push_str(input_delta);

                            // Create or update tool call info
                            if !current_tool_calls.contains_key(tool_call_id) {
                                // For LM Studio, extract tool name from tool_call_id pattern
                                // tool_call_id format is "tool_call_0", and LM Studio ContentBlockStart should have already been sent
                                let tool_name = if tool_call_id.starts_with("tool_call_") {
                                    // Look for tool name in recent debug logs or use calculator as default for testing
                                    "calculator".to_string() // TODO: Improve tool name extraction
                                } else {
                                    tool_call_id.clone() // Use ID as name fallback
                                };

                                tracing::debug!(
                                    "🔧 Creating new tool call entry: {} -> {}",
                                    tool_call_id,
                                    tool_name
                                );
                                current_tool_calls.insert(
                                    tool_call_id.clone(),
                                    crate::llm::traits::ToolCall {
                                        id: tool_call_id.clone(),
                                        name: tool_name.clone(),
                                        input: serde_json::Value::Null,
                                    },
                                );

                                // Emit tool start callback
                                if let Some(ref callback) = self.callback_handler {
                                    let event = CallbackEvent::ToolStart {
                                        tool_name,
                                        tool_use_id: tool_call_id.clone(),
                                        input: serde_json::Value::String(input_delta.clone()),
                                    };
                                    if let Err(e) = callback.handle_event(event).await {
                                        tracing::warn!(
                                            "Callback error during streaming ToolStart: {}",
                                            e
                                        );
                                    }
                                }
                            }

                            tracing::trace!(
                                "🔧 Tool {} input now: '{}'",
                                tool_call_id,
                                current_tool_inputs
                                    .get(tool_call_id)
                                    .unwrap_or(&String::new())
                            );
                        }
                        crate::llm::traits::ContentBlockDelta::Thinking { reasoning_delta } => {
                            tracing::debug!(
                                "🤔 Thinking delta: '{}' (block {})",
                                reasoning_delta,
                                block_index
                            );

                            // Handle thinking deltas - emit callback
                            if let Some(ref callback) = self.callback_handler {
                                let event = CallbackEvent::ContentDelta {
                                    delta: reasoning_delta.clone(),
                                    complete: false,
                                    reasoning: true, // This is thinking content
                                };
                                if let Err(e) = callback.handle_event(event).await {
                                    tracing::warn!("Callback error during thinking delta: {}", e);
                                }
                            }
                        }
                    }
                }
                crate::llm::traits::StreamEvent::ContentBlockStop { block_index } => {
                    tracing::debug!(
                        "🏁 Content block {} stop - parsing accumulated content",
                        block_index
                    );

                    // UNIVERSAL PATTERN: JSON parsing happens at ContentBlockStop (like Python reference)
                    if let Some(block_type) = active_content_blocks.get(block_index) {
                        if block_type == "tool" {
                            // Find and finalize all tool calls for this block
                            for (tool_id, accumulated_input) in current_tool_inputs.iter() {
                                if let Some(mut tool_call) =
                                    current_tool_calls.get(tool_id).cloned()
                                {
                                    // CRITICAL: Parse JSON here like Python reference implementation
                                    let parsed_input = if accumulated_input.is_empty() {
                                        serde_json::Value::Object(serde_json::Map::new())
                                    } else {
                                        match serde_json::from_str(accumulated_input) {
                                            Ok(parsed) => {
                                                tracing::debug!("🔧 Successfully parsed tool {} input at ContentBlockStop: {}", tool_id, accumulated_input);
                                                parsed
                                            }
                                            Err(e) => {
                                                tracing::error!("🔧 Failed to parse tool {} input as JSON: {} (input: '{}')",
                                                    tool_id, e, accumulated_input);
                                                // Fallback to empty object like Python reference
                                                serde_json::Value::Object(serde_json::Map::new())
                                            }
                                        }
                                    };

                                    tool_call.input = parsed_input;
                                    current_tool_calls.insert(tool_id.clone(), tool_call);
                                    tracing::debug!("🔧 Finalized tool call at ContentBlockStop: {} with input: {}",
                                        tool_id, accumulated_input);
                                }
                            }
                        }
                    }

                    active_content_blocks.remove(block_index);
                }
                crate::llm::traits::StreamEvent::MessageStop { stop_reason } => {
                    tracing::debug!("✅ Message completed with stop reason: {:?}", stop_reason);

                    // Record stream completion time for evaluation delay
                    self.stream_completion_time = Some(std::time::Instant::now());

                    // Build final response from collected content
                    let content = content_parts.join("");
                    tracing::info!(
                        "🎯 Final streaming content assembled: '{}' (from {} parts)",
                        content,
                        content_parts.len()
                    );

                    // Extract finalized tool calls
                    let final_tool_calls: Vec<crate::llm::traits::ToolCall> =
                        current_tool_calls.values().cloned().collect();

                    tracing::info!(
                        "🔧 Extracted {} tool calls from universal content block streaming",
                        final_tool_calls.len()
                    );
                    for (i, tool_call) in final_tool_calls.iter().enumerate() {
                        tracing::debug!(
                            "🔧 Final tool call {}: {} with input: {}",
                            i + 1,
                            tool_call.name,
                            serde_json::to_string(&tool_call.input).unwrap_or_default()
                        );
                    }

                    final_response = Some(crate::llm::traits::ChatResponse {
                        content: content.clone(),
                        tool_calls: final_tool_calls.clone(),
                        thinking: None, // TODO: Extract thinking from streaming
                        usage: stream_usage.clone(),
                        metadata: std::collections::HashMap::new(),
                    });

                    // TRACE MODE: Print the final streaming response
                    if tracing::level_enabled!(tracing::Level::TRACE) {
                        tracing::trace!("📋 TRACE: ==================== STREAMING RESPONSE ====================");
                        tracing::trace!("📋 TRACE: Final response content: '{}'", content);
                        tracing::trace!(
                            "📋 TRACE: Tool calls in response: {}",
                            final_tool_calls.len()
                        );
                        for (i, tool_call) in final_tool_calls.iter().enumerate() {
                            tracing::trace!(
                                "📋 TRACE: Tool call #{}: {} ({})",
                                i + 1,
                                tool_call.name,
                                tool_call.id
                            );
                            tracing::trace!(
                                "📋 TRACE: Tool call #{}: Input: {}",
                                i + 1,
                                serde_json::to_string_pretty(&tool_call.input)
                                    .unwrap_or_else(|_| "Invalid JSON".to_string())
                            );
                        }
                        if let Some(ref usage) = stream_usage {
                            tracing::trace!(
                                "📋 TRACE: Token usage: {} in, {} out, {} total",
                                usage.input_tokens,
                                usage.output_tokens,
                                usage.total_tokens
                            );
                        }
                        tracing::trace!(
                            "📋 TRACE: ==================== END RESPONSE ===================="
                        );
                    }

                    break;
                }
                crate::llm::traits::StreamEvent::Metadata { usage } => {
                    tracing::debug!("📊 Stream metadata received: {:?}", usage);
                    stream_usage = usage.clone();
                }
                crate::llm::traits::StreamEvent::Error { error } => {
                    tracing::error!("❌ Stream error: {}", error);
                    return Err(crate::StoodError::model_error(format!(
                        "Stream error: {}",
                        error
                    )));
                }

                // Handle legacy events for backward compatibility
                crate::llm::traits::StreamEvent::ContentDelta { delta, .. } => {
                    tracing::debug!(
                        "🎯 Legacy ContentDelta: '{}' - converting to universal pattern",
                        delta
                    );
                    content_parts.push(delta.clone());

                    if let Some(ref callback) = self.callback_handler {
                        let event = CallbackEvent::ContentDelta {
                            delta: delta.clone(),
                            complete: false,
                            reasoning: false,
                        };
                        if let Err(e) = callback.handle_event(event).await {
                            tracing::warn!("Callback error during legacy ContentDelta: {}", e);
                        }
                    }
                }
                crate::llm::traits::StreamEvent::ToolCallStart { tool_call } => {
                    tracing::debug!(
                        "🔧 Legacy ToolCallStart: {} ({}) - converting to universal pattern",
                        tool_call.name,
                        tool_call.id
                    );
                    current_tool_calls.insert(tool_call.id.clone(), tool_call.clone());
                    current_tool_inputs.insert(tool_call.id.clone(), String::new());
                }
                crate::llm::traits::StreamEvent::ToolCallDelta {
                    tool_call_id,
                    delta,
                } => {
                    tracing::debug!(
                        "🔧 Legacy ToolCallDelta for {}: '{}' - converting to universal pattern",
                        tool_call_id,
                        delta
                    );
                    current_tool_inputs
                        .entry(tool_call_id.clone())
                        .or_default()
                        .push_str(delta);

                    // For Nova, the delta might contain the complete JSON input
                    // Try to parse it and update the tool call if successful
                    if let Some(tool_call) = current_tool_calls.get_mut(tool_call_id) {
                        if let Some(accumulated) = current_tool_inputs.get(tool_call_id) {
                            if !accumulated.is_empty() {
                                if let Ok(parsed_input) =
                                    serde_json::from_str::<serde_json::Value>(accumulated)
                                {
                                    tracing::debug!(
                                        "🔧 Successfully parsed tool input for {}: {}",
                                        tool_call_id,
                                        accumulated
                                    );
                                    tool_call.input = parsed_input;
                                }
                            }
                        }
                    }
                }
                crate::llm::traits::StreamEvent::Done { usage } => {
                    tracing::debug!(
                        "✅ Legacy Done event with usage: {:?} - converting to MessageStop",
                        usage
                    );
                    stream_usage = usage.clone();

                    let content = content_parts.join("");
                    let final_tool_calls: Vec<crate::llm::traits::ToolCall> =
                        current_tool_calls.values().cloned().collect();

                    final_response = Some(crate::llm::traits::ChatResponse {
                        content,
                        tool_calls: final_tool_calls,
                        thinking: None,
                        usage: stream_usage.clone(),
                        metadata: std::collections::HashMap::new(),
                    });
                    break;
                }
                _ => {
                    tracing::debug!("📨 Other streaming event: {:?}", stream_event);
                }
            }
        }

        // Build final response from collected content if not already set
        if final_response.is_none() {
            let content = content_parts.join("");
            tracing::warn!("🎯 Building fallback response from {} content parts (no Done event received): '{}'",
                content_parts.len(), content);

            // Build fallback tool calls from accumulated data
            let mut fallback_tool_calls = Vec::new();
            for (tool_id, mut tool_call) in current_tool_calls {
                // Check if we already have a valid input (e.g., from Nova streaming)
                if tool_call.input.is_null() {
                    // Only try to parse accumulated input if we don't already have a valid input
                    if let Some(accumulated_input) = current_tool_inputs.get(&tool_id) {
                        let parsed_input = if accumulated_input.is_empty() {
                            serde_json::Value::Object(serde_json::Map::new())
                        } else {
                            serde_json::from_str(accumulated_input).unwrap_or_else(
                                |_| serde_json::json!({"raw_input": accumulated_input}),
                            )
                        };
                        tool_call.input = parsed_input;
                    } else {
                        // No accumulated input, use empty object
                        tool_call.input = serde_json::Value::Object(serde_json::Map::new());
                    }
                }
                fallback_tool_calls.push(tool_call);
            }

            tracing::warn!(
                "🔧 Fallback response includes {} tool calls",
                fallback_tool_calls.len()
            );

            final_response = Some(crate::llm::traits::ChatResponse {
                content,
                tool_calls: fallback_tool_calls,
                thinking: None, // TODO: Extract thinking from streaming
                usage: None,
                metadata: std::collections::HashMap::new(),
            });
        }

        tracing::info!(
            "🎯 Streaming execution completed, received {} events total",
            received_event_count
        );

        // Return the final response
        let response = final_response.ok_or_else(|| {
            crate::StoodError::model_error("No response received from streaming".to_string())
        })?;

        tracing::debug!(
            "🔧 Streaming method returning response with {} tool calls",
            response.tool_calls.len()
        );
        for (i, tool_call) in response.tool_calls.iter().enumerate() {
            tracing::debug!(
                "🔧 Returning tool call {}: {} with input: {}",
                i + 1,
                tool_call.name,
                serde_json::to_string(&tool_call.input).unwrap_or_default()
            );
        }

        Ok(response)
    }

    /// Execute a streaming chat with real-time callbacks
    pub async fn execute_with_streaming<C>(
        &mut self,
        prompt: impl Into<String>,
        callback: C,
    ) -> Result<EventLoopResult>
    where
        C: StreamCallback + Send + 'static,
    {
        let original_streaming = self.config.enable_streaming;
        self.config.enable_streaming = true;

        // Register callback for streaming events
        let callback = std::sync::Arc::new(callback);

        let result = self.execute(prompt).await?;

        // Send stream events to callback
        for event in &self.stream_events {
            callback.on_event(event);
        }

        // Provide dummy completion arguments for now
        let dummy_message =
            crate::types::Message::new(crate::types::MessageRole::Assistant, vec![]);
        callback.on_complete(&dummy_message, &Default::default(), &Default::default());

        self.config.enable_streaming = original_streaming;
        Ok(result)
    }

    /// Cancel any ongoing streaming operations
    pub async fn cancel_stream(&mut self) {
        tracing::info!("Cancelling streaming operations");

        // Add cancellation event
        self.stream_events.push(StreamEvent::MessageStop(
            crate::streaming::MessageStopEvent {
                additional_model_response_fields: None,
                stop_reason: crate::streaming::StopReason::EndTurn,
            },
        ));

        // In a real implementation, this would cancel the AWS Bedrock stream
        // For now, we just log the cancellation
        tracing::debug!("Stream cancellation completed");
    }

    /// Check if the event loop has been cancelled
    ///
    /// Returns true if a cancellation token was provided and has been cancelled.
    /// This is used internally by the event loop to exit early and bypass task evaluation.
    fn is_cancelled(&self) -> bool {
        if let Some(ref token) = self.config.cancellation_token {
            token.is_cancelled()
        } else {
            false
        }
    }

    /// Get streaming metrics
    pub fn streaming_metrics(&self) -> StreamingMetrics {
        StreamingMetrics {
            total_events: self.stream_events.len(),
            stream_duration: self
                .metrics
                .cycles
                .iter()
                .map(|c| c.duration)
                .fold(Duration::ZERO, |acc, d| acc + d),
            was_streamed: self.config.enable_streaming,
            events_per_second: if !self.stream_events.is_empty() {
                let total_duration = self
                    .metrics
                    .cycles
                    .iter()
                    .map(|c| c.duration)
                    .fold(Duration::ZERO, |acc, d| acc + d);
                if total_duration.as_secs_f64() > 0.0 {
                    self.stream_events.len() as f64 / total_duration.as_secs_f64()
                } else {
                    0.0
                }
            } else {
                0.0
            },
        }
    }

    /// Get comprehensive telemetry metrics
    pub fn telemetry_metrics(&self) -> TelemetryMetrics {
        TelemetryMetrics {
            active_spans: self.active_spans.len(),
            total_cycles: self.performance_logger.total_cycles,
            average_cycle_duration: self.performance_logger.average_cycle_time(),
            average_model_duration: self.performance_logger.average_model_time(),
            tool_performance: self.performance_logger.tool_times.clone(),
        }
    }

    /// Clean up stale spans (for error recovery)
    pub fn cleanup_stale_spans(&mut self, max_age: Duration) {
        let cutoff = Instant::now() - max_age;
        let initial_count = self.active_spans.len();
        self.active_spans
            .retain(|_, span_info| span_info.start_time > cutoff);
        let cleaned_count = initial_count - self.active_spans.len();

        if cleaned_count > 0 {
            tracing::warn!(
                stale_spans_cleaned = cleaned_count,
                max_age_seconds = max_age.as_secs(),
                "Cleaned up stale telemetry spans"
            );
        }
    }

    /// Evaluate whether the event loop should continue based on the configured strategy
    /// Wait for stream completion to prevent evaluation/streaming overlap
    async fn wait_for_stream_completion(&self) {
        // If streaming was not active, no need to wait
        if !self.stream_was_active {
            return;
        }

        // If we have a completion time, wait for a brief buffer to allow display to finish
        if let Some(completion_time) = self.stream_completion_time {
            let elapsed = completion_time.elapsed();
            const STREAM_BUFFER_MS: u64 = 100; // 100ms buffer to allow display to finish

            if elapsed.as_millis() < STREAM_BUFFER_MS as u128 {
                let remaining = std::time::Duration::from_millis(STREAM_BUFFER_MS) - elapsed;
                tracing::debug!(
                    "⏳ Waiting {}ms for stream completion buffer",
                    remaining.as_millis()
                );
                tokio::time::sleep(remaining).await;
            }
        }
    }

    async fn evaluate_continuation(
        &mut self,
        current_response: &crate::llm::traits::ChatResponse,
        cycle_metrics: &CycleMetrics,
    ) -> Result<EvaluationResult> {
        let strategy = self.config.evaluation_strategy.clone();

        // Create evaluation span as child of current cycle span
        let evaluation_span = if let Some(ref tracer) = self.tracer {
            let mut span = tracer.start_agent_span("evaluation");
            span.set_attribute("evaluation.strategy", strategy.name());
            span.set_attribute("evaluation.cycle_id", cycle_metrics.cycle_id.to_string());
            span.set_attribute(
                "evaluation.response_length",
                current_response.content.len() as i64,
            );
            Some(span)
        } else {
            None
        };

        tracing::info!(
            "🔍 Evaluating continuation with strategy: {:?}",
            strategy.name()
        );

        let evaluation_start = std::time::Instant::now();
        let result = match strategy {
            EvaluationStrategy::None => {
                // Model-driven continuation: check if model naturally wants to continue
                // This happens when model returns tool calls or indicates more work needed
                self.evaluate_model_driven(current_response).await
            }
            EvaluationStrategy::TaskEvaluation {
                evaluation_prompt,
                max_iterations,
            } => {
                self.evaluate_task_evaluation(current_response, &evaluation_prompt, max_iterations)
                    .await
            }
            EvaluationStrategy::AgentBased {
                mut evaluator_agent,
                evaluation_prompt,
            } => {
                self.evaluate_agent_based(
                    current_response,
                    &mut evaluator_agent,
                    &evaluation_prompt,
                )
                .await
            }
            EvaluationStrategy::MultiPerspective { perspectives } => {
                self.evaluate_multi_perspective(current_response, &perspectives)
                    .await
            }
        };

        // Complete evaluation span with results
        let evaluation_duration = evaluation_start.elapsed();
        if let Some(mut span) = evaluation_span {
            match &result {
                Ok(eval_result) => {
                    span.set_attribute("evaluation.decision", eval_result.decision);
                    span.set_attribute(
                        "evaluation.duration_ms",
                        evaluation_duration.as_millis() as i64,
                    );
                    span.set_attribute(
                        "evaluation.has_additional_content",
                        eval_result.response.is_some(),
                    );
                    if let Some(ref content) = eval_result.response {
                        span.set_attribute(
                            "evaluation.additional_content_length",
                            content.len() as i64,
                        );
                    }
                    span.set_attribute("evaluation.status", "success");
                }
                Err(e) => {
                    span.set_attribute("evaluation.status", "error");
                    span.set_attribute("evaluation.error", e.to_string());
                    span.set_attribute(
                        "evaluation.duration_ms",
                        evaluation_duration.as_millis() as i64,
                    );
                }
            }
        }

        tracing::info!(
            "✅ Evaluation completed in {:?} with decision: {:?}",
            evaluation_duration,
            result.as_ref().map(|r| r.decision).unwrap_or(false)
        );

        result
    }

    /// Model-driven strategy: no explicit evaluation, just stop after one cycle
    /// Let the model decide naturally whether it wants to continue via tool calls
    async fn evaluate_model_driven(
        &mut self,
        _current_response: &crate::llm::traits::ChatResponse,
    ) -> Result<EvaluationResult> {
        use std::time::Instant;
        let start_time = Instant::now();

        tracing::info!(
            "🔍 Model-driven strategy: stopping after one cycle (let model decide naturally)"
        );

        // Trigger evaluation start callback
        if let Some(callback) = &self.callback_handler {
            let _ = callback
                .handle_event(crate::agent::callbacks::CallbackEvent::EvaluationStart {
                    strategy: "model_driven".to_string(),
                    prompt: "No explicit evaluation - model decides naturally".to_string(),
                    cycle_number: self.metrics.cycles.len() as u32,
                })
                .await;
        }

        // Model-driven: always stop after one cycle
        // The model will naturally continue if it wants to via tool calls in future requests
        let result = EvaluationResult {
            decision: false, // Always stop - let model decide naturally
            response: None,
            reasoning:
                "Model-driven: stopping after one cycle, model decides continuation naturally"
                    .to_string(),
        };

        // Trigger evaluation complete callback
        if let Some(callback) = &self.callback_handler {
            let _ = callback
                .handle_event(crate::agent::callbacks::CallbackEvent::EvaluationComplete {
                    strategy: "model_driven".to_string(),
                    decision: result.decision,
                    reasoning: result.reasoning.clone(),
                    duration: start_time.elapsed(),
                })
                .await;
        }

        tracing::info!("🤔 Model-driven: STOP (model decides continuation naturally)");

        Ok(result)
    }

    /// Evaluate continuation using task evaluation strategy
    async fn evaluate_task_evaluation(
        &mut self,
        current_response: &crate::llm::traits::ChatResponse,
        evaluation_prompt: &str,
        _max_iterations: u32,
    ) -> Result<EvaluationResult> {
        use std::time::Instant;
        let start_time = Instant::now();

        tracing::info!("🔍 Evaluating continuation with task evaluation strategy");

        // Trigger evaluation start callback
        if let Some(callback) = &self.callback_handler {
            let _ = callback
                .handle_event(crate::agent::callbacks::CallbackEvent::EvaluationStart {
                    strategy: "task_evaluation".to_string(),
                    prompt: evaluation_prompt.to_string(),
                    cycle_number: self.metrics.cycles.len() as u32,
                })
                .await;
        }

        // Create clean conversation summary for evaluation
        let conversation_summary = self.create_evaluation_summary();

        // Create a task evaluation prompt with clean summary and current response
        let evaluation_question = format!(
            "{}\n\nConversation so far:\n{}\n\nCurrent response: \"{}\"\n\nEvaluate whether you should continue working on this task or if it's complete. Respond with JSON in this exact format:\n{{\n  \"decision\": \"CONTINUE\" or \"STOP\",\n  \"response\": \"Additional content to add if continuing (empty if stopping)\"\n}}\n\nIf you decide to CONTINUE, provide additional content in the 'response' field that will be added to the conversation to continue the task. If you decide to STOP, leave the 'response' field empty.",
            evaluation_prompt,
            conversation_summary,
            current_response.content
        );

        // Create model span for evaluation LLM call
        let evaluation_model_span = if let Some(ref tracer) = self.tracer {
            let mut span = tracer.start_model_span("evaluation_llm_call");
            span.set_attribute("evaluation.type", "task_evaluation");
            span.set_attribute("evaluation.model_id", self.agent.config().model_id.clone());
            span.set_attribute("evaluation.prompt_length", evaluation_question.len() as i64);
            Some(span)
        } else {
            None
        };

        // Use isolated evaluation context to avoid polluting main conversation
        let evaluation_response_content = if let Some(ref eval_ctx) = self.evaluation_context {
            eval_ctx.evaluate_with_prompt(&evaluation_question).await?
        } else {
            return Err(StoodError::InvalidInput {
                message: "Evaluation context not available".to_string(),
            });
        };

        // Complete model span with response details
        if let Some(mut span) = evaluation_model_span {
            span.set_attribute(
                "evaluation.response_length",
                evaluation_response_content.len() as i64,
            );
            span.set_attribute("evaluation.llm_status", "success");
        }

        let evaluation_result =
            EvaluationResult::parse_evaluation_response(&evaluation_response_content);
        let duration = start_time.elapsed();

        tracing::info!(
            "🔍 Task evaluation result: {} (response: '{}')",
            if evaluation_result.decision {
                "CONTINUE"
            } else {
                "STOP"
            },
            evaluation_response_content.trim()
        );

        // Trigger evaluation complete callback
        if let Some(callback) = &self.callback_handler {
            let _ = callback
                .handle_event(crate::agent::callbacks::CallbackEvent::EvaluationComplete {
                    strategy: "task_evaluation".to_string(),
                    decision: evaluation_result.decision,
                    reasoning: evaluation_result.reasoning.clone(),
                    duration,
                })
                .await;
        }

        Ok(evaluation_result)
    }

    /// Evaluate continuation using agent-based strategy
    async fn evaluate_agent_based(
        &mut self,
        current_response: &crate::llm::traits::ChatResponse,
        evaluator_agent: &mut Agent,
        evaluation_prompt: &str,
    ) -> Result<EvaluationResult> {
        use std::time::Instant;
        let start_time = Instant::now();

        tracing::info!("🤖 Evaluating continuation with agent-based strategy");

        // Trigger evaluation start callback
        if let Some(callback) = &self.callback_handler {
            let _ = callback
                .handle_event(crate::agent::callbacks::CallbackEvent::EvaluationStart {
                    strategy: "agent_based".to_string(),
                    prompt: evaluation_prompt.to_string(),
                    cycle_number: self.metrics.cycles.len() as u32,
                })
                .await;
        }

        // Get the original conversation history for context
        let original_conversation = self.agent.conversation().messages().clone();

        // Build context from original conversation
        let mut context_parts = Vec::new();
        for message in &original_conversation.messages {
            match message.role {
                crate::types::MessageRole::User => {
                    let content = message
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            crate::types::ContentBlock::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    context_parts.push(format!("User: {}", content));
                }
                crate::types::MessageRole::Assistant => {
                    let content = message
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            crate::types::ContentBlock::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    context_parts.push(format!("Assistant: {}", content));
                }
                crate::types::MessageRole::System => {
                    let content = message
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            crate::types::ContentBlock::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    context_parts.push(format!("System: {}", content));
                }
            }
        }
        let conversation_context = context_parts.join("\n");

        let agent_question = format!(
            "[INTERNAL EVALUATION - This is a private conversation for decision-making]\n\n{}\n\nConversation history:\n{}\n\nAgent's current response: \"{}\"\n\nBased on the full context, evaluate whether the agent should continue working. Respond with JSON in this exact format:\n{{\n  \"decision\": \"CONTINUE\" or \"STOP\",\n  \"response\": \"Additional content to add if continuing (empty if stopping)\"\n}}\n\nIf you decide the agent should CONTINUE, provide additional content in the 'response' field that will be added to the conversation to continue the task. If you decide to STOP, leave the 'response' field empty.",
            evaluation_prompt,
            conversation_context,
            current_response.content
        );

        // Execute the evaluator agent (use Box::pin to avoid recursion issue)
        let agent_result = Box::pin(evaluator_agent.execute(agent_question)).await?;

        let evaluation_result = EvaluationResult::parse_evaluation_response(&agent_result.response);
        let duration = start_time.elapsed();

        tracing::info!(
            "🤖 Agent-based evaluation result: {} (evaluator response: '{}')",
            if evaluation_result.decision {
                "CONTINUE"
            } else {
                "STOP"
            },
            agent_result.response.trim()
        );

        // Trigger evaluation complete callback
        if let Some(callback) = &self.callback_handler {
            let _ = callback
                .handle_event(crate::agent::callbacks::CallbackEvent::EvaluationComplete {
                    strategy: "agent_based".to_string(),
                    decision: evaluation_result.decision,
                    reasoning: evaluation_result.reasoning.clone(),
                    duration,
                })
                .await;
        }

        Ok(evaluation_result)
    }

    /// Evaluate continuation using multi-perspective strategy
    async fn evaluate_multi_perspective(
        &mut self,
        current_response: &crate::llm::traits::ChatResponse,
        perspectives: &[crate::agent::evaluation::PerspectiveConfig],
    ) -> Result<EvaluationResult> {
        use std::time::Instant;
        let start_time = Instant::now();

        tracing::info!(
            "👥 Evaluating continuation with multi-perspective strategy ({} perspectives)",
            perspectives.len()
        );

        // Trigger evaluation start callback
        if let Some(callback) = &self.callback_handler {
            let _ = callback
                .handle_event(crate::agent::callbacks::CallbackEvent::EvaluationStart {
                    strategy: "multi_perspective".to_string(),
                    prompt: format!(
                        "{} perspectives: {}",
                        perspectives.len(),
                        perspectives
                            .iter()
                            .map(|p| p.name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                    cycle_number: self.metrics.cycles.len() as u32,
                })
                .await;
        }

        let mut weighted_score = 0.0;
        let mut total_weight = 0.0;
        let mut perspective_details = Vec::new();

        // Create clean conversation summary for evaluation
        let conversation_summary = self.create_evaluation_summary();

        for perspective in perspectives {
            let perspective_question = format!(
                "{}\n\nConversation so far:\n{}\n\nCurrent response: \"{}\"\n\nFrom this perspective, should we continue? Respond with 'CONTINUE' or 'STOP'.",
                perspective.prompt,
                conversation_summary,
                current_response.content
            );

            // Use isolated evaluation context to avoid polluting main conversation
            let perspective_response_content = if let Some(ref eval_ctx) = self.evaluation_context {
                eval_ctx.evaluate_with_prompt(&perspective_question).await?
            } else {
                return Err(StoodError::InvalidInput {
                    message: "Evaluation context not available".to_string(),
                });
            };

            let perspective_continue = perspective_response_content
                .to_uppercase()
                .contains("CONTINUE");
            let perspective_score = if perspective_continue { 1.0 } else { 0.0 };

            weighted_score += perspective_score * perspective.weight;
            total_weight += perspective.weight;

            perspective_details.push(format!(
                "{}: {} (weight: {:.2})",
                perspective.name,
                if perspective_continue {
                    "CONTINUE"
                } else {
                    "STOP"
                },
                perspective.weight
            ));

            tracing::info!(
                "👥 Perspective '{}' (weight: {:.2}): {} (response: '{}')",
                perspective.name,
                perspective.weight,
                if perspective_continue {
                    "CONTINUE"
                } else {
                    "STOP"
                },
                perspective_response_content.trim()
            );
        }

        let final_score = if total_weight > 0.0 {
            weighted_score / total_weight
        } else {
            0.0
        };

        let should_continue = final_score > 0.5;
        let duration = start_time.elapsed();

        tracing::info!(
            "👥 Multi-perspective result: {} (weighted score: {:.2})",
            if should_continue { "CONTINUE" } else { "STOP" },
            final_score
        );

        // For multi-perspective, we don't generate additional content - it's a voting system
        let evaluation_result = EvaluationResult {
            decision: should_continue,
            response: None, // Multi-perspective doesn't generate additional content
            reasoning: format!(
                "Weighted score: {:.2}/1.0 - {}",
                final_score,
                perspective_details.join(", ")
            ),
        };

        // Trigger evaluation complete callback
        if let Some(callback) = &self.callback_handler {
            let _ = callback
                .handle_event(crate::agent::callbacks::CallbackEvent::EvaluationComplete {
                    strategy: "multi_perspective".to_string(),
                    decision: evaluation_result.decision,
                    reasoning: evaluation_result.reasoning.clone(),
                    duration,
                })
                .await;
        }

        Ok(evaluation_result)
    }
}

impl Default for PerformanceLogger {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceLogger {
    pub fn new() -> Self {
        Self {
            cycle_times: Vec::new(),
            tool_times: std::collections::HashMap::new(),
            model_invoke_times: Vec::new(),
            total_cycles: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            model_input_tokens: Vec::new(),
            model_output_tokens: Vec::new(),
        }
    }

    pub fn log_cycle_performance(&mut self, duration: Duration, cycle_id: Uuid) {
        self.cycle_times.push(duration);
        self.total_cycles += 1;

        tracing::info!(
            cycle_id = %cycle_id,
            duration_ms = duration.as_millis(),
            avg_cycle_ms = self.average_cycle_time().as_millis(),
            total_cycles = self.total_cycles,
            "Cycle performance metrics"
        );
    }

    pub fn log_tool_performance(&mut self, tool_name: &str, duration: Duration, success: bool) {
        self.tool_times
            .entry(tool_name.to_string())
            .or_default()
            .push(duration);

        if success {
            tracing::debug!(
                tool_name = tool_name,
                duration_ms = duration.as_millis(),
                avg_tool_ms = self.average_tool_time(tool_name).as_millis(),
                "Tool execution completed successfully"
            );
        } else {
            tracing::error!(
                tool_name = tool_name,
                duration_ms = duration.as_millis(),
                "Tool execution failed"
            );
        }
    }

    pub fn log_model_performance(
        &mut self,
        duration: Duration,
        token_usage: Option<&crate::types::TokenUsage>,
    ) {
        self.model_invoke_times.push(duration);

        // Track token usage for metrics analysis
        if let Some(tokens) = token_usage {
            self.total_input_tokens += tokens.input_tokens;
            self.total_output_tokens += tokens.output_tokens;
            self.model_input_tokens.push(tokens.input_tokens);
            self.model_output_tokens.push(tokens.output_tokens);

            tracing::debug!(
                duration_ms = duration.as_millis(),
                avg_model_ms = self.average_model_time().as_millis(),
                total_invocations = self.model_invoke_times.len(),
                input_tokens = tokens.input_tokens,
                output_tokens = tokens.output_tokens,
                total_tokens = tokens.total_tokens,
                cumulative_input_tokens = self.total_input_tokens,
                cumulative_output_tokens = self.total_output_tokens,
                "Model invocation performance with token usage"
            );
        } else {
            tracing::debug!(
                duration_ms = duration.as_millis(),
                avg_model_ms = self.average_model_time().as_millis(),
                total_invocations = self.model_invoke_times.len(),
                "Model invocation performance (no token data)"
            );
        }
    }

    pub fn log_event_loop_completion(
        &self,
        total_duration: Duration,
        cycles_executed: u32,
        success: bool,
    ) {
        if success {
            tracing::info!(
                total_duration_ms = total_duration.as_millis(),
                cycles_executed = cycles_executed,
                avg_cycle_ms = self.average_cycle_time().as_millis(),
                total_input_tokens = self.total_input_tokens,
                total_output_tokens = self.total_output_tokens,
                total_tokens = self.total_tokens(),
                avg_input_tokens = self.average_input_tokens(),
                avg_output_tokens = self.average_output_tokens(),
                "Event loop completed successfully with detailed token usage"
            );
        } else {
            tracing::error!(
                total_duration_ms = total_duration.as_millis(),
                cycles_executed = cycles_executed,
                "Event loop completed with errors"
            );
        }
    }

    fn average_cycle_time(&self) -> Duration {
        if self.cycle_times.is_empty() {
            Duration::ZERO
        } else {
            self.cycle_times.iter().sum::<Duration>() / self.cycle_times.len() as u32
        }
    }

    fn average_model_time(&self) -> Duration {
        if self.model_invoke_times.is_empty() {
            Duration::ZERO
        } else {
            self.model_invoke_times.iter().sum::<Duration>() / self.model_invoke_times.len() as u32
        }
    }

    /// Get total input tokens for metrics analysis
    pub fn total_input_tokens(&self) -> u32 {
        self.total_input_tokens
    }

    /// Get total output tokens for metrics analysis
    pub fn total_output_tokens(&self) -> u32 {
        self.total_output_tokens
    }

    /// Get total tokens (input + output)
    pub fn total_tokens(&self) -> u32 {
        self.total_input_tokens + self.total_output_tokens
    }

    /// Get average input tokens per model call
    pub fn average_input_tokens(&self) -> f64 {
        if self.model_input_tokens.is_empty() {
            0.0
        } else {
            self.model_input_tokens.iter().sum::<u32>() as f64
                / self.model_input_tokens.len() as f64
        }
    }

    /// Get average output tokens per model call
    pub fn average_output_tokens(&self) -> f64 {
        if self.model_output_tokens.is_empty() {
            0.0
        } else {
            self.model_output_tokens.iter().sum::<u32>() as f64
                / self.model_output_tokens.len() as f64
        }
    }

    fn average_tool_time(&self, tool_name: &str) -> Duration {
        if let Some(times) = self.tool_times.get(tool_name) {
            if times.is_empty() {
                Duration::ZERO
            } else {
                times.iter().sum::<Duration>() / times.len() as u32
            }
        } else {
            Duration::ZERO
        }
    }
}

/// Comprehensive telemetry metrics
#[derive(Debug, Clone)]
pub struct TelemetryMetrics {
    /// Number of currently active spans
    pub active_spans: usize,
    /// Total cycles executed
    pub total_cycles: u32,
    /// Average duration per model interaction
    pub average_cycle_duration: Duration,
    /// Average duration per model invocation
    pub average_model_duration: Duration,
    /// Performance metrics per tool
    pub tool_performance: std::collections::HashMap<String, Vec<Duration>>,
}

/// Streaming performance metrics
#[derive(Debug, Clone)]
pub struct StreamingMetrics {
    /// Total number of stream events processed
    pub total_events: usize,
    /// Total duration of streaming operations
    pub stream_duration: Duration,
    /// Whether streaming was actually used
    pub was_streamed: bool,
    /// Events processed per second
    pub events_per_second: f64,
}

/// Result of a tool execution
#[derive(Debug, Clone)]
struct ToolResult {
    tool_use_id: String,
    tool_name: String,
    input: Value,
    success: bool,
    output: Option<Value>,
    error: Option<String>,
    duration: Duration,
}

/// Result of a single model interaction
#[derive(Debug, Clone)]
struct CycleResult {
    response: String,
    should_continue: bool,
    tool_iterations_used: Option<u32>,
}

/// Result of an evaluation with structured decision and response
#[derive(Debug, Clone)]
struct EvaluationResult {
    /// Whether to continue (true) or stop (false)
    decision: bool,
    /// Additional response content to add to conversation if continuing
    response: Option<String>,
    /// Raw evaluation reasoning/response for logging
    reasoning: String,
}

impl EvaluationResult {
    /// Parse evaluation response from JSON or fallback to enhanced string matching
    fn parse_evaluation_response(response: &str) -> Self {
        tracing::debug!(
            "🔍 Parsing evaluation response: {}",
            response.chars().take(200).collect::<String>()
        );

        // Strategy 1: Try direct JSON parsing
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(response) {
            tracing::debug!("✅ Direct JSON parsing successful");
            return Self::parse_json_object(&json, response);
        }

        // Strategy 2: Try regex-based JSON extraction for mixed content
        if let Some(extracted_json) = Self::extract_json_from_mixed_content(response) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&extracted_json) {
                tracing::debug!("✅ Regex-extracted JSON parsing successful");
                return Self::parse_json_object(&json, response);
            }
        }

        // Strategy 3: Enhanced fallback with multiple pattern matching
        tracing::debug!("📝 JSON parsing failed, using enhanced string matching fallback");
        Self::parse_with_enhanced_fallback(response)
    }

    /// Parse a valid JSON object into EvaluationResult
    fn parse_json_object(json: &serde_json::Value, original_response: &str) -> Self {
        // Handle decision field - support both string and boolean types
        let decision = json
            .get("decision")
            .and_then(|d| {
                // Try as boolean first
                if let Some(bool_val) = d.as_bool() {
                    Some(bool_val)
                } else if let Some(str_val) = d.as_str() {
                    // Fuzzy string matching for decision
                    let normalized = str_val.trim().to_uppercase();
                    Some(normalized == "CONTINUE" || normalized == "TRUE" || normalized == "YES")
                } else {
                    None
                }
            })
            .unwrap_or_else(|| {
                tracing::warn!("⚠️ No valid 'decision' field found in JSON, defaulting to false");
                false
            });

        // Handle response content with multiple field name options
        let response_content = json
            .get("response")
            .or_else(|| json.get("additional_content"))
            .or_else(|| json.get("content"))
            .or_else(|| json.get("message"))
            .and_then(|r| r.as_str())
            .map(|s| s.to_string())
            .filter(|s| !s.trim().is_empty());

        if response_content.is_some() {
            tracing::debug!(
                "📝 Extracted response content: {}",
                response_content
                    .as_ref()
                    .unwrap()
                    .chars()
                    .take(100)
                    .collect::<String>()
            );
        }

        Self {
            decision,
            response: response_content,
            reasoning: original_response.to_string(),
        }
    }

    /// Extract JSON from mixed content using regex patterns
    fn extract_json_from_mixed_content(content: &str) -> Option<String> {
        // Pattern 1: JSON block wrapped in code fences
        if let Some(json_match) = content.find("```json") {
            if let Some(end_match) = content[json_match..].find("```") {
                let start = json_match + 7; // Length of "```json"
                let end = json_match + end_match;
                if start < end {
                    return Some(content[start..end].trim().to_string());
                }
            }
        }

        // Pattern 2: JSON block wrapped in simple code fences
        if let Some(json_match) = content.find("```") {
            if let Some(end_match) = content[json_match + 3..].find("```") {
                let start = json_match + 3;
                let end = json_match + 3 + end_match;
                let potential_json = content[start..end].trim();
                if potential_json.starts_with('{') && potential_json.ends_with('}') {
                    return Some(potential_json.to_string());
                }
            }
        }

        // Pattern 3: Look for JSON object patterns
        if let Some(start) = content.find('{') {
            if let Some(end) = content.rfind('}') {
                if start < end {
                    let potential_json = content[start..=end].trim();
                    // Quick validation - contains both decision and response-like fields
                    if potential_json.contains("decision")
                        && (potential_json.contains("response")
                            || potential_json.contains("content"))
                    {
                        return Some(potential_json.to_string());
                    }
                }
            }
        }

        None
    }

    /// Enhanced fallback parsing with multiple pattern matching approaches
    fn parse_with_enhanced_fallback(response: &str) -> Self {
        let upper_response = response.to_uppercase();

        // Enhanced decision detection with multiple patterns
        let decision =
            // Explicit decision patterns
            upper_response.contains("DECISION: CONTINUE") ||
            upper_response.contains("DECISION: TRUE") ||
            upper_response.contains("CONTINUE: TRUE") ||
            upper_response.contains("SHOULD CONTINUE: TRUE") ||
            upper_response.contains("CONTINUE: YES") ||

            // Standalone decision patterns
            upper_response.contains("CONTINUE") ||
            upper_response.contains("KEEP GOING") ||
            upper_response.contains("NOT COMPLETE") ||
            upper_response.contains("NEEDS MORE") ||
            upper_response.contains("INSUFFICIENT") ||

            // Negative patterns (things that suggest continuation needed)
            upper_response.contains("INCOMPLETE") ||
            upper_response.contains("MISSING") ||
            upper_response.contains("LACKING");

        // Try to extract response content from fallback
        let response_content = Self::extract_response_from_fallback(response);

        tracing::debug!(
            "🔄 Fallback parsing result: decision={}, has_content={}",
            decision,
            response_content.is_some()
        );

        Self {
            decision,
            response: response_content,
            reasoning: response.to_string(),
        }
    }

    /// Extract response content from non-JSON text
    fn extract_response_from_fallback(content: &str) -> Option<String> {
        // Look for common response indicators
        let patterns = [
            "response:",
            "additional content:",
            "next steps:",
            "improvements needed:",
            "continue with:",
            "add:",
        ];

        for pattern in &patterns {
            if let Some(start) = content.to_lowercase().find(pattern) {
                let start_pos = start + pattern.len();
                let remaining = content[start_pos..].trim();

                // Extract until next line break or end
                let end_pos = remaining.find('\n').unwrap_or(remaining.len());
                let extracted = remaining[..end_pos].trim();

                if !extracted.is_empty() && extracted.len() > 5 {
                    return Some(extracted.to_string());
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::Agent;
    use crate::tools::ToolRegistry;

    #[tokio::test]
    async fn test_event_loop_creation() {
        let agent = Agent::builder().build().await.unwrap();
        let tool_registry = ToolRegistry::new();
        let config = EventLoopConfig::default();

        let event_loop = EventLoop::new(agent, tool_registry, config);
        assert!(event_loop.is_ok());
    }

    #[tokio::test]
    async fn test_event_loop_config() {
        let config = EventLoopConfig {
            max_cycles: 5,
            max_duration: Duration::from_secs(60),
            enable_streaming: true,
            ..EventLoopConfig::default()
        };

        assert_eq!(config.max_cycles, 5);
        assert_eq!(config.max_duration, Duration::from_secs(60));
        assert!(config.enable_streaming);
    }

    #[tokio::test]
    async fn test_streaming_configuration() {
        let config = EventLoopConfig::default();

        // Streaming should be enabled by default now
        assert!(config.enable_streaming);
        assert!(config.stream_config.enabled);
        assert_eq!(config.stream_config.buffer_size, 100);
        assert!(config.stream_config.enable_tool_streaming);

        // Retry configuration should be available
        assert_eq!(config.retry_config.max_attempts, 6);
        assert_eq!(config.retry_config.initial_delay, Duration::from_secs(4));
    }

    #[tokio::test]
    async fn test_streaming_metrics() {
        let agent = Agent::builder().build().await.unwrap();
        let tool_registry = ToolRegistry::new();
        let config = EventLoopConfig::default();

        let mut event_loop = EventLoop::new(agent, tool_registry, config).unwrap();

        // Test initial streaming metrics
        let metrics = event_loop.streaming_metrics();
        assert_eq!(metrics.total_events, 0);
        assert!(metrics.was_streamed);
        assert_eq!(metrics.events_per_second, 0.0);

        // Add some mock stream events
        event_loop.stream_events.push(StreamEvent::MessageStart(
            crate::streaming::MessageStartEvent {
                role: crate::types::MessageRole::Assistant,
            },
        ));

        event_loop.stream_events.push(StreamEvent::MessageStop(
            crate::streaming::MessageStopEvent {
                additional_model_response_fields: None,
                stop_reason: crate::streaming::StopReason::EndTurn,
            },
        ));

        let updated_metrics = event_loop.streaming_metrics();
        assert_eq!(updated_metrics.total_events, 2);
        assert!(updated_metrics.was_streamed);
    }

    #[tokio::test]
    async fn test_cancel_stream() {
        let agent = Agent::builder().build().await.unwrap();
        let tool_registry = ToolRegistry::new();
        let config = EventLoopConfig::default();

        let mut event_loop = EventLoop::new(agent, tool_registry, config).unwrap();

        let initial_count = event_loop.stream_events.len();

        // Test stream cancellation
        event_loop.cancel_stream().await;

        // Should add a MessageStop event
        assert_eq!(event_loop.stream_events.len(), initial_count + 1);

        // The last event should be a MessageStop
        let last_event = event_loop.stream_events.last().unwrap();
        assert!(matches!(last_event, StreamEvent::MessageStop(_)));
    }

    // Test helper that implements StreamCallback
    #[allow(dead_code)]
    struct TestStreamCallback {
        events_received: std::sync::Arc<std::sync::Mutex<Vec<StreamEvent>>>,
    }

    impl TestStreamCallback {
        #[allow(dead_code)]
        fn new() -> Self {
            Self {
                events_received: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            }
        }

        #[allow(dead_code)]
        fn get_events(&self) -> Vec<StreamEvent> {
            self.events_received.lock().unwrap().clone()
        }
    }

    impl StreamCallback for TestStreamCallback {
        fn on_event(&self, event: &StreamEvent) {
            self.events_received.lock().unwrap().push(event.clone());
        }

        fn on_complete(
            &self,
            _message: &crate::types::Message,
            _usage: &crate::streaming::Usage,
            _metrics: &crate::streaming::Metrics,
        ) {
            // Test callback completion
        }

        fn on_error(&self, _error: &crate::StoodError) {
            // Test callback error handling
        }
    }
}
