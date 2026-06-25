//! LM Studio provider implementation.
//!
//! This provider connects to a local LM Studio instance via HTTP API
//! and handles OpenAI-compatible request/response formatting.

use crate::llm::providers::retry::{retry_llm_operation, BoxFuture, RetryConfig};
use crate::llm::traits::{
    ChatConfig, ChatResponse, HealthStatus, LlmError, LlmProvider, ProviderCapabilities,
    ProviderType, StreamEvent, Tool,
};
use crate::types::{ContentBlock, MessageRole, Messages};
use async_trait::async_trait;
use futures::Stream;
use serde_json::Value;
use std::time::Instant;

/// Tool state management for LM Studio streaming (following Claude's pattern)
#[derive(Debug)]
struct LMStudioToolState {
    /// Currently active tool calls being assembled (supports multiple parallel tools)
    active_tool_calls: std::collections::HashMap<String, crate::llm::traits::ToolCall>,
    /// Input buffers for each tool call (tool_id -> accumulated JSON string)
    tool_input_buffers: std::collections::HashMap<String, String>,
}

impl LMStudioToolState {
    fn new() -> Self {
        Self {
            active_tool_calls: std::collections::HashMap::new(),
            tool_input_buffers: std::collections::HashMap::new(),
        }
    }

    fn reset(&mut self) {
        self.active_tool_calls.clear();
        self.tool_input_buffers.clear();
    }

    fn start_tool_call(&mut self, tool_id: String, name: String) {
        let tool_call = crate::llm::traits::ToolCall {
            id: tool_id.clone(),
            name,
            input: serde_json::Value::Null,
        };
        self.active_tool_calls.insert(tool_id.clone(), tool_call);
        self.tool_input_buffers.insert(tool_id, String::new());
    }

    fn add_input_delta(&mut self, tool_id: &str, delta: &str) {
        if let Some(buffer) = self.tool_input_buffers.get_mut(tool_id) {
            buffer.push_str(delta);

            // Try to parse accumulated input as complete JSON
            if let Ok(parsed_input) = serde_json::from_str::<serde_json::Value>(buffer) {
                if let Some(tool_call) = self.active_tool_calls.get_mut(tool_id) {
                    tool_call.input = parsed_input;
                }
            }
        }
    }

    fn get_all_complete_tools(&mut self) -> Vec<crate::llm::traits::ToolCall> {
        // Return all tools that have valid JSON input
        let complete_tools: Vec<_> = self
            .active_tool_calls
            .values()
            .filter(|tool| !tool.input.is_null())
            .cloned()
            .collect();

        if !complete_tools.is_empty() {
            self.reset();
        }

        complete_tools
    }
}

/// LM Studio provider
///
/// This provider connects to a local LM Studio instance and handles
/// OpenAI-compatible API formatting for local models.
///
/// Includes retry logic with exponential backoff to handle model loading delays.
#[derive(Debug)]
pub struct LMStudioProvider {
    base_url: String,
    client: reqwest::Client,
    retry_config: RetryConfig,
}

impl LMStudioProvider {
    /// Create a new LM Studio provider with default retry configuration
    pub async fn new(base_url: String) -> Result<Self, LlmError> {
        Self::with_retry_config(base_url, RetryConfig::lm_studio_default()).await
    }

    /// Create a new LM Studio provider with custom retry configuration
    pub async fn with_retry_config(
        base_url: String,
        retry_config: RetryConfig,
    ) -> Result<Self, LlmError> {
        let client = reqwest::Client::new();

        // TODO: Test connection to LM Studio with retry logic

        Ok(Self {
            base_url,
            client,
            retry_config,
        })
    }

    /// Get the current retry configuration
    pub fn retry_config(&self) -> &RetryConfig {
        &self.retry_config
    }

    /// Update the retry configuration
    pub fn set_retry_config(&mut self, config: RetryConfig) {
        self.retry_config = config;
    }

    /// Make a retryable HTTP request to LM Studio
    async fn make_retryable_request(
        &self,
        endpoint: &str,
        request_body: serde_json::Value,
    ) -> Result<String, LlmError> {
        let url = format!("{}{}", self.base_url, endpoint);
        tracing::trace!("🔵 LM Studio POST to: {}", url);
        tracing::trace!(
            "🔵 LM Studio POST body: {}",
            serde_json::to_string_pretty(&request_body)
                .unwrap_or_else(|_| "<invalid json>".to_string())
        );

        // Clone necessary data for retry closure
        let client = self.client.clone();
        let url_clone = url.clone();
        let request_body_clone = request_body.clone();

        retry_llm_operation(
            move || {
                let client = client.clone();
                let url = url_clone.clone();
                let request_body = request_body_clone.clone();

                Box::pin(async move {
                    // Make HTTP request
                    let response = client
                        .post(&url)
                        .header("Content-Type", "application/json")
                        .json(&request_body)
                        .send()
                        .await
                        .map_err(|e| {
                            tracing::debug!("🔄 LM Studio HTTP request attempt failed: {}", e);
                            LlmError::ProviderError {
                                provider: ProviderType::LmStudio,
                                message: format!("HTTP request failed: {}", e),
                                source: Some(Box::new(e)),
                            }
                        })?;

                    let status = response.status();

                    // Check if response indicates a retryable condition
                    if !status.is_success() {
                        let error_text = response
                            .text()
                            .await
                            .unwrap_or_else(|_| "Unknown error".to_string());

                        // Retryable errors for LM Studio (model loading, startup issues)
                        let _retryable_error = match status.as_u16() {
                            502 | 503 => {
                                tracing::warn!(
                                    "🔄 LM Studio returned {} (model may be loading): {}",
                                    status,
                                    error_text
                                );
                                true
                            }
                            _ => {
                                tracing::error!(
                                    "❌ LM Studio API error {} (non-retryable): {}",
                                    status,
                                    error_text
                                );
                                false
                            }
                        };

                        return Err(LlmError::ProviderError {
                            provider: ProviderType::LmStudio,
                            message: format!("LM Studio API error {}: {}", status, error_text),
                            source: None,
                        });
                    }

                    // Parse response text
                    response.text().await.map_err(|e| {
                        tracing::error!("❌ Failed to read response text: {}", e);
                        LlmError::ProviderError {
                            provider: ProviderType::LmStudio,
                            message: format!("Failed to read response text: {}", e),
                            source: Some(Box::new(e)),
                        }
                    })
                }) as BoxFuture<'_, Result<String, LlmError>>
            },
            &self.retry_config,
        )
        .await
    }
}

#[async_trait]
impl LlmProvider for LMStudioProvider {
    async fn chat(
        &self,
        model_id: &str,
        messages: &Messages,
        config: &ChatConfig,
    ) -> Result<ChatResponse, LlmError> {
        tracing::info!("🔵 LM Studio chat request starting for model: {}", model_id);
        tracing::debug!(
            "🔵 LM Studio request config: temp={:?}, max_tokens={:?}",
            config.temperature,
            config.max_tokens
        );

        // Convert Messages to OpenAI format
        let openai_messages = self.convert_messages_to_openai(messages)?;
        tracing::debug!(
            "🔵 Converted {} messages to OpenAI format",
            openai_messages.len()
        );

        // Build OpenAI-compatible request
        let mut request_body = serde_json::json!({
            "model": model_id,
            "messages": openai_messages,
            "max_tokens": config.max_tokens.unwrap_or(1000),
            "temperature": config.temperature.unwrap_or(0.7),
            "stream": false
        });

        // Add additional parameters if present
        for (key, value) in &config.additional_params {
            request_body[key] = value.clone();
        }

        // Make HTTP request to LM Studio with retry logic
        let start_time = std::time::Instant::now();
        let response_text = self
            .make_retryable_request("/v1/chat/completions", request_body)
            .await?;
        let request_duration = start_time.elapsed();
        tracing::debug!(
            "🔵 LM Studio HTTP request completed in {:?}",
            request_duration
        );

        tracing::trace!("🔵 LM Studio raw response: {}", response_text);

        let response_json: serde_json::Value =
            serde_json::from_str(&response_text).map_err(|e| {
                tracing::error!(
                    "❌ Failed to parse JSON response: {} | Raw response: {}",
                    e,
                    response_text
                );
                LlmError::ProviderError {
                    provider: ProviderType::LmStudio,
                    message: format!("Failed to parse JSON response: {}", e),
                    source: Some(Box::new(e)),
                }
            })?;

        tracing::debug!(
            "🔵 LM Studio parsed JSON response structure: {}",
            serde_json::to_string(&response_json).unwrap_or_else(|_| "<invalid>".to_string())
        );

        // Convert OpenAI response to ChatResponse
        let chat_response = self.convert_openai_response_to_chat_response(response_json)?;

        tracing::info!(
            "✅ LM Studio chat response completed in {:?}, content length: {} chars",
            start_time.elapsed(),
            chat_response.content.len()
        );
        tracing::debug!("🔵 LM Studio response content: {}", chat_response.content);

        Ok(chat_response)
    }

    async fn chat_with_tools(
        &self,
        model_id: &str,
        messages: &Messages,
        tools: &[Tool],
        config: &ChatConfig,
    ) -> Result<ChatResponse, LlmError> {
        if tools.is_empty() {
            // No tools, use basic chat
            return self.chat(model_id, messages, config).await;
        }

        tracing::info!(
            "🔧 LM Studio chat with {} tools for model: {}",
            tools.len(),
            model_id
        );
        let start_time = Instant::now();

        // Convert Messages to OpenAI format
        let openai_messages = self.convert_messages_to_openai(messages)?;

        // Convert tools to OpenAI tool format
        let openai_tools: Vec<serde_json::Value> = tools
            .iter()
            .map(|tool| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema
                    }
                })
            })
            .collect();

        // Map ToolChoice to OpenAI format
        use crate::types::tools::ToolChoice;
        let openai_tool_choice = match &config.tool_choice {
            ToolChoice::Auto => serde_json::json!("auto"),
            ToolChoice::Any => serde_json::json!("required"),
            ToolChoice::Tool { name } => {
                serde_json::json!({"type": "function", "function": {"name": name}})
            }
            // ToolChoice::None is handled at event_loop level (empty tools list)
            ToolChoice::None => serde_json::json!("none"),
        };

        // Build OpenAI-compatible request with tools
        let mut request_body = serde_json::json!({
            "model": model_id,
            "messages": openai_messages,
            "tools": openai_tools,
            "tool_choice": openai_tool_choice,
            "temperature": config.temperature.unwrap_or(0.7),
            "max_tokens": config.max_tokens,
        });

        // Add additional parameters if present in config
        for (key, value) in &config.additional_params {
            request_body[key] = value.clone();
        }

        tracing::debug!(
            "🔵 LM Studio request with tools: {}",
            serde_json::to_string_pretty(&request_body).unwrap_or_else(|_| "<invalid>".to_string())
        );

        // Make HTTP request to LM Studio with retry logic
        let response_text = self
            .make_retryable_request("/v1/chat/completions", request_body)
            .await?;

        let response_json: serde_json::Value =
            serde_json::from_str(&response_text).map_err(|e| LlmError::ProviderError {
                provider: ProviderType::LmStudio,
                message: format!("Failed to parse LM Studio JSON: {}", e),
                source: Some(Box::new(e)),
            })?;

        tracing::debug!(
            "🔵 LM Studio response with tools: {}",
            serde_json::to_string_pretty(&response_json)
                .unwrap_or_else(|_| "<invalid>".to_string())
        );

        // Convert OpenAI response to ChatResponse
        let chat_response =
            self.convert_openai_response_to_chat_response_with_tools(response_json)?;

        // Debug log to show parsed tool calls
        if !chat_response.tool_calls.is_empty() {
            tracing::debug!("🔧 LM Studio parsed tool calls:");
            for (i, tool_call) in chat_response.tool_calls.iter().enumerate() {
                tracing::debug!(
                    "  Tool {}: {} - args: {}",
                    i + 1,
                    tool_call.name,
                    serde_json::to_string(&tool_call.input)
                        .unwrap_or_else(|_| "<invalid>".to_string())
                );
            }
        }

        tracing::info!("✅ LM Studio chat with tools completed in {:?}, content length: {} chars, tool calls: {}",
            start_time.elapsed(), chat_response.content.len(), chat_response.tool_calls.len());

        Ok(chat_response)
    }

    async fn chat_streaming(
        &self,
        model_id: &str,
        messages: &Messages,
        config: &ChatConfig,
    ) -> Result<Box<dyn Stream<Item = StreamEvent> + Send + Unpin>, LlmError> {
        tracing::info!(
            "🌊 LM Studio streaming request starting for model: {}",
            model_id
        );

        // Convert Messages to OpenAI format
        let openai_messages = self.convert_messages_to_openai(messages)?;

        // Build OpenAI-compatible streaming request
        let mut request_body = serde_json::json!({
            "model": model_id,
            "messages": openai_messages,
            "max_tokens": config.max_tokens.unwrap_or(1000),
            "temperature": config.temperature.unwrap_or(0.7),
            "stream": true  // Enable streaming
        });

        // Add additional parameters if present
        for (key, value) in &config.additional_params {
            request_body[key] = value.clone();
        }

        let url = format!("{}/v1/chat/completions", self.base_url);
        tracing::trace!("🌊 LM Studio streaming POST to: {}", url);
        tracing::trace!(
            "🌊 LM Studio streaming POST body: {}",
            serde_json::to_string_pretty(&request_body)
                .unwrap_or_else(|_| "<invalid json>".to_string())
        );

        // JSON ONLY output for debug_json_only example
        if std::env::var("RUST_LOG")
            .unwrap_or_default()
            .contains("debug_json_only")
        {
            println!(
                "OUT {}",
                serde_json::to_string(&request_body).unwrap_or_default()
            );
        }

        // Make streaming HTTP request
        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                tracing::error!("❌ LM Studio streaming request failed: {}", e);
                LlmError::ProviderError {
                    provider: ProviderType::LmStudio,
                    message: format!("Streaming request failed: {}", e),
                    source: Some(Box::new(e)),
                }
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            tracing::error!(
                "❌ LM Studio streaming API error {}: {}",
                status,
                error_text
            );
            return Err(LlmError::ProviderError {
                provider: ProviderType::LmStudio,
                message: format!("LM Studio streaming API error {}: {}", status, error_text),
                source: None,
            });
        }

        tracing::debug!("🌊 LM Studio streaming response received, processing SSE stream...");

        // Convert the response to a stream of events
        let stream = self.parse_sse_stream(response).await?;
        Ok(stream)
    }

    async fn chat_streaming_with_tools(
        &self,
        model_id: &str,
        messages: &Messages,
        tools: &[Tool],
        config: &ChatConfig,
    ) -> Result<Box<dyn Stream<Item = StreamEvent> + Send + Unpin>, LlmError> {
        tracing::info!(
            "🔧🌊 LM Studio streaming with tools request starting for model: {} with {} tools",
            model_id,
            tools.len()
        );

        // Convert Messages to OpenAI format
        let openai_messages = self.convert_messages_to_openai(messages)?;

        // Debug: Log the conversation being sent to LM Studio
        tracing::debug!(
            "🔧 LM Studio conversation has {} messages:",
            openai_messages.len()
        );
        for (i, msg) in openai_messages.iter().enumerate() {
            tracing::debug!("🔧 Message {}: {:?}", i, msg);
        }

        // Convert tools to OpenAI format
        let openai_tools = self.convert_tools_to_openai(tools)?;

        // Map ToolChoice to OpenAI format for streaming
        use crate::types::tools::ToolChoice;
        let openai_tool_choice = match &config.tool_choice {
            ToolChoice::Auto => serde_json::json!("auto"),
            ToolChoice::Any => serde_json::json!("required"),
            ToolChoice::Tool { name } => {
                serde_json::json!({"type": "function", "function": {"name": name}})
            }
            // ToolChoice::None handled at event_loop level (empty tools list)
            ToolChoice::None => serde_json::json!("none"),
        };

        // Build OpenAI-compatible streaming request with tools
        let mut request_body = serde_json::json!({
            "model": model_id,
            "messages": openai_messages,
            "max_tokens": config.max_tokens.unwrap_or(1000),
            "temperature": config.temperature.unwrap_or(0.7),
            "stream": true,  // Enable streaming
            "tools": openai_tools,  // Include tools
            "tool_choice": openai_tool_choice
        });

        // Add additional parameters if present
        for (key, value) in &config.additional_params {
            request_body[key] = value.clone();
        }

        let url = format!("{}/v1/chat/completions", self.base_url);
        tracing::debug!("🔧🌊 LM Studio streaming with tools POST to: {}", url);
        tracing::debug!("🔧🌊 LM Studio streaming with tools REQUEST:");
        tracing::debug!("-----------------------------------------------------------");
        tracing::debug!(
            "{}",
            serde_json::to_string_pretty(&request_body)
                .unwrap_or_else(|_| "<invalid json>".to_string())
        );
        tracing::debug!("-----------------------------------------------------------");

        // JSON ONLY output for debug_json_only example
        if std::env::var("RUST_LOG")
            .unwrap_or_default()
            .contains("debug_json_only")
        {
            println!(
                "OUT {}",
                serde_json::to_string(&request_body).unwrap_or_default()
            );
        }

        // Make streaming HTTP request
        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                tracing::error!("❌ LM Studio streaming with tools request failed: {}", e);
                LlmError::ProviderError {
                    provider: ProviderType::LmStudio,
                    message: format!("Streaming with tools request failed: {}", e),
                    source: Some(Box::new(e)),
                }
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            tracing::error!(
                "❌ LM Studio streaming with tools API error {}: {}",
                status,
                error_text
            );
            return Err(LlmError::ProviderError {
                provider: ProviderType::LmStudio,
                message: format!(
                    "LM Studio streaming with tools API error {}: {}",
                    status, error_text
                ),
                source: None,
            });
        }

        tracing::debug!(
            "🔧🌊 LM Studio streaming with tools response received, processing SSE stream..."
        );

        // Convert the response to a stream of events (same parsing logic can handle tools)
        let stream = self.parse_sse_stream(response).await?;
        Ok(stream)
    }

    async fn health_check(&self) -> Result<HealthStatus, LlmError> {
        // Test connection to LM Studio with conservative retry (1 retry for connection failures only)
        let url = format!("{}/v1/models", self.base_url);

        let start = std::time::Instant::now();
        let conservative_retry = RetryConfig {
            max_attempts: 1, // Only 1 retry for health checks
            initial_delay: std::time::Duration::from_millis(500),
            max_delay: std::time::Duration::from_millis(1000),
            backoff_multiplier: 2.0,
            jitter: false,
        };

        let client = self.client.clone();
        let url_clone = url.clone();

        let result = retry_llm_operation(
            move || {
                let client = client.clone();
                let url = url_clone.clone();

                Box::pin(async move {
                    client.get(&url).send().await.map_err(|e| {
                        // Only retry on connection errors, not HTTP status errors
                        if e.is_connect() || e.is_timeout() {
                            tracing::debug!("🔄 Health check connection failed (retryable): {}", e);
                        } else {
                            tracing::debug!("🔄 Health check failed (non-retryable): {}", e);
                        }
                        LlmError::NetworkError {
                            message: format!("Health check request failed: {}", e),
                            source: Some(Box::new(e)),
                        }
                    })
                }) as BoxFuture<'_, Result<reqwest::Response, LlmError>>
            },
            &conservative_retry,
        )
        .await;

        let latency = start.elapsed().as_millis() as u64;

        match result {
            Ok(response) if response.status().is_success() => Ok(HealthStatus {
                healthy: true,
                provider: ProviderType::LmStudio,
                latency_ms: Some(latency),
                error: None,
            }),
            Ok(response) => Ok(HealthStatus {
                healthy: false,
                provider: ProviderType::LmStudio,
                latency_ms: Some(latency),
                error: Some(format!(
                    "HTTP {}: {}",
                    response.status(),
                    response.status().canonical_reason().unwrap_or("Unknown")
                )),
            }),
            Err(e) => Ok(HealthStatus {
                healthy: false,
                provider: ProviderType::LmStudio,
                latency_ms: None,
                error: Some(format!("Connection failed: {}", e)),
            }),
        }
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supports_streaming: true,
            supports_tools: false, // Depends on model and LM Studio configuration
            supports_thinking: false,
            supports_vision: false, // Depends on model
            supports_prompt_caching: false, // Local models don't support prompt caching
            supports_tool_caching: false,
            max_tokens: Some(4096), // LM Studio default - varies by model but 4096 is common default
            available_models: vec![
                "google/gemma-3-12b".to_string(),
                "google/gemma-3-27b".to_string(),
                "llama-3-70b".to_string(),
                "mistralai/mistral-7b-instruct-v0.3".to_string(),
                "tessa-rust-t1-7b".to_string(),
            ],
        }
    }

    fn provider_type(&self) -> ProviderType {
        ProviderType::LmStudio
    }

    fn supported_models(&self) -> Vec<&'static str> {
        vec![
            "google/gemma-3-12b",
            "google/gemma-3-27b",
            "llama-3-70b",
            "mistralai/mistral-7b-instruct-v0.3",
            "tessa-rust-t1-7b",
        ]
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl LMStudioProvider {
    /// Convert Stood Messages format to OpenAI chat completion format
    fn convert_messages_to_openai(&self, messages: &Messages) -> Result<Vec<Value>, LlmError> {
        let mut openai_messages = Vec::new();

        for message in &messages.messages {
            let role = match message.role {
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
                MessageRole::System => "system",
            };

            // Convert content blocks to OpenAI format (handle tool use and results)
            let mut content_parts = Vec::new();
            let mut tool_calls = Vec::new();
            let mut tool_results = Vec::new();

            for block in &message.content {
                match block {
                    ContentBlock::Text { text } => {
                        content_parts.push(text.clone());
                    }
                    ContentBlock::ToolUse { id, name, input } => {
                        // Convert tool use to OpenAI tool call format
                        // Ensure arguments is always a valid JSON object string
                        let arguments_str = if input.is_null() || !input.is_object() {
                            "{}".to_string()
                        } else {
                            serde_json::to_string(input).unwrap_or_else(|_| "{}".to_string())
                        };
                        tool_calls.push(serde_json::json!({
                            "id": id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": arguments_str
                            }
                        }));
                    }
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content: tool_content,
                        ..
                    } => {
                        // Convert tool result to OpenAI tool message format
                        // Tool results should always be converted to tool messages regardless of the containing message role
                        tool_results.push(serde_json::json!({
                            "role": "tool",
                            "tool_call_id": tool_use_id,
                            "content": tool_content.to_display_string()
                        }));
                    }
                    _ => {
                        // Skip other content types for now
                    }
                }
            }

            let content = content_parts.join(" ");

            if !content.is_empty() || !tool_calls.is_empty() {
                let mut message_json = serde_json::json!({
                    "role": role,
                    "content": if content.is_empty() { None } else { Some(content) }
                });

                // Add tool calls if present
                if !tool_calls.is_empty() {
                    message_json["tool_calls"] = serde_json::Value::Array(tool_calls);
                }

                openai_messages.push(message_json);
            }

            // Add tool results after the current message
            for tool_result in tool_results {
                openai_messages.push(tool_result);
            }
        }

        Ok(openai_messages)
    }

    /// Convert Stood Tool format to OpenAI tools format
    fn convert_tools_to_openai(&self, tools: &[Tool]) -> Result<Vec<serde_json::Value>, LlmError> {
        let openai_tools = tools
            .iter()
            .map(|tool| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema
                    }
                })
            })
            .collect();
        Ok(openai_tools)
    }

    /// Convert OpenAI chat completion response to Stood ChatResponse format
    fn convert_openai_response_to_chat_response(
        &self,
        response: Value,
    ) -> Result<ChatResponse, LlmError> {
        // Extract the completion from OpenAI response format
        let choices = response
            .get("choices")
            .and_then(|c| c.as_array())
            .ok_or_else(|| LlmError::ProviderError {
                provider: ProviderType::LmStudio,
                message: "Invalid response format: missing choices array".to_string(),
                source: None,
            })?;

        if choices.is_empty() {
            return Err(LlmError::ProviderError {
                provider: ProviderType::LmStudio,
                message: "Invalid response format: empty choices array".to_string(),
                source: None,
            });
        }

        let first_choice = &choices[0];
        let message = first_choice
            .get("message")
            .ok_or_else(|| LlmError::ProviderError {
                provider: ProviderType::LmStudio,
                message: "Invalid response format: missing message".to_string(),
                source: None,
            })?;

        let content = message
            .get("content")
            .and_then(|c| c.as_str())
            .ok_or_else(|| LlmError::ProviderError {
                provider: ProviderType::LmStudio,
                message: "Invalid response format: missing content".to_string(),
                source: None,
            })?;

        // Extract usage information if available
        let usage = response.get("usage").and_then(|u| {
            Some(crate::llm::traits::Usage {
                input_tokens: u.get("prompt_tokens")?.as_u64()? as u32,
                output_tokens: u.get("completion_tokens")?.as_u64()? as u32,
                total_tokens: u.get("total_tokens")?.as_u64()? as u32,
                // LM Studio doesn't support prompt caching
                cache_read_tokens: None,
                cache_write_tokens: None,
            })
        });

        // Create metadata with finish reason
        let mut metadata = std::collections::HashMap::new();
        if let Some(finish_reason) = first_choice.get("finish_reason").and_then(|r| r.as_str()) {
            metadata.insert(
                "finish_reason".to_string(),
                serde_json::Value::String(finish_reason.to_string()),
            );
        }

        Ok(ChatResponse {
            content: content.to_string(),
            tool_calls: Vec::new(), // LM Studio typically doesn't support function calling
            thinking: None,         // LM Studio doesn't have thinking mode
            usage,
            metadata,
        })
    }

    /// Convert OpenAI response to ChatResponse with tool calling support
    fn convert_openai_response_to_chat_response_with_tools(
        &self,
        response: serde_json::Value,
    ) -> Result<ChatResponse, LlmError> {
        let choices = response
            .get("choices")
            .and_then(|c| c.as_array())
            .ok_or_else(|| LlmError::ProviderError {
                provider: ProviderType::LmStudio,
                message: "Invalid response format: missing choices array".to_string(),
                source: None,
            })?;

        if choices.is_empty() {
            return Err(LlmError::ProviderError {
                provider: ProviderType::LmStudio,
                message: "Invalid response format: empty choices array".to_string(),
                source: None,
            });
        }

        let first_choice = &choices[0];
        let message = first_choice
            .get("message")
            .ok_or_else(|| LlmError::ProviderError {
                provider: ProviderType::LmStudio,
                message: "Invalid response format: missing message".to_string(),
                source: None,
            })?;

        // Extract content (might be null if only tool calls are present)
        let content = message
            .get("content")
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();

        // Extract tool calls if present
        let mut tool_calls = Vec::new();
        if let Some(calls) = message.get("tool_calls").and_then(|tc| tc.as_array()) {
            for call in calls {
                if let (Some(id), Some(function)) = (
                    call.get("id").and_then(|i| i.as_str()),
                    call.get("function"),
                ) {
                    if let Some(name) = function.get("name").and_then(|n| n.as_str()) {
                        // Handle arguments - they might be a string (JSON) or already parsed object
                        let parsed_args: serde_json::Value = match function.get("arguments") {
                            Some(serde_json::Value::String(s)) => {
                                // Arguments came as JSON string - try to parse it
                                serde_json::from_str(s).unwrap_or_else(|parse_err| {
                                    tracing::warn!(
                                        "Failed to parse tool arguments as JSON: {} | Raw: {}",
                                        parse_err,
                                        s
                                    );
                                    // If parsing fails, return the string as-is
                                    serde_json::Value::String(s.clone())
                                })
                            }
                            Some(obj) => {
                                // Arguments already parsed as object
                                obj.clone()
                            }
                            None => serde_json::Value::Object(serde_json::Map::new()),
                        };

                        tool_calls.push(crate::llm::traits::ToolCall {
                            id: id.to_string(),
                            name: name.to_string(),
                            input: parsed_args,
                        });
                    }
                }
            }
        }

        // Extract usage information if available
        let usage = response.get("usage").and_then(|u| {
            Some(crate::llm::traits::Usage {
                input_tokens: u.get("prompt_tokens")?.as_u64()? as u32,
                output_tokens: u.get("completion_tokens")?.as_u64()? as u32,
                total_tokens: u.get("total_tokens")?.as_u64()? as u32,
                // LM Studio doesn't support prompt caching
                cache_read_tokens: None,
                cache_write_tokens: None,
            })
        });

        // Create metadata with finish reason
        let mut metadata = std::collections::HashMap::new();
        if let Some(finish_reason) = first_choice.get("finish_reason").and_then(|r| r.as_str()) {
            metadata.insert(
                "finish_reason".to_string(),
                serde_json::Value::String(finish_reason.to_string()),
            );
        }

        Ok(ChatResponse {
            content,
            tool_calls,
            thinking: None, // LM Studio doesn't have thinking mode
            usage,
            metadata,
        })
    }

    async fn parse_sse_stream(
        &self,
        response: reqwest::Response,
    ) -> Result<Box<dyn Stream<Item = crate::llm::traits::StreamEvent> + Send + Unpin>, LlmError>
    {
        use crate::llm::traits::StreamEvent;
        use futures::stream::{StreamExt, TryStreamExt};

        let byte_stream = response.bytes_stream().map_err(std::io::Error::other);

        let sse_stream = async_stream::stream! {
            let mut buffer = String::new();
            let mut lines_stream = byte_stream.map(|chunk_result| {
                chunk_result.map(|bytes| String::from_utf8_lossy(&bytes).to_string())
            });

            // Claude-style stateful tool management
            let mut tool_state = LMStudioToolState::new();

            // Track content for token estimation
            let mut total_content = String::new();

            tracing::debug!("🌊 Starting SSE stream processing with stateful tool management...");
            let mut chunk_count = 0;
            let mut event_count = 0;

            while let Some(chunk_result) = lines_stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        chunk_count += 1;
                        tracing::trace!("🌊 Received HTTP chunk #{}: {} bytes: '{}'",
                            chunk_count, chunk.len(), chunk.chars().take(100).collect::<String>());

                        buffer.push_str(&chunk);
                        tracing::trace!("🌊 Current buffer size: {} bytes", buffer.len());

                        // Process complete lines
                        while let Some(newline_pos) = buffer.find('\n') {
                            let line = buffer[..newline_pos].trim().to_string();
                            buffer = buffer[newline_pos + 1..].to_string();

                            tracing::trace!("🌊 Processing SSE line: '{}'", line);

                            // Parse SSE line with stateful tool management
                            if let Some(events) = Self::parse_sse_line_with_state(&line, &mut tool_state) {
                                for event in events {
                                    event_count += 1;

                                    // Track text content for token estimation
                                    if let StreamEvent::ContentBlockDelta { delta: crate::llm::traits::ContentBlockDelta::Text { text }, .. } = &event {
                                        total_content.push_str(text);
                                    }

                                    tracing::debug!("🌊 Yielding event #{}: {:?}", event_count,
                                        match &event {
                                            StreamEvent::ContentBlockDelta { delta, .. } => {
                                                match delta {
                                                    crate::llm::traits::ContentBlockDelta::Text { text } => format!("ContentBlockDelta::Text('{}')", text),
                                                    crate::llm::traits::ContentBlockDelta::ToolUse { tool_call_id, input_delta } => format!("ContentBlockDelta::ToolUse({}:'{}')", tool_call_id, input_delta),
                                                    _ => "ContentBlockDelta::Other".to_string(),
                                                }
                                            },
                                            StreamEvent::ContentBlockStart { block_type, .. } => format!("ContentBlockStart({:?})", block_type),
                                            StreamEvent::ContentBlockStop { .. } => "ContentBlockStop".to_string(),
                                            StreamEvent::MessageStop { .. } => "MessageStop".to_string(),
                                            StreamEvent::Error { error } => format!("Error({})", error),
                                            _ => "Other".to_string(),
                                        }
                                    );
                                    yield event;
                                }
                            } else {
                                tracing::trace!("🌊 No events parsed from line: '{}'", line);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("❌ SSE stream error: {}", e);
                        yield StreamEvent::Error {
                            error: format!("Stream error: {}", e),
                        };
                        break;
                    }
                }
            }

            tracing::debug!("🌊 SSE stream completed - processed {} chunks, yielded {} events", chunk_count, event_count);

            // If we haven't sent a Done event yet, send one now (handles cases where [DONE] is missing)
            if event_count > 0 {
                tracing::debug!("🌊 Sending final MessageStop event since stream ended");

                // Estimate token usage based on content length (approximation)
                // Typical ratio is ~4 characters per token for English text
                let output_tokens = (total_content.len() / 4).max(1) as u32;
                let input_tokens = 100; // Rough estimate for input - this could be improved

                let usage = Some(crate::llm::traits::Usage {
                    input_tokens,
                    output_tokens,
                    total_tokens: input_tokens + output_tokens,
                    // LM Studio doesn't support prompt caching
                    cache_read_tokens: None,
                    cache_write_tokens: None,
                });

                tracing::debug!("🌊 LM Studio estimated token usage: input={}, output={}, total={}",
                              input_tokens, output_tokens, input_tokens + output_tokens);

                yield StreamEvent::Done { usage };
            }
        };

        // Use boxed() to make the stream Unpin
        Ok(Box::new(sse_stream.boxed()))
    }

    /// Parse SSE line with stateful tool management (following Claude's pattern)
    fn parse_sse_line_with_state(
        line: &str,
        tool_state: &mut LMStudioToolState,
    ) -> Option<Vec<crate::llm::traits::StreamEvent>> {
        use crate::llm::traits::{ContentBlockDelta, ContentBlockType, StreamEvent, Usage};

        let mut events = Vec::new();

        // SSE format: "data: {json}"
        if let Some(data) = line.strip_prefix("data: ") {
            // Debug log the raw SSE data
            if tracing::level_enabled!(tracing::Level::DEBUG) {
                tracing::debug!("🔧🌊 LM Studio SSE data: {}", data);
            }

            // JSON ONLY output for debug_json_only example
            if std::env::var("RUST_LOG")
                .unwrap_or_default()
                .contains("debug_json_only")
                && data != "[DONE]"
            {
                println!("IN {}", data);
            }

            if data.trim() == "[DONE]" {
                tracing::debug!("🌊 Received [DONE] marker - finalizing any remaining tools");

                // Finalize any remaining tool calls like Claude does at content_block_stop
                let complete_tools = tool_state.get_all_complete_tools();
                for tool_call in complete_tools {
                    tracing::debug!(
                        "🔧🌊 Finalizing tool call at stream end: {} with input: {}",
                        tool_call.name,
                        serde_json::to_string(&tool_call.input).unwrap_or_default()
                    );

                    // Emit ContentBlockStart for tool use (like Claude)
                    events.push(StreamEvent::ContentBlockStart {
                        block_type: ContentBlockType::ToolUse,
                        block_index: 0,
                    });

                    // Emit complete tool call (like Claude at content_block_stop)
                    events.push(StreamEvent::ToolCallStart {
                        tool_call: tool_call.clone(),
                    });

                    // Emit complete input as delta for compatibility
                    if !tool_call.input.is_null() {
                        let input_str = serde_json::to_string(&tool_call.input).unwrap_or_default();
                        events.push(StreamEvent::ContentBlockDelta {
                            delta: ContentBlockDelta::ToolUse {
                                tool_call_id: tool_call.id.clone(),
                                input_delta: input_str,
                            },
                            block_index: 0,
                        });
                    }

                    // Emit ContentBlockStop (like Claude)
                    events.push(StreamEvent::ContentBlockStop { block_index: 0 });
                }

                // Stream ending - don't send MessageStop, let main stream handle Done event
                // events.push(StreamEvent::MessageStop {
                //     stop_reason: Some("end_turn".to_string())
                // });
                return Some(events);
            }

            // Parse JSON chunk
            match serde_json::from_str::<serde_json::Value>(data) {
                Ok(json) => {
                    tracing::trace!(
                        "🌊 Parsed SSE chunk: {}",
                        serde_json::to_string(&json).unwrap_or_default()
                    );

                    // Extract delta content from OpenAI streaming format
                    if let Some(choices) = json.get("choices").and_then(|c| c.as_array()) {
                        if let Some(choice) = choices.first() {
                            if let Some(delta) = choice.get("delta") {
                                // Handle multiple tool calls in delta (fix major bug)
                                if let Some(tool_calls) =
                                    delta.get("tool_calls").and_then(|tc| tc.as_array())
                                {
                                    tracing::trace!(
                                        "🔧🌊 Found {} tool calls in delta",
                                        tool_calls.len()
                                    );

                                    // Process ALL tool calls, not just the first one (major fix)
                                    for tool_call in tool_calls {
                                        let tool_index = tool_call
                                            .get("index")
                                            .and_then(|i| i.as_u64())
                                            .unwrap_or(0);
                                        let tool_id = if let Some(actual_id) =
                                            tool_call.get("id").and_then(|i| i.as_str())
                                        {
                                            actual_id.to_string()
                                        } else {
                                            format!("tool_call_{}", tool_index)
                                        };

                                        if let Some(function) = tool_call.get("function") {
                                            // Tool call start - has function name
                                            if let Some(name) =
                                                function.get("name").and_then(|n| n.as_str())
                                            {
                                                tracing::debug!(
                                                    "🔧🌊 Starting tool call: {} ({})",
                                                    name,
                                                    tool_id
                                                );
                                                tool_state.start_tool_call(
                                                    tool_id.clone(),
                                                    name.to_string(),
                                                );

                                                // Emit ContentBlockStart for tool use (like Claude)
                                                events.push(StreamEvent::ContentBlockStart {
                                                    block_type: ContentBlockType::ToolUse,
                                                    block_index: tool_index as usize,
                                                });
                                            }

                                            // Tool call delta - function arguments
                                            if let Some(arguments) = function
                                                .get("arguments")
                                                .and_then(|args| args.as_str())
                                            {
                                                if !arguments.is_empty() {
                                                    tracing::debug!(
                                                        "🔧🌊 Adding input delta for {}: '{}'",
                                                        tool_id,
                                                        arguments
                                                    );
                                                    tool_state.add_input_delta(&tool_id, arguments);

                                                    // Emit ContentBlockDelta for tool use (like Claude)
                                                    events.push(StreamEvent::ContentBlockDelta {
                                                        delta: ContentBlockDelta::ToolUse {
                                                            tool_call_id: tool_id,
                                                            input_delta: arguments.to_string(),
                                                        },
                                                        block_index: tool_index as usize,
                                                    });
                                                }
                                            }
                                        }
                                    }
                                }

                                // Handle regular content delta
                                if let Some(content) = delta.get("content").and_then(|c| c.as_str())
                                {
                                    if !content.is_empty() {
                                        tracing::trace!("🌊 Content delta: '{}'", content);
                                        events.push(StreamEvent::ContentBlockDelta {
                                            delta: ContentBlockDelta::Text {
                                                text: content.to_string(),
                                            },
                                            block_index: 0,
                                        });
                                    }
                                }
                            }

                            // Handle finish_reason (finalize tools like Claude's content_block_stop)
                            if let Some(finish_reason) = choice.get("finish_reason") {
                                if !finish_reason.is_null() {
                                    let reason_str = finish_reason.as_str().unwrap_or("unknown");
                                    tracing::debug!(
                                        "🌊 Finish reason: {} - finalizing tools",
                                        reason_str
                                    );

                                    // Finalize all complete tool calls (like Claude at content_block_stop)
                                    let complete_tools = tool_state.get_all_complete_tools();
                                    for tool_call in complete_tools {
                                        tracing::info!(
                                            "🔧🌊 Finalizing complete tool call: {} with input: {}",
                                            tool_call.name,
                                            serde_json::to_string(&tool_call.input)
                                                .unwrap_or_default()
                                        );

                                        // Emit complete tool call (like Claude)
                                        events.push(StreamEvent::ToolCallStart {
                                            tool_call: tool_call.clone(),
                                        });

                                        // Emit ContentBlockStop for this tool (like Claude)
                                        events
                                            .push(StreamEvent::ContentBlockStop { block_index: 0 });
                                    }

                                    // Stream ending - don't send MessageStop, let main stream handle Done event
                                    // events.push(StreamEvent::MessageStop {
                                    //     stop_reason: Some(reason_str.to_string()),
                                    // });
                                }
                            }
                        }
                    }

                    // Handle usage information - only emit metadata if we have valid token data
                    if let Some(usage_data) = json.get("usage") {
                        let input_tokens = usage_data
                            .get("prompt_tokens")
                            .and_then(|t| t.as_u64())
                            .unwrap_or(0) as u32;
                        let output_tokens = usage_data
                            .get("completion_tokens")
                            .and_then(|t| t.as_u64())
                            .unwrap_or(0) as u32;
                        let total_tokens = usage_data
                            .get("total_tokens")
                            .and_then(|t| t.as_u64())
                            .unwrap_or(0) as u32;

                        // Only emit metadata if we have valid (non-zero) token data
                        // This allows fallback estimation to work when LM Studio doesn't provide usage data
                        if input_tokens > 0 || output_tokens > 0 || total_tokens > 0 {
                            let usage = Usage {
                                input_tokens,
                                output_tokens,
                                total_tokens,
                                // LM Studio doesn't support prompt caching
                                cache_read_tokens: None,
                                cache_write_tokens: None,
                            };

                            tracing::debug!("🌊 Usage metadata: {:?}", usage);
                            events.push(StreamEvent::Metadata { usage: Some(usage) });
                        } else {
                            tracing::debug!("🌊 Skipping zero-token usage metadata, allowing fallback estimation");
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("⚠️ Failed to parse SSE JSON: {} | Data: {}", e, data);
                    // Convert JSON parse errors to StreamEvent::Error to properly handle 404 and other errors
                    events.push(StreamEvent::Error {
                        error: format!("Failed to parse SSE response: {} | Data: {}", e, data),
                    });
                }
            }
        }

        if events.is_empty() {
            None
        } else {
            Some(events)
        }
    }
}
