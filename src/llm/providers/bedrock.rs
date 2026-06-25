//! AWS Bedrock provider implementation.
//!
//! This provider owns ALL Bedrock-specific logic including request formatting,
//! response parsing, streaming, and error handling for Claude, Nova, and Llama models.

use crate::llm::traits::{
    CacheStrategy, ChatConfig, ChatResponse, HealthStatus, LlmError, LlmProvider,
    ProviderCapabilities, ProviderType, StreamEvent, Tool,
};
use crate::types::{ContentBlock, MessageRole, Messages};
use async_trait::async_trait;
use aws_sdk_bedrockruntime::Client as BedrockRuntimeClient;
#[allow(unused_imports)] // Used for future vision/image features
use base64;
use futures::Stream;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Instant;
use tracing::{debug, info};
use uuid::Uuid;

/// Model-specific streaming strategy for handling different Bedrock model formats
#[derive(Debug, Clone)]
enum ModelType {
    Claude,
    Nova,
}

/// Flexible tool state management for different Bedrock models
#[derive(Debug)]
struct ToolState {
    current_tool_call: Option<crate::llm::traits::ToolCall>,
    tool_input_buffer: String,
    model_type: ModelType,
}

impl ToolState {
    fn new(model_type: ModelType) -> Self {
        Self {
            current_tool_call: None,
            tool_input_buffer: String::new(),
            model_type,
        }
    }

    fn reset(&mut self) {
        self.current_tool_call = None;
        self.tool_input_buffer.clear();
    }
}

/// AWS Bedrock provider
///
/// This provider handles all AWS Bedrock models (Claude, Nova, Llama) and owns
/// all implementation details including model-specific request formatting.
#[derive(Debug)]
pub struct BedrockProvider {
    /// AWS Bedrock Runtime client
    client: BedrockRuntimeClient,
    /// AWS config for the client
    #[allow(dead_code)] // Stored for potential future features like region switching
    aws_config: aws_config::SdkConfig,
    /// Last request JSON for raw capture (if enabled)
    last_request_json: std::sync::Arc<std::sync::Mutex<Option<String>>>,
}

impl BedrockProvider {
    /// Create a new Bedrock provider
    pub async fn new(region: Option<String>) -> Result<Self, LlmError> {
        crate::perf_checkpoint!("stood.bedrock.new.start");

        // Configure AWS SDK
        crate::perf_checkpoint!("stood.bedrock.config_loader_setup");
        let mut config_loader = aws_config::defaults(aws_config::BehaviorVersion::latest());

        if let Some(region) = region {
            config_loader = config_loader.region(aws_config::Region::new(region));
        }

        // AWS config load (MAIN BOTTLENECK - TLS/SSL initialization)
        let aws_config = crate::perf_timed!("stood.bedrock.aws_config_load", {
            config_loader.load().await
        });

        // Client creation
        let client = crate::perf_timed!("stood.bedrock.client_new", {
            BedrockRuntimeClient::new(&aws_config)
        });

        crate::perf_checkpoint!("stood.bedrock.new.end");
        Ok(Self {
            client,
            aws_config,
            last_request_json: std::sync::Arc::new(std::sync::Mutex::new(None)),
        })
    }

    /// Create a new Bedrock provider with custom AWS credentials
    ///
    /// This allows injecting credentials programmatically instead of relying on
    /// the default AWS credential provider chain. Useful for applications that
    /// manage credentials centrally (e.g., AWS Identity Center, STS assume role).
    ///
    /// # Arguments
    /// * `region` - Optional AWS region (defaults to AWS SDK default)
    /// * `access_key` - AWS access key ID
    /// * `secret_key` - AWS secret access key
    /// * `session_token` - Optional session token for temporary credentials
    ///
    /// # Example
    /// ```no_run
    /// # use stood::llm::providers::bedrock::BedrockProvider;
    /// # async fn example() -> Result<(), stood::llm::traits::LlmError> {
    /// let provider = BedrockProvider::with_credentials(
    ///     Some("us-east-1".to_string()),
    ///     "AKIA...".to_string(),
    ///     "secret".to_string(),
    ///     Some("token".to_string())
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn with_credentials(
        region: Option<String>,
        access_key: String,
        secret_key: String,
        session_token: Option<String>,
    ) -> Result<Self, LlmError> {
        crate::perf_checkpoint!("stood.bedrock.with_credentials.start");

        // Create credentials from provided parameters
        crate::perf_checkpoint!("stood.bedrock.create_credentials");
        let creds = aws_sdk_bedrockruntime::config::Credentials::new(
            access_key,
            secret_key,
            session_token,
            None, // expiration - None means no known expiration
            "StoodLibraryCustomCredentials",
        );

        // Configure AWS SDK with custom credentials
        crate::perf_checkpoint!("stood.bedrock.config_loader_setup");
        let mut config_loader =
            aws_config::defaults(aws_config::BehaviorVersion::latest()).credentials_provider(creds);

        if let Some(region) = region {
            config_loader = config_loader.region(aws_config::Region::new(region));
        }

        // AWS config load (MAIN BOTTLENECK - TLS/SSL initialization ~950ms on first call)
        let aws_config = crate::perf_timed!("stood.bedrock.aws_config_load", {
            config_loader.load().await
        });

        // Client creation
        let client = crate::perf_timed!("stood.bedrock.client_new", {
            BedrockRuntimeClient::new(&aws_config)
        });

        crate::perf_checkpoint!("stood.bedrock.with_credentials.end");
        Ok(Self {
            client,
            aws_config,
            last_request_json: std::sync::Arc::new(std::sync::Mutex::new(None)),
        })
    }

    /// Store the last request JSON for raw capture
    fn store_request_json(&self, request_json: &str) {
        if let Ok(mut last_request) = self.last_request_json.lock() {
            *last_request = Some(request_json.to_string());
        }
    }

    /// Get the last request JSON for raw capture (returns None if capture disabled or no request)
    pub fn get_last_request_json(&self) -> Option<String> {
        self.last_request_json.lock().ok()?.clone()
    }

    /// Check if a model supports prompt caching
    ///
    /// Returns `true` if the model supports prompt caching on AWS Bedrock.
    /// Currently supported: Claude models and Nova models.
    /// Not supported: Mistral models.
    pub fn supports_prompt_caching(model_id: &str) -> bool {
        model_id.contains("anthropic.claude") || model_id.contains("amazon.nova")
    }

    /// Check if a model supports tool definition caching
    ///
    /// Returns `true` if the model supports caching tool definitions.
    /// Currently only Claude models support tool caching.
    /// Nova models only support system prompt caching.
    pub fn supports_tool_caching(model_id: &str) -> bool {
        model_id.contains("anthropic.claude")
    }

    /// Get the model family for cache-specific behavior
    fn get_model_family(model_id: &str) -> Option<&'static str> {
        if model_id.contains("anthropic.claude") {
            Some("claude")
        } else if model_id.contains("amazon.nova") {
            Some("nova")
        } else if model_id.contains("mistral.mistral") {
            Some("mistral")
        } else {
            None
        }
    }

    /// Build request body for Bedrock API
    fn build_request_body(
        &self,
        messages: &Messages,
        model_id: &str,
        tools: &[Tool],
        config: &ChatConfig,
    ) -> Result<String, LlmError> {
        let operation_id = Uuid::new_v4().to_string();

        // Validate cache strategy for the model
        if !matches!(config.cache_strategy, CacheStrategy::Disabled) {
            if !Self::supports_prompt_caching(model_id) {
                debug!(
                    "[{}] ⚠️ Prompt caching requested but model '{}' does not support it. Caching disabled.",
                    operation_id, model_id
                );
            } else {
                let model_family = Self::get_model_family(model_id).unwrap_or("unknown");
                debug!(
                    "[{}] 🗄️ Prompt caching enabled for {} model with strategy {:?}",
                    operation_id, model_family, config.cache_strategy
                );
            }
        }

        // Route to appropriate builder based on model family
        if model_id.contains("anthropic.claude") {
            self.build_claude_request(messages, tools, config, &operation_id)
        } else if model_id.contains("amazon.nova") {
            self.build_nova_request(messages, tools, config, &operation_id)
        } else if model_id.contains("mistral.mistral") {
            self.build_mistral_request(messages, tools, config, &operation_id)
        } else {
            Err(LlmError::ModelNotFound {
                model_id: model_id.to_string(),
                provider: ProviderType::Bedrock,
            })
        }
    }

    /// Build Claude-specific request
    fn build_claude_request(
        &self,
        messages: &Messages,
        tools: &[Tool],
        config: &ChatConfig,
        operation_id: &str,
    ) -> Result<String, LlmError> {
        let mut request_messages = Vec::new();

        // Use system_prompt from Messages struct if available, otherwise extract from System role messages
        let mut system_prompt = messages.system_prompt.clone();

        // Process messages
        for message in &messages.messages {
            match message.role {
                MessageRole::System => {
                    // Extract system prompt from messages (fallback if not in Messages struct)
                    if system_prompt.is_none() {
                        let text = message
                            .content
                            .iter()
                            .filter_map(|block| match block {
                                ContentBlock::Text { text } => Some(text.as_str()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join(" ");
                        if !text.is_empty() {
                            system_prompt = Some(text);
                        }
                    }
                }
                MessageRole::User | MessageRole::Assistant => {
                    let mut content = Vec::new();

                    for block in &message.content {
                        match block {
                            ContentBlock::Text { text } => {
                                content.push(json!({
                                    "type": "text",
                                    "text": text
                                }));
                            }
                            ContentBlock::ToolUse { id, name, input } => {
                                // Claude tool use format - ensure input is never null or invalid
                                let safe_input = if input.is_null() || !input.is_object() {
                                    // If input is null or not an object, use empty object
                                    serde_json::Value::Object(serde_json::Map::new())
                                } else {
                                    input.clone()
                                };
                                content.push(json!({
                                    "type": "tool_use",
                                    "id": id,
                                    "name": name,
                                    "input": safe_input
                                }));
                            }
                            ContentBlock::ToolResult {
                                tool_use_id,
                                content: tool_content,
                                is_error,
                            } => {
                                content.push(json!({
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "content": [{"type": "text", "text": tool_content.to_display_string()}],
                                    "is_error": is_error
                                }));
                            }
                            _ => {} // Skip other content types for now
                        }
                    }

                    if !content.is_empty() {
                        request_messages.push(json!({
                            "role": match message.role {
                                MessageRole::User => "user",
                                MessageRole::Assistant => "assistant",
                                _ => unreachable!()
                            },
                            "content": content
                        }));
                    }
                }
            }
        }

        // Build request
        let mut request = json!({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": config.max_tokens.unwrap_or(4096),
            "messages": request_messages
        });

        // Determine if we should add cache markers
        let enable_system_cache = matches!(
            config.cache_strategy,
            CacheStrategy::SystemOnly | CacheStrategy::SystemAndTools | CacheStrategy::Auto
        );
        let enable_tool_cache = matches!(
            config.cache_strategy,
            CacheStrategy::SystemAndTools | CacheStrategy::Auto
        );

        // Add system prompt if present
        // When caching is enabled, use array format with cache_control marker
        if let Some(system) = &system_prompt {
            if enable_system_cache {
                // Use array format with cache_control for prompt caching
                request["system"] = json!([{
                    "type": "text",
                    "text": system,
                    "cache_control": { "type": "ephemeral" }
                }]);
                debug!(
                    "[{}] 📋🗄️ System prompt with cache_control ({} chars): {}...",
                    operation_id,
                    system.len(),
                    &system.chars().take(200).collect::<String>()
                );
            } else {
                request["system"] = json!(system);
                debug!(
                    "[{}] 📋 System prompt being sent ({} chars): {}...",
                    operation_id,
                    system.len(),
                    &system.chars().take(200).collect::<String>()
                );
            }
        } else {
            debug!("[{}] ⚠️  NO SYSTEM PROMPT - this may cause unexpected behavior", operation_id);
        }

        // Add temperature if specified
        if let Some(temp) = config.temperature {
            request["temperature"] = json!(temp);
        }

        // Add tools if provided
        if !tools.is_empty() {
            let tool_count = tools.len();
            let claude_tools: Vec<Value> = tools
                .iter()
                .enumerate()
                .map(|(idx, tool)| {
                    let mut tool_json = json!({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.input_schema
                    });

                    // Add cache_control to the LAST tool when tool caching is enabled
                    // This caches all tools up to and including this point
                    if enable_tool_cache && idx == tool_count - 1 {
                        tool_json["cache_control"] = json!({ "type": "ephemeral" });
                        debug!("[{}] 🗄️ Added cache_control to tool definitions", operation_id);
                    }

                    tool_json
                })
                .collect();

            request["tools"] = json!(claude_tools);

            // Apply tool_choice from config (Claude format)
            use crate::types::tools::ToolChoice;
            match &config.tool_choice {
                ToolChoice::Auto => {
                    request["tool_choice"] = json!({"type": "auto"});
                }
                ToolChoice::Any => {
                    request["tool_choice"] = json!({"type": "any"});
                }
                ToolChoice::Tool { name } => {
                    request["tool_choice"] = json!({"type": "tool", "name": name});
                }
                // ToolChoice::None is handled at the event_loop level (empty tools list),
                // so if we reach here with None, fall back to auto for safety
                ToolChoice::None => {
                    request["tool_choice"] = json!({"type": "auto"});
                }
            }
        }

        debug!(
            "[{}] 📤 Full request body (pretty formatted):\n{}",
            operation_id,
            serde_json::to_string_pretty(&request).unwrap_or_else(|_| "Invalid JSON".to_string())
        );

        serde_json::to_string(&request).map_err(|e| LlmError::SerializationError {
            message: format!("Failed to serialize Claude request: {}", e),
        })
    }

    /// Build Nova-specific request
    fn build_nova_request(
        &self,
        messages: &Messages,
        tools: &[Tool],
        config: &ChatConfig,
        operation_id: &str,
    ) -> Result<String, LlmError> {
        let mut request_messages = Vec::new();

        // Use system_prompt from Messages struct if available, otherwise extract from System role messages
        let mut system_prompt = messages.system_prompt.clone();

        // Process messages - Nova format is similar to Claude but with different structure
        for message in &messages.messages {
            match message.role {
                MessageRole::System => {
                    // Extract system prompt for Nova (fallback if not in Messages struct)
                    if system_prompt.is_none() {
                        let text = message
                            .content
                            .iter()
                            .filter_map(|block| match block {
                                ContentBlock::Text { text } => Some(text.as_str()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join(" ");
                        if !text.is_empty() {
                            system_prompt = Some(text);
                        }
                    }
                }
                MessageRole::User | MessageRole::Assistant => {
                    let mut content = Vec::new();

                    for block in &message.content {
                        match block {
                            ContentBlock::Text { text } => {
                                content.push(json!({
                                    "text": text
                                }));
                            }
                            ContentBlock::ToolUse { id, name, input } => {
                                // Nova tool use format - ensure input is never null or invalid
                                let safe_input = if input.is_null() || !input.is_object() {
                                    // If input is null or not an object, use empty object
                                    serde_json::Value::Object(serde_json::Map::new())
                                } else {
                                    input.clone()
                                };
                                content.push(json!({
                                    "toolUse": {
                                        "toolUseId": id,
                                        "name": name,
                                        "input": safe_input
                                    }
                                }));
                            }
                            ContentBlock::ToolResult {
                                tool_use_id,
                                content: tool_content,
                                is_error,
                            } => {
                                // Nova tool result format (if supported)
                                content.push(json!({
                                    "toolResult": {
                                        "toolUseId": tool_use_id,
                                        "content": [{"text": tool_content.to_display_string()}],
                                        "status": if *is_error { "error" } else { "success" }
                                    }
                                }));
                            }
                            _ => {} // Skip other content types for now
                        }
                    }

                    if !content.is_empty() {
                        request_messages.push(json!({
                            "role": match message.role {
                                MessageRole::User => "user",
                                MessageRole::Assistant => "assistant",
                                _ => unreachable!()
                            },
                            "content": content
                        }));
                    }
                }
            }
        }

        // Build Nova request structure based on Invoke API documentation
        let mut request = json!({
            "schemaVersion": "messages-v1",
            "messages": request_messages,
            "inferenceConfig": {
                "maxTokens": config.max_tokens.unwrap_or(2048), // Nova Micro default
            }
        });

        // Determine if we should add cache markers
        // Note: Nova only supports system prompt caching, NOT tool caching
        let enable_system_cache = matches!(
            config.cache_strategy,
            CacheStrategy::SystemOnly | CacheStrategy::SystemAndTools | CacheStrategy::Auto
        );

        // Log warning if user requested tool caching (not supported by Nova)
        if matches!(config.cache_strategy, CacheStrategy::SystemAndTools) && !tools.is_empty() {
            debug!(
                "[{}] ⚠️ Nova does not support tool caching. Only system prompt will be cached.",
                operation_id
            );
        }

        // Add system prompt if present (Nova Invoke API format)
        // When caching is enabled, add cachePoint marker
        if let Some(system) = &system_prompt {
            if enable_system_cache {
                // Nova uses cachePoint instead of cache_control
                request["system"] = json!([{
                    "text": system,
                    "cachePoint": { "type": "default" }
                }]);
                debug!(
                    "[{}] 📋🗄️ Nova system prompt with cachePoint ({} chars): {}...",
                    operation_id,
                    system.len(),
                    &system.chars().take(200).collect::<String>()
                );
            } else {
                request["system"] = json!([{"text": system}]);
                debug!(
                    "[{}] 📋 System prompt being sent to Nova ({} chars): {}...",
                    operation_id,
                    system.len(),
                    &system.chars().take(200).collect::<String>()
                );
            }
        } else {
            debug!("[{}] ⚠️  NO SYSTEM PROMPT for Nova - this may cause unexpected behavior", operation_id);
        }

        // Add temperature if specified
        if let Some(temp) = config.temperature {
            request["inferenceConfig"]["temperature"] = json!(temp);
        }

        // Add tools if provided (Nova tool format)
        // Note: Nova does NOT support tool caching - tools are sent without cache markers
        if !tools.is_empty() {
            let nova_tools: Vec<Value> = tools
                .iter()
                .map(|tool| {
                    json!({
                        "toolSpec": {
                            "name": tool.name,
                            "description": tool.description,
                            "inputSchema": {
                                "json": tool.input_schema
                            }
                        }
                    })
                })
                .collect();

            // Apply tool_choice from config (Nova format)
            use crate::types::tools::ToolChoice;
            let tool_choice_value = match &config.tool_choice {
                ToolChoice::Auto => json!({"auto": {}}),
                ToolChoice::Any => json!({"any": {}}),
                ToolChoice::Tool { name } => json!({"tool": {"name": name}}),
                // ToolChoice::None handled at event_loop level; fall back to auto if reached
                ToolChoice::None => json!({"auto": {}}),
            };

            request["toolConfig"] = json!({
                "tools": nova_tools,
                "toolChoice": tool_choice_value
            });
        }

        debug!(
            "[{}] 📤 Nova request body (pretty formatted):\n{}",
            operation_id,
            serde_json::to_string_pretty(&request).unwrap_or_else(|_| "Invalid JSON".to_string())
        );

        serde_json::to_string(&request).map_err(|e| LlmError::SerializationError {
            message: format!("Failed to serialize Nova request: {}", e),
        })
    }

    /// Build Mistral-specific request
    fn build_mistral_request(
        &self,
        messages: &Messages,
        tools: &[Tool],
        config: &ChatConfig,
        operation_id: &str,
    ) -> Result<String, LlmError> {
        let mut request_messages = Vec::new();

        // Process messages - Mistral includes system messages in the messages array
        for message in &messages.messages {
            match message.role {
                MessageRole::System => {
                    // System messages go in the messages array with role: "system"
                    let text = message
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    if !text.is_empty() {
                        request_messages.push(json!({
                            "role": "system",
                            "content": text
                        }));
                    }
                }
                MessageRole::User | MessageRole::Assistant => {
                    let mut content_parts = Vec::new();
                    let mut has_tool_calls = false;
                    let mut tool_calls = Vec::new();

                    for block in &message.content {
                        match block {
                            ContentBlock::Text { text } => {
                                content_parts.push(json!({
                                    "type": "text",
                                    "text": text
                                }));
                            }
                            ContentBlock::ToolUse { id, name, input } => {
                                has_tool_calls = true;
                                // Mistral uses OpenAI-style function calling format
                                let safe_input = if input.is_null() || !input.is_object() {
                                    serde_json::Value::Object(serde_json::Map::new())
                                } else {
                                    input.clone()
                                };

                                // Convert input object to JSON string for arguments
                                let arguments_str = serde_json::to_string(&safe_input)
                                    .unwrap_or_else(|_| "{}".to_string());

                                tool_calls.push(json!({
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
                                is_error,
                            } => {
                                // Mistral tool results come back as role: "tool" messages
                                // We'll handle these separately
                                request_messages.push(json!({
                                    "role": "tool",
                                    "tool_call_id": tool_use_id,
                                    "content": tool_content.to_display_string(),
                                    "is_error": is_error
                                }));
                            }
                            _ => {} // Skip other content types
                        }
                    }

                    // Build message based on role and content
                    if message.role == MessageRole::Assistant && has_tool_calls {
                        // Assistant message with tool calls
                        let mut msg = json!({
                            "role": "assistant",
                            "tool_calls": tool_calls
                        });
                        // Add content if there's text along with tool calls
                        if !content_parts.is_empty() {
                            msg["content"] = if content_parts.len() == 1
                                && content_parts[0]["type"] == "text" {
                                // Single text content can be a string
                                json!(content_parts[0]["text"].as_str().unwrap_or(""))
                            } else {
                                json!(content_parts)
                            };
                        }
                        request_messages.push(msg);
                    } else if !content_parts.is_empty() {
                        // Regular user/assistant message
                        request_messages.push(json!({
                            "role": match message.role {
                                MessageRole::User => "user",
                                MessageRole::Assistant => "assistant",
                                _ => unreachable!()
                            },
                            "content": if content_parts.len() == 1
                                && content_parts[0]["type"] == "text" {
                                // Single text content can be a string
                                json!(content_parts[0]["text"].as_str().unwrap_or(""))
                            } else {
                                json!(content_parts)
                            }
                        }));
                    }
                }
            }
        }

        // Build tools array in Mistral format (OpenAI-style)
        let mut mistral_tools = Vec::new();
        for tool in tools {
            mistral_tools.push(json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema
                }
            }));
        }

        // Build final request
        let mut request = json!({
            "messages": request_messages,
            "max_tokens": config.max_tokens.unwrap_or(8192)
        });

        // Add optional parameters
        if let Some(temp) = config.temperature {
            request["temperature"] = json!(temp);
        }

        // Add tools if provided
        if !mistral_tools.is_empty() {
            request["tools"] = json!(mistral_tools);

            // Apply tool_choice from config (OpenAI/Mistral format)
            use crate::types::tools::ToolChoice;
            match &config.tool_choice {
                ToolChoice::Auto => {
                    request["tool_choice"] = json!("auto");
                }
                ToolChoice::Any => {
                    request["tool_choice"] = json!("required");
                }
                ToolChoice::Tool { name } => {
                    request["tool_choice"] =
                        json!({"type": "function", "function": {"name": name}});
                }
                // ToolChoice::None handled at event_loop level; fall back to auto if reached
                ToolChoice::None => {
                    request["tool_choice"] = json!("auto");
                }
            }
        }

        debug!(
            "[{}] 📤 Mistral request body (pretty formatted):\n{}",
            operation_id,
            serde_json::to_string_pretty(&request).unwrap_or_else(|_| "Invalid JSON".to_string())
        );

        serde_json::to_string(&request).map_err(|e| LlmError::SerializationError {
            message: format!("Failed to serialize Mistral request: {}", e),
        })
    }

    /// Parse Claude response
    fn parse_claude_response(
        &self,
        response_body: &str,
        operation_id: &str,
    ) -> Result<ChatResponse, LlmError> {
        let response: Value =
            serde_json::from_str(response_body).map_err(|e| LlmError::SerializationError {
                message: format!("Failed to parse Claude response: {}", e),
            })?;

        debug!(
            "[{}] 📨 Full response body (pretty formatted):\n{}",
            operation_id,
            serde_json::to_string_pretty(&response).unwrap_or_else(|_| "Invalid JSON".to_string())
        );

        // Extract content
        let content = response["content"]
            .as_array()
            .unwrap_or(&Vec::new())
            .iter()
            .filter_map(|block| {
                if block["type"] == "text" {
                    block["text"].as_str().map(|s| s.to_string())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(" ");

        // Extract tool calls
        let tool_calls = response["content"]
            .as_array()
            .unwrap_or(&Vec::new())
            .iter()
            .filter_map(|block| {
                if block["type"] == "tool_use" {
                    Some(crate::llm::traits::ToolCall {
                        id: block["id"].as_str().unwrap_or("").to_string(),
                        name: block["name"].as_str().unwrap_or("").to_string(),
                        input: block["input"].clone(),
                    })
                } else {
                    None
                }
            })
            .collect();

        // Extract usage
        let usage = response["usage"]
            .as_object()
            .map(|usage| crate::llm::traits::Usage {
                input_tokens: usage["input_tokens"].as_u64().unwrap_or(0) as u32,
                output_tokens: usage["output_tokens"].as_u64().unwrap_or(0) as u32,
                total_tokens: (usage["input_tokens"].as_u64().unwrap_or(0)
                    + usage["output_tokens"].as_u64().unwrap_or(0))
                    as u32,
                cache_read_tokens: usage["cache_read_input_tokens"].as_u64().map(|t| t as u32),
                cache_write_tokens: usage["cache_creation_input_tokens"].as_u64().map(|t| t as u32),
            });

        // Create metadata
        let mut metadata = HashMap::new();
        metadata.insert("stop_reason".to_string(), response["stop_reason"].clone());
        metadata.insert("model".to_string(), json!(response["model"]));

        Ok(ChatResponse {
            content,
            tool_calls,
            thinking: None,
            usage,
            metadata,
        })
    }

    /// Parse Nova response
    fn parse_nova_response(
        &self,
        response_body: &str,
        operation_id: &str,
    ) -> Result<ChatResponse, LlmError> {
        let response: Value =
            serde_json::from_str(response_body).map_err(|e| LlmError::SerializationError {
                message: format!("Failed to parse Nova response: {}", e),
            })?;

        debug!(
            "[{}] 📨 Nova response body (pretty formatted):\n{}",
            operation_id,
            serde_json::to_string_pretty(&response).unwrap_or_else(|_| "Invalid JSON".to_string())
        );

        // Nova Invoke API response structure - extract from output wrapper
        let mut content = String::new();
        let mut tool_calls = Vec::new();

        // Navigate through Nova response structure: output.message.content
        if let Some(output) = response.get("output") {
            if let Some(message) = output.get("message") {
                // Nova message content structure
                if let Some(content_array) = message.get("content").and_then(|c| c.as_array()) {
                    for content_block in content_array {
                        // Nova doesn't specify type in content blocks for text
                        if let Some(text) = content_block.get("text").and_then(|t| t.as_str()) {
                            if !content.is_empty() {
                                content.push(' ');
                            }
                            content.push_str(text);
                        } else if let Some(tool_use) = content_block.get("toolUse") {
                            // Nova tool use format
                            let tool_call = crate::llm::traits::ToolCall {
                                id: tool_use
                                    .get("toolUseId")
                                    .and_then(|id| id.as_str())
                                    .unwrap_or("")
                                    .to_string(),
                                name: tool_use
                                    .get("name")
                                    .and_then(|n| n.as_str())
                                    .unwrap_or("")
                                    .to_string(),
                                input: tool_use
                                    .get("input")
                                    .cloned()
                                    .unwrap_or(serde_json::Value::Object(serde_json::Map::new())),
                            };
                            tool_calls.push(tool_call);
                        }
                    }
                }
            }

            // Extract usage information from top-level usage field
            let usage = response
                .get("usage")
                .and_then(|u| u.as_object())
                .map(|usage| crate::llm::traits::Usage {
                    input_tokens: usage
                        .get("inputTokens")
                        .and_then(|t| t.as_u64())
                        .unwrap_or(0) as u32,
                    output_tokens: usage
                        .get("outputTokens")
                        .and_then(|t| t.as_u64())
                        .unwrap_or(0) as u32,
                    total_tokens: usage
                        .get("totalTokens")
                        .and_then(|t| t.as_u64())
                        .unwrap_or(0) as u32,
                    // Nova uses different field names for cache metrics
                    cache_read_tokens: usage
                        .get("cacheReadInputTokenCount")
                        .and_then(|t| t.as_u64())
                        .map(|t| t as u32),
                    cache_write_tokens: usage
                        .get("cacheWriteInputTokenCount")
                        .and_then(|t| t.as_u64())
                        .map(|t| t as u32),
                });

            // Create metadata from top-level fields
            let mut metadata = HashMap::new();
            if let Some(stop_reason) = response.get("stopReason") {
                metadata.insert("stop_reason".to_string(), stop_reason.clone());
            }
            metadata.insert("model".to_string(), json!("amazon-nova-micro"));

            return Ok(ChatResponse {
                content,
                tool_calls,
                thinking: None,
                usage,
                metadata,
            });
        }

        // Fallback if no body found
        let mut metadata = HashMap::new();
        metadata.insert("model".to_string(), json!("amazon-nova-micro"));

        Ok(ChatResponse {
            content,
            tool_calls,
            thinking: None,
            usage: None,
            metadata,
        })
    }

    /// Parse Mistral response
    fn parse_mistral_response(
        &self,
        response_body: &str,
        operation_id: &str,
    ) -> Result<ChatResponse, LlmError> {
        let response: Value =
            serde_json::from_str(response_body).map_err(|e| LlmError::SerializationError {
                message: format!("Failed to parse Mistral response: {}", e),
            })?;

        debug!(
            "[{}] 📨 Mistral response body (pretty formatted):\n{}",
            operation_id,
            serde_json::to_string_pretty(&response).unwrap_or_else(|_| "Invalid JSON".to_string())
        );

        // Mistral response structure: outputs[0].message
        let mut content = String::new();
        let mut tool_calls = Vec::new();

        if let Some(outputs) = response.get("outputs").and_then(|o| o.as_array()) {
            if let Some(first_output) = outputs.first() {
                if let Some(message) = first_output.get("message") {
                    // Extract content from message.content
                    if let Some(msg_content) = message.get("content") {
                        if let Some(text) = msg_content.as_str() {
                            // Simple string content
                            content = text.to_string();
                        } else if let Some(content_array) = msg_content.as_array() {
                            // Array of content parts
                            for content_part in content_array {
                                if let Some(text) = content_part.get("text").and_then(|t| t.as_str()) {
                                    if !content.is_empty() {
                                        content.push(' ');
                                    }
                                    content.push_str(text);
                                }
                            }
                        }
                    }

                    // Extract tool calls from message.tool_calls
                    if let Some(tool_calls_array) = message.get("tool_calls").and_then(|tc| tc.as_array()) {
                        for tool_call in tool_calls_array {
                            if let Some(function) = tool_call.get("function") {
                                let name = function
                                    .get("name")
                                    .and_then(|n| n.as_str())
                                    .unwrap_or("")
                                    .to_string();

                                // Parse arguments JSON string to Value
                                let input = if let Some(args_str) = function.get("arguments").and_then(|a| a.as_str()) {
                                    serde_json::from_str(args_str)
                                        .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()))
                                } else {
                                    serde_json::Value::Object(serde_json::Map::new())
                                };

                                let id = tool_call
                                    .get("id")
                                    .and_then(|id| id.as_str())
                                    .unwrap_or("")
                                    .to_string();

                                tool_calls.push(crate::llm::traits::ToolCall {
                                    id,
                                    name,
                                    input,
                                });
                            }
                        }
                    }
                }
            }
        }

        // Extract usage information
        let usage = response
            .get("usage")
            .and_then(|u| u.as_object())
            .map(|usage| crate::llm::traits::Usage {
                input_tokens: usage
                    .get("prompt_tokens")
                    .and_then(|t| t.as_u64())
                    .unwrap_or(0) as u32,
                output_tokens: usage
                    .get("completion_tokens")
                    .and_then(|t| t.as_u64())
                    .unwrap_or(0) as u32,
                total_tokens: usage
                    .get("total_tokens")
                    .and_then(|t| t.as_u64())
                    .unwrap_or(0) as u32,
                // Mistral doesn't support prompt caching on Bedrock
                cache_read_tokens: None,
                cache_write_tokens: None,
            });

        // Create metadata
        let mut metadata = HashMap::new();
        if let Some(outputs) = response.get("outputs").and_then(|o| o.as_array()) {
            if let Some(first_output) = outputs.first() {
                if let Some(stop_reason) = first_output.get("stop_reason") {
                    metadata.insert("stop_reason".to_string(), stop_reason.clone());
                }
            }
        }
        metadata.insert("model".to_string(), json!("mistral-large"));

        Ok(ChatResponse {
            content,
            tool_calls,
            thinking: None,
            usage,
            metadata,
        })
    }

    /// Convert AWS Bedrock response stream to StreamEvent stream
    async fn convert_bedrock_stream_to_events(
        &self,
        response: aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamOutput,
        model_id: &str,
    ) -> Result<Box<dyn Stream<Item = StreamEvent> + Send + Unpin>, LlmError> {
        use futures::stream::StreamExt;

        let event_stream = response.body;

        let is_nova = model_id.contains("amazon.nova");
        let converted_stream = async_stream::stream! {
            tracing::debug!("🌊 Starting Bedrock stream processing for {} model...", if is_nova { "Nova" } else { "Claude" });
            let mut chunk_count = 0;
            let mut total_content = String::new();

            // AWS Bedrock streaming works with EventReceiver
            let mut stream = event_stream;

            loop {
                match stream.recv().await {
                    Ok(Some(event)) => {
                        chunk_count += 1;
                        tracing::trace!("🌊 Received Bedrock stream event #{}: {:?}", chunk_count, event);

                        match event {
                            aws_sdk_bedrockruntime::types::ResponseStream::Chunk(chunk) => {
                                // Parse the chunk bytes as JSON
                                let chunk_bytes = chunk.bytes().map(|b| b.as_ref()).unwrap_or(&[]);

                                if is_nova {
                                    // Nova streaming: decode base64 content from body.chunk.bytes
                                    tracing::trace!("🌊 Nova chunk bytes length: {}", chunk_bytes.len());

                                    if let Ok(chunk_str) = String::from_utf8(chunk_bytes.to_vec()) {
                                        if let Ok(chunk_json) = serde_json::from_str::<serde_json::Value>(&chunk_str) {
                                            // Nova chunks might be wrapped differently - need to handle base64 decoding
                                            if let Some(content_block_delta) = chunk_json.get("contentBlockDelta") {
                                                if let Some(delta) = content_block_delta.get("delta") {
                                                    if let Some(text) = delta.get("text").and_then(|t| t.as_str()) {
                                                        if !text.is_empty() {
                                                            total_content.push_str(text);
                                                            tracing::trace!("🌊 Nova content delta: '{}'", text);
                                                            yield StreamEvent::ContentDelta {
                                                                delta: text.to_string(),
                                                                index: 0,
                                                            };
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    // Claude streaming: direct JSON parsing
                                    let chunk_str = String::from_utf8_lossy(chunk_bytes);
                                    tracing::trace!("🌊 Claude chunk content: {}", chunk_str);

                                    if let Ok(chunk_json) = serde_json::from_str::<serde_json::Value>(&chunk_str) {
                                        // Extract delta content from Claude response format
                                        if let Some(delta) = chunk_json.get("delta") {
                                            if let Some(text) = delta.get("text").and_then(|t| t.as_str()) {
                                                if !text.is_empty() {
                                                    total_content.push_str(text);
                                                    tracing::trace!("🌊 Claude content delta: '{}'", text);
                                                    yield StreamEvent::ContentDelta {
                                                        delta: text.to_string(),
                                                        index: 0,
                                                    };
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            _ => {
                                tracing::trace!("🌊 Unhandled Bedrock stream event: {:?}", event);
                            }
                        }
                    }
                    Ok(None) => {
                        tracing::debug!("🌊 Bedrock stream ended");

                        // Estimate token usage based on content length (approximation)
                        // Typical ratio is ~4 characters per token for English text
                        let output_tokens = (total_content.len() / 4).max(1) as u32;
                        let input_tokens = 50; // Rough estimate for input - this could be improved

                        let usage = Some(crate::llm::traits::Usage {
                            input_tokens,
                            output_tokens,
                            total_tokens: input_tokens + output_tokens,
                            // Streaming doesn't provide cache metrics in chunk events
                            cache_read_tokens: None,
                            cache_write_tokens: None,
                        });

                        tracing::debug!("🌊 Estimated token usage: input={}, output={}, total={}",
                                      input_tokens, output_tokens, input_tokens + output_tokens);

                        yield StreamEvent::Done { usage };
                        break;
                    }
                    Err(e) => {
                        tracing::error!("❌ Bedrock stream error: {}", e);
                        yield StreamEvent::Error {
                            error: format!("Bedrock stream error: {}", e),
                        };
                        break;
                    }
                }
            }

            tracing::debug!("🌊 Bedrock stream completed - processed {} chunks, total content: {} chars",
                chunk_count, total_content.len());
        };

        Ok(Box::new(converted_stream.boxed()))
    }

    /// Convert AWS Bedrock response stream to StreamEvent stream with tool support
    async fn convert_bedrock_stream_to_events_with_tools(
        &self,
        response: aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamOutput,
        model_id: &str,
    ) -> Result<Box<dyn Stream<Item = StreamEvent> + Send + Unpin>, LlmError> {
        use futures::stream::StreamExt;

        tracing::info!(
            "🔧🌊 Starting convert_bedrock_stream_to_events_with_tools for model: {}",
            model_id
        );

        // Determine model type for model-aware streaming
        let model_type = if model_id.contains("anthropic.claude") {
            tracing::info!("🔧🌊 Detected Claude model type");
            ModelType::Claude
        } else if model_id.contains("amazon.nova") {
            tracing::info!("🔧🌊 Detected Nova model type");
            ModelType::Nova
        } else {
            tracing::error!(
                "🔧🌊 Unsupported model for streaming with tools: {}",
                model_id
            );
            return Err(LlmError::UnsupportedFeature {
                feature: format!("streaming with tools for model: {}", model_id),
                provider: ProviderType::Bedrock,
            });
        };

        let event_stream = response.body;

        let converted_stream = async_stream::stream! {
            tracing::debug!("🔧🌊 Starting Bedrock stream processing with tools for model type: {:?}", model_type);
            let mut chunk_count = 0;
            let mut total_content = String::new();
            let mut tool_state = ToolState::new(model_type.clone());

            // AWS Bedrock streaming works with EventReceiver
            let mut stream = event_stream;

            loop {
                match stream.recv().await {
                    Ok(Some(event)) => {
                        chunk_count += 1;
                        tracing::trace!("🔧🌊 Received Bedrock stream event #{}: {:?}", chunk_count, event);

                        match event {
                            aws_sdk_bedrockruntime::types::ResponseStream::Chunk(chunk) => {
                                // Parse the chunk bytes - model-aware processing
                                let chunk_bytes = chunk.bytes().map(|b| b.as_ref()).unwrap_or(&[]);

                                match tool_state.model_type {
                                    ModelType::Claude => {
                                        // Claude format: direct JSON
                                        let chunk_str = String::from_utf8_lossy(chunk_bytes);
                                        tracing::trace!("🔧🌊 Claude chunk content: {}", chunk_str);

                                        if let Ok(chunk_json) = serde_json::from_str::<serde_json::Value>(&chunk_str) {
                                            // Process Claude streaming events
                                            if let Some(events) = Self::process_claude_streaming_chunk(&chunk_json, &mut tool_state) {
                                                for event in events {
                                                    if let StreamEvent::ContentDelta { delta, .. } = &event {
                                                        total_content.push_str(delta);
                                                    }
                                                    yield event;
                                                }
                                            }
                                        }
                                    },
                                    ModelType::Nova => {
                                        // Nova format: debugging raw chunks first
                                        if let Some(events) = Self::process_nova_streaming_chunk(chunk_bytes, &mut tool_state) {
                                            for event in events {
                                                if let StreamEvent::ContentDelta { delta, .. } = &event {
                                                    total_content.push_str(delta);
                                                }
                                                yield event;
                                            }
                                        }
                                    }
                                }
                            }
                            _ => {
                                tracing::trace!("🔧🌊 Unhandled Bedrock stream event: {:?}", event);
                            }
                        }
                    }
                    Ok(None) => {
                        tracing::info!("🔧🌊 Bedrock stream with tools ended after {} chunks", chunk_count);

                        // Estimate token usage based on content length (approximation)
                        // Typical ratio is ~4 characters per token for English text
                        let output_tokens = (total_content.len() / 4).max(1) as u32;
                        let input_tokens = 100; // Rough estimate for input with tools - higher than non-tools

                        let usage = Some(crate::llm::traits::Usage {
                            input_tokens,
                            output_tokens,
                            total_tokens: input_tokens + output_tokens,
                            // Streaming doesn't provide cache metrics in chunk events
                            cache_read_tokens: None,
                            cache_write_tokens: None,
                        });

                        tracing::debug!("🔧🌊 Estimated token usage with tools: input={}, output={}, total={}",
                                      input_tokens, output_tokens, input_tokens + output_tokens);

                        yield StreamEvent::Done { usage };
                        break;
                    }
                    Err(e) => {
                        tracing::error!("❌ Bedrock stream with tools error after {} chunks: {}", chunk_count, e);
                        yield StreamEvent::Error {
                            error: format!("Bedrock stream with tools error: {}", e),
                        };
                        break;
                    }
                }
            }

            tracing::debug!("🔧🌊 Bedrock stream with tools completed - processed {} chunks, total content: {} chars",
                chunk_count, total_content.len());
        };

        Ok(Box::new(converted_stream.boxed()))
    }

    /// Classify Bedrock API errors for better user feedback (based on test_bedrock_credentials_direct)
    fn classify_bedrock_error(
        &self,
        sdk_error: &aws_sdk_bedrockruntime::error::SdkError<
            aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError,
        >,
        _model_id: &str,
    ) -> String {
        match sdk_error {
            aws_sdk_bedrockruntime::error::SdkError::ServiceError(context) => {
                let service_error = context.err();
                // Extract error type and message directly from AWS SDK
                let error_type = match service_error {
                    aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError::AccessDeniedException(_) => "AccessDeniedException",
                    aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError::ValidationException(_) => "ValidationException",
                    aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError::ResourceNotFoundException(_) => "ResourceNotFoundException",
                    aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError::ThrottlingException(_) => "ThrottlingException",
                    aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError::ServiceUnavailableException(_) => "ServiceUnavailableException",
                    aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError::ModelNotReadyException(_) => "ModelNotReadyException",
                    aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError::InternalServerException(_) => "InternalServerException",
                    _ => "UnknownServiceError",
                };

                // Get the message from the error
                let error_message = match service_error {
                    aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError::AccessDeniedException(e) => e.message(),
                    aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError::ValidationException(e) => e.message(),
                    aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError::ResourceNotFoundException(e) => e.message(),
                    aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError::ThrottlingException(e) => e.message(),
                    aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError::ServiceUnavailableException(e) => e.message(),
                    aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError::ModelNotReadyException(e) => e.message(),
                    aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError::InternalServerException(e) => e.message(),
                    _ => None,
                };

                if let Some(message) = error_message {
                    format!("🚨 {}: {}", error_type, message)
                } else {
                    // Handle common unhandled error types by parsing the service_error string
                    let service_error_str = format!("{}", service_error);
                    if service_error_str.contains("UnrecognizedClientException") {
                        "🚨 UnrecognizedClientException: Invalid or expired AWS credentials"
                            .to_string()
                    } else if service_error_str.contains("SignatureDoesNotMatch") {
                        "🚨 SignatureDoesNotMatch: AWS credential signature invalid".to_string()
                    } else if service_error_str.contains("TokenRefreshRequired") {
                        "🚨 TokenRefreshRequired: AWS session token expired".to_string()
                    } else {
                        format!("🚨 {}: {}", error_type, service_error)
                    }
                }
            }
            aws_sdk_bedrockruntime::error::SdkError::ConstructionFailure(e) => {
                format!("🔧 ConstructionFailure: {:?}", e)
            }
            aws_sdk_bedrockruntime::error::SdkError::DispatchFailure(e) => {
                format!("🌐 DispatchFailure: {:?}", e)
            }
            aws_sdk_bedrockruntime::error::SdkError::ResponseError(e) => {
                format!("📨 ResponseError: {:?}", e)
            }
            aws_sdk_bedrockruntime::error::SdkError::TimeoutError(e) => {
                format!("⏰ TimeoutError: {:?}", e)
            }
            _ => {
                format!("❓ Unknown SDK error: {}", sdk_error)
            }
        }
    }

    /// Classify Bedrock streaming API errors for better user feedback
    fn classify_bedrock_streaming_error(
        &self,
        sdk_error: &aws_sdk_bedrockruntime::error::SdkError<aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamError>,
        _model_id: &str,
    ) -> String {
        match sdk_error {
            aws_sdk_bedrockruntime::error::SdkError::ServiceError(context) => {
                let service_error = context.err();
                // Extract error type and message directly from AWS SDK
                let error_type = match service_error {
                    aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamError::AccessDeniedException(_) => "AccessDeniedException",
                    aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamError::ValidationException(_) => "ValidationException",
                    aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamError::ResourceNotFoundException(_) => "ResourceNotFoundException",
                    aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamError::ThrottlingException(_) => "ThrottlingException",
                    aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamError::ServiceUnavailableException(_) => "ServiceUnavailableException",
                    aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamError::ModelNotReadyException(_) => "ModelNotReadyException",
                    aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamError::InternalServerException(_) => "InternalServerException",
                    _ => "UnknownServiceError",
                };

                // Get the message from the error
                let error_message = match service_error {
                    aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamError::AccessDeniedException(e) => e.message(),
                    aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamError::ValidationException(e) => e.message(),
                    aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamError::ResourceNotFoundException(e) => e.message(),
                    aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamError::ThrottlingException(e) => e.message(),
                    aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamError::ServiceUnavailableException(e) => e.message(),
                    aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamError::ModelNotReadyException(e) => e.message(),
                    aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamError::InternalServerException(e) => e.message(),
                    _ => None,
                };

                if let Some(message) = error_message {
                    format!("🚨 {}: {}", error_type, message)
                } else {
                    // Handle common unhandled error types by parsing the service_error string
                    let service_error_str = format!("{}", service_error);
                    if service_error_str.contains("UnrecognizedClientException") {
                        "🚨 UnrecognizedClientException: Invalid or expired AWS credentials"
                            .to_string()
                    } else if service_error_str.contains("SignatureDoesNotMatch") {
                        "🚨 SignatureDoesNotMatch: AWS credential signature invalid".to_string()
                    } else if service_error_str.contains("TokenRefreshRequired") {
                        "🚨 TokenRefreshRequired: AWS session token expired".to_string()
                    } else {
                        format!("🚨 {}: {}", error_type, service_error)
                    }
                }
            }
            aws_sdk_bedrockruntime::error::SdkError::ConstructionFailure(e) => {
                format!("🔧 ConstructionFailure: {:?}", e)
            }
            aws_sdk_bedrockruntime::error::SdkError::DispatchFailure(e) => {
                format!("🌐 DispatchFailure: {:?}", e)
            }
            aws_sdk_bedrockruntime::error::SdkError::ResponseError(e) => {
                format!("📨 ResponseError: {:?}", e)
            }
            aws_sdk_bedrockruntime::error::SdkError::TimeoutError(e) => {
                format!("⏰ TimeoutError: {:?}", e)
            }
            _ => {
                format!("❓ Unknown SDK error: {}", sdk_error)
            }
        }
    }
}

#[async_trait]
impl LlmProvider for BedrockProvider {
    async fn chat(
        &self,
        model_id: &str,
        messages: &Messages,
        config: &ChatConfig,
    ) -> Result<ChatResponse, LlmError> {
        // Delegate to chat_with_tools with no tools
        self.chat_with_tools(model_id, messages, &[], config).await
    }

    async fn chat_with_tools(
        &self,
        model_id: &str,
        messages: &Messages,
        tools: &[Tool],
        config: &ChatConfig,
    ) -> Result<ChatResponse, LlmError> {
        let _guard = crate::perf_guard!("stood.bedrock.chat_with_tools");
        let operation_id = Uuid::new_v4();
        let start_time = Instant::now();

        // Model validation happens in build_request_body

        debug!(
            "[{}] 🗨️ Conversation structure: {} messages",
            operation_id,
            messages.messages.len()
        );
        for (i, msg) in messages.messages.iter().enumerate() {
            debug!(
                "[{}]   Message {}: role={:?}, content_blocks={}",
                operation_id,
                i + 1,
                msg.role,
                msg.content.len()
            );
        }

        debug!(
            "[{}] 🔧 Tools available: {} tools",
            operation_id,
            tools.len()
        );
        for (i, tool) in tools.iter().enumerate() {
            debug!(
                "[{}]   Tool {}: name={}, description={}",
                operation_id,
                i + 1,
                tool.name,
                tool.description
            );
        }

        debug!(
            "[{}] 🤖 Model: {}, max_tokens={:?}, temperature={:?}",
            operation_id, model_id, config.max_tokens, config.temperature
        );

        // Build request body
        let request_body = crate::perf_timed!("stood.bedrock.build_request_body", {
            self.build_request_body(messages, model_id, tools, config)?
        });

        // Store request JSON for raw capture
        self.store_request_json(&request_body);

        debug!(
            "[{}] Sending request to Bedrock model: {} with {} tools",
            operation_id,
            model_id,
            tools.len()
        );

        info!(
            "[{}] 🚀 Attempting Bedrock API call to model: {} (request size: {} bytes)",
            operation_id,
            model_id,
            request_body.len()
        );

        // Make API call with detailed error classification (network latency)
        let response = crate::perf_timed!("stood.bedrock.invoke_model", {
            self.client
                .invoke_model()
                .model_id(model_id)
                .body(aws_sdk_bedrockruntime::primitives::Blob::new(
                    request_body.as_bytes(),
                ))
                .send()
                .await
                .map_err(|e| {
                    let detailed_error = self.classify_bedrock_error(&e, model_id);
                    LlmError::ProviderError {
                        provider: ProviderType::Bedrock,
                        message: detailed_error,
                        source: Some(Box::new(e)),
                    }
                })
        })?;

        let duration = start_time.elapsed();
        info!(
            "[{}] ✅ Bedrock API call completed in {:.2}s",
            operation_id,
            duration.as_secs_f64()
        );

        // Parse response
        let response_body = String::from_utf8(response.body().as_ref().to_vec()).map_err(|e| {
            LlmError::SerializationError {
                message: format!("Invalid UTF-8 in response: {}", e),
            }
        })?;

        // Route to appropriate response parser based on model family
        crate::perf_timed!("stood.bedrock.parse_response", {
            if model_id.contains("amazon.nova") {
                self.parse_nova_response(&response_body, &operation_id.to_string())
            } else if model_id.contains("mistral.mistral") {
                self.parse_mistral_response(&response_body, &operation_id.to_string())
            } else {
                self.parse_claude_response(&response_body, &operation_id.to_string())
            }
        })
    }

    async fn chat_streaming(
        &self,
        model_id: &str,
        messages: &Messages,
        config: &ChatConfig,
    ) -> Result<Box<dyn Stream<Item = StreamEvent> + Send + Unpin>, LlmError> {
        tracing::info!(
            "🌊 Bedrock streaming request starting for model: {}",
            model_id
        );

        // Build request body using existing method (no tools for streaming)
        let request_body = self.build_request_body(messages, model_id, &[], config)?;

        // Store request JSON for raw capture
        self.store_request_json(&request_body);

        tracing::debug!("🌊 Bedrock streaming request body: {}", request_body);

        // Make streaming API call
        let response = self
            .client
            .invoke_model_with_response_stream()
            .model_id(model_id)
            .body(aws_sdk_bedrockruntime::primitives::Blob::new(
                request_body.as_bytes(),
            ))
            .send()
            .await
            .map_err(|e| {
                let detailed_error = self.classify_bedrock_streaming_error(&e, model_id);
                LlmError::ProviderError {
                    provider: ProviderType::Bedrock,
                    message: detailed_error,
                    source: Some(Box::new(e)),
                }
            })?;

        // Convert AWS Bedrock stream to our StreamEvent stream
        let stream = self
            .convert_bedrock_stream_to_events(response, model_id)
            .await?;
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
            "🔧🌊 Bedrock streaming with tools request starting for model: {} with {} tools",
            model_id,
            tools.len()
        );

        // Build request body with tools (key difference from chat_streaming)
        let request_body = self.build_request_body(messages, model_id, tools, config)?;

        // Store request JSON for raw capture
        self.store_request_json(&request_body);

        tracing::debug!(
            "🔧🌊 Bedrock streaming with tools request body: {}",
            request_body
        );

        // Make streaming API call (same as regular streaming)
        let response = self
            .client
            .invoke_model_with_response_stream()
            .model_id(model_id)
            .body(aws_sdk_bedrockruntime::primitives::Blob::new(
                request_body.as_bytes(),
            ))
            .send()
            .await
            .map_err(|e| {
                let detailed_error = self.classify_bedrock_streaming_error(&e, model_id);
                LlmError::ProviderError {
                    provider: ProviderType::Bedrock,
                    message: detailed_error,
                    source: Some(Box::new(e)),
                }
            })?;

        // Convert AWS Bedrock stream to our StreamEvent stream (with tool support)
        let stream = self
            .convert_bedrock_stream_to_events_with_tools(response, model_id)
            .await?;
        Ok(stream)
    }

    async fn health_check(&self) -> Result<HealthStatus, LlmError> {
        // Use the existing BedrockClient health check
        // TODO: Implement direct health check
        Ok(HealthStatus {
            healthy: true,
            provider: ProviderType::Bedrock,
            latency_ms: Some(0),
            error: None,
        })
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supports_streaming: true, // Now implemented!
            supports_tools: true,
            supports_thinking: true, // Claude 4.5 supports extended thinking
            supports_vision: true,   // Claude 4.5 supports vision
            supports_prompt_caching: true, // Supported for Claude and Nova models
            supports_tool_caching: true,   // Claude supports tool caching (Nova does not)
            max_tokens: Some(200000),
            available_models: vec![
                "us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string(),
                "us.anthropic.claude-sonnet-4-5-20250929-v1:0".to_string(),
                "us.anthropic.claude-opus-4-5-20251101-v1:0".to_string(),
                "us.amazon.nova-lite-v1:0".to_string(),
                "us.amazon.nova-pro-v1:0".to_string(),
                "us.amazon.nova-micro-v1:0".to_string(),
                "us.amazon.nova-premier-v1:0".to_string(),
                "amazon.nova-premier-v1:0".to_string(),
                "us.amazon.nova-2-lite-v1:0".to_string(),
                "us.amazon.nova-2-pro-v1:0".to_string(),
                "mistral.mistral-large-2407-v1:0".to_string(),
                "mistral.mistral-large-3-675b-instruct".to_string(),
            ],
        }
    }

    fn provider_type(&self) -> ProviderType {
        ProviderType::Bedrock
    }

    fn supported_models(&self) -> Vec<&'static str> {
        vec![
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "us.anthropic.claude-opus-4-5-20251101-v1:0",
            "us.amazon.nova-lite-v1:0",
            "us.amazon.nova-pro-v1:0",
            "us.amazon.nova-micro-v1:0",
            "us.amazon.nova-premier-v1:0",
            "amazon.nova-premier-v1:0",
            "us.amazon.nova-2-lite-v1:0",
            "us.amazon.nova-2-pro-v1:0",
            "mistral.mistral-large-2407-v1:0",
            "mistral.mistral-large-3-675b-instruct",
        ]
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl BedrockProvider {
    /// Process Claude streaming chunk with tool support
    fn process_claude_streaming_chunk(
        chunk_json: &serde_json::Value,
        tool_state: &mut ToolState,
    ) -> Option<Vec<StreamEvent>> {
        let mut events = Vec::new();

        // Handle both direct event types and nested event types
        let event_type = chunk_json.get("type").and_then(|t| t.as_str()).or_else(|| {
            // Nova format detection by checking for specific keys
            if chunk_json.get("messageStart").is_some() {
                Some("message_start")
            } else if chunk_json.get("contentBlockStart").is_some() {
                Some("content_block_start")
            } else if chunk_json.get("contentBlockDelta").is_some() {
                Some("content_block_delta")
            } else if chunk_json.get("contentBlockStop").is_some() {
                Some("content_block_stop")
            } else {
                None
            }
        });

        if let Some(event_type) = event_type {
            match event_type {
                "content_block_start" => {
                    // Handle Claude format: content_block.type == "tool_use"
                    let claude_tool_use =
                        chunk_json.get("content_block").and_then(|content_block| {
                            let block_type = content_block
                                .get("type")
                                .and_then(|t| t.as_str())
                                .unwrap_or("text");

                            if block_type == "tool_use" {
                                Some((
                                    content_block
                                        .get("id")
                                        .and_then(|id| id.as_str())
                                        .unwrap_or(""),
                                    content_block
                                        .get("name")
                                        .and_then(|n| n.as_str())
                                        .unwrap_or(""),
                                ))
                            } else {
                                None
                            }
                        });

                    // Handle Nova format: contentBlockStart.start.toolUse
                    let nova_tool_use = chunk_json
                        .get("contentBlockStart")
                        .and_then(|cbs| cbs.get("start"))
                        .and_then(|start| start.get("toolUse"))
                        .map(|tool_use| {
                            (
                                tool_use
                                    .get("toolUseId")
                                    .and_then(|id| id.as_str())
                                    .unwrap_or(""),
                                tool_use.get("name").and_then(|n| n.as_str()).unwrap_or(""),
                            )
                        });

                    if let Some((tool_use_id, name)) = claude_tool_use.or(nova_tool_use) {
                        let tool_use_id = tool_use_id.to_string();
                        let name = name.to_string();

                        tool_state.current_tool_call = Some(crate::llm::traits::ToolCall {
                            id: tool_use_id.clone(),
                            name: name.clone(),
                            input: serde_json::Value::Null,
                        });
                        tool_state.tool_input_buffer.clear();

                        // Don't emit ToolCallStart yet - wait until we have the complete input
                        // For now, just track it in tool_state
                    }
                }
                "content_block_delta" | "contentBlockDelta" => {
                    // Handle both Claude format (content_block_delta) and Nova format (contentBlockDelta)
                    let delta = chunk_json.get("delta").or_else(|| {
                        chunk_json
                            .get("contentBlockDelta")
                            .and_then(|cbd| cbd.get("delta"))
                    });

                    if let Some(delta) = delta {
                        if let Some(text) = delta.get("text").and_then(|t| t.as_str()) {
                            if !text.is_empty() {
                                events.push(StreamEvent::ContentDelta {
                                    delta: text.to_string(),
                                    index: 0,
                                });
                            }
                        } else if let Some(partial_json) =
                            delta.get("partial_json").and_then(|j| j.as_str())
                        {
                            // Claude format: partial_json field
                            if let Some(ref tool_call) = tool_state.current_tool_call {
                                tool_state.tool_input_buffer.push_str(partial_json);
                                events.push(StreamEvent::ToolCallDelta {
                                    tool_call_id: tool_call.id.clone(),
                                    delta: partial_json.to_string(),
                                });
                            }
                        } else if let Some(tool_use) = delta.get("toolUse") {
                            // Nova format: toolUse.input field
                            // Nova can send input as either a JSON object directly or as a string to be parsed
                            if let Some(ref mut tool_call) = tool_state.current_tool_call {
                                if let Some(input_obj) = tool_use.get("input") {
                                    if input_obj.is_object() || input_obj.is_array() {
                                        // Input is already a JSON object/array
                                        tool_call.input = input_obj.clone();
                                        let input_str =
                                            serde_json::to_string(input_obj).unwrap_or_default();
                                        events.push(StreamEvent::ToolCallDelta {
                                            tool_call_id: tool_call.id.clone(),
                                            delta: input_str,
                                        });
                                    } else if let Some(input_str) = input_obj.as_str() {
                                        // Input is a string that needs to be accumulated
                                        tool_state.tool_input_buffer.push_str(input_str);

                                        // Try to parse complete JSON input
                                        if let Ok(input_json) =
                                            serde_json::from_str::<serde_json::Value>(
                                                &tool_state.tool_input_buffer,
                                            )
                                        {
                                            tool_call.input = input_json;
                                        }
                                        events.push(StreamEvent::ToolCallDelta {
                                            tool_call_id: tool_call.id.clone(),
                                            delta: input_str.to_string(),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
                "content_block_stop" | "contentBlockStop" => {
                    if let Some(ref mut tool_call) = tool_state.current_tool_call {
                        // Final attempt to parse any remaining buffered input
                        if tool_call.input.is_null() && !tool_state.tool_input_buffer.is_empty() {
                            if let Ok(input_json) = serde_json::from_str::<serde_json::Value>(
                                &tool_state.tool_input_buffer,
                            ) {
                                tool_call.input = input_json;
                            } else {
                                // If parsing fails, use empty object to prevent ValidationException
                                tool_call.input = serde_json::Value::Object(serde_json::Map::new());
                            }
                        }

                        // Emit the complete tool call now that we have all the input
                        events.push(StreamEvent::ToolCallStart {
                            tool_call: tool_call.clone(),
                        });

                        // Also emit a delta with the complete input for compatibility
                        if !tool_call.input.is_null() {
                            let input_str =
                                serde_json::to_string(&tool_call.input).unwrap_or_default();
                            events.push(StreamEvent::ToolCallDelta {
                                tool_call_id: tool_call.id.clone(),
                                delta: input_str,
                            });
                        }
                    }
                    tool_state.reset();
                }
                "message_start" | "message_delta" | "message_stop" => {
                    // Handle message-level events (Nova and Claude)
                }
                _ => {}
            }
        }

        if events.is_empty() {
            None
        } else {
            Some(events)
        }
    }

    /// Process Nova streaming chunk with tool support
    /// Note: Nova uses the same streaming format as Claude!
    fn process_nova_streaming_chunk(
        chunk_bytes: &[u8],
        tool_state: &mut ToolState,
    ) -> Option<Vec<StreamEvent>> {
        // Nova actually uses Claude's streaming format, so we can reuse Claude's processor
        let chunk_str = String::from_utf8_lossy(chunk_bytes).into_owned();

        if let Ok(chunk_json) = serde_json::from_str::<serde_json::Value>(&chunk_str) {
            // Nova uses the exact same format as Claude, so delegate to Claude processor
            return Self::process_claude_streaming_chunk(&chunk_json, tool_state);
        } else {
            // Try to handle as plain text if it's not JSON
            if !chunk_str.trim().is_empty() {
                return Some(vec![StreamEvent::ContentDelta {
                    delta: chunk_str,
                    index: 0,
                }]);
            }
        }

        None
    }
}
