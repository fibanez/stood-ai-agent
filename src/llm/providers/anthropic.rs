//! Anthropic Direct API provider implementation.
//!
//! This provider connects directly to Anthropic's API for Claude models
//! without going through AWS Bedrock.
//!
//! **Status: NOT YET IMPLEMENTED** - See README.md "TODO - Work in Progress" section
//! This is a placeholder implementation that returns appropriate errors.
//! Future implementation will support direct Claude API access.

use crate::llm::traits::{
    ChatConfig, ChatResponse, HealthStatus, LlmError, LlmProvider, ProviderCapabilities,
    ProviderType, StreamEvent, Tool,
};
use crate::types::Messages;
use async_trait::async_trait;
use futures::Stream;

/// Anthropic Direct API provider - NOT YET IMPLEMENTED
///
/// This provider connects directly to Anthropic's API for Claude models.
/// See README.md "🚧 Planned Providers (Not Yet Implemented)" section.
#[derive(Debug)]
#[allow(dead_code)] // Planned for future implementation
pub struct AnthropicProvider {
    #[allow(dead_code)] // Planned for future implementation
    api_key: String,
    #[allow(dead_code)] // Planned for future implementation
    base_url: String,
    #[allow(dead_code)] // Planned for future implementation
    client: reqwest::Client,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider
    pub async fn new(api_key: String, base_url: Option<String>) -> Result<Self, LlmError> {
        let client = reqwest::Client::new();
        let base_url = base_url.unwrap_or_else(|| "https://api.anthropic.com".to_string());

        Ok(Self {
            api_key,
            base_url,
            client,
        })
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    async fn chat(
        &self,
        model_id: &str,
        messages: &Messages,
        config: &ChatConfig,
    ) -> Result<ChatResponse, LlmError> {
        tracing::info!(
            "🔵 Anthropic Direct API chat request for model: {}",
            model_id
        );

        // Convert Messages to Anthropic API format
        let (anthropic_messages, system_message) = self.convert_messages_to_anthropic(messages)?;

        // Build request body
        let mut request_body = serde_json::json!({
            "model": model_id,
            "max_tokens": config.max_tokens.unwrap_or(4096),
            "messages": anthropic_messages
        });

        // Add system message if present
        if let Some(system) = system_message {
            request_body["system"] = serde_json::json!(system);
        }

        // Add temperature if specified
        if let Some(temp) = config.temperature {
            request_body["temperature"] = serde_json::json!(temp);
        }

        // Add additional parameters
        for (key, value) in &config.additional_params {
            request_body[key] = value.clone();
        }

        tracing::debug!(
            "🔵 Anthropic request body: {}",
            serde_json::to_string_pretty(&request_body).unwrap_or_default()
        );

        // Make HTTP request
        let response = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("Content-Type", "application/json")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LlmError::NetworkError {
                message: format!("Anthropic API request failed: {}", e),
                source: Some(Box::new(e)),
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LlmError::ProviderError {
                provider: ProviderType::Anthropic,
                message: format!("Anthropic API error {}: {}", status, error_text),
                source: None,
            });
        }

        // Parse response
        let response_text = response.text().await.map_err(|e| LlmError::NetworkError {
            message: format!("Failed to read Anthropic response: {}", e),
            source: Some(Box::new(e)),
        })?;

        tracing::debug!("🔵 Anthropic response: {}", response_text);

        let response_json: serde_json::Value =
            serde_json::from_str(&response_text).map_err(|e| LlmError::ProviderError {
                provider: ProviderType::Anthropic,
                message: format!("Failed to parse Anthropic JSON: {}", e),
                source: Some(Box::new(e)),
            })?;

        // Convert response to ChatResponse
        self.convert_anthropic_response_to_chat_response(response_json)
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
            "🔧 Anthropic chat with {} tools for model: {}",
            tools.len(),
            model_id
        );

        // Convert Messages to Anthropic API format
        let (anthropic_messages, system_message) = self.convert_messages_to_anthropic(messages)?;

        // Convert tools to Anthropic tool format
        let anthropic_tools: Vec<serde_json::Value> = tools
            .iter()
            .map(|tool| {
                serde_json::json!({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema
                })
            })
            .collect();

        // Build request body with tools
        let mut request_body = serde_json::json!({
            "model": model_id,
            "max_tokens": config.max_tokens.unwrap_or(4096),
            "messages": anthropic_messages,
            "tools": anthropic_tools
        });

        // Apply tool_choice from config (Anthropic format)
        use crate::types::tools::ToolChoice;
        match &config.tool_choice {
            ToolChoice::Auto => {
                request_body["tool_choice"] = serde_json::json!({"type": "auto"});
            }
            ToolChoice::Any => {
                request_body["tool_choice"] = serde_json::json!({"type": "any"});
            }
            ToolChoice::Tool { name } => {
                request_body["tool_choice"] =
                    serde_json::json!({"type": "tool", "name": name});
            }
            // ToolChoice::None handled at event_loop level (empty tools list),
            // but also set the API field to "none" if it somehow reaches here
            ToolChoice::None => {
                request_body["tool_choice"] = serde_json::json!({"type": "none"});
            }
        }

        // Add system message if present
        if let Some(system) = system_message {
            request_body["system"] = serde_json::json!(system);
        }

        // Add temperature if specified
        if let Some(temp) = config.temperature {
            request_body["temperature"] = serde_json::json!(temp);
        }

        // Add additional parameters
        for (key, value) in &config.additional_params {
            request_body[key] = value.clone();
        }

        tracing::debug!(
            "🔧 Anthropic tools request: {}",
            serde_json::to_string_pretty(&request_body).unwrap_or_default()
        );

        // Make HTTP request
        let response = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("Content-Type", "application/json")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LlmError::NetworkError {
                message: format!("Anthropic API tools request failed: {}", e),
                source: Some(Box::new(e)),
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(LlmError::ProviderError {
                provider: ProviderType::Anthropic,
                message: format!("Anthropic API tools error {}: {}", status, error_text),
                source: None,
            });
        }

        // Parse response
        let response_text = response.text().await.map_err(|e| LlmError::NetworkError {
            message: format!("Failed to read Anthropic tools response: {}", e),
            source: Some(Box::new(e)),
        })?;

        tracing::debug!("🔧 Anthropic tools response: {}", response_text);

        let response_json: serde_json::Value =
            serde_json::from_str(&response_text).map_err(|e| LlmError::ProviderError {
                provider: ProviderType::Anthropic,
                message: format!("Failed to parse Anthropic tools JSON: {}", e),
                source: Some(Box::new(e)),
            })?;

        // Convert response to ChatResponse
        self.convert_anthropic_response_to_chat_response(response_json)
    }

    async fn chat_streaming(
        &self,
        _model_id: &str,
        _messages: &Messages,
        _config: &ChatConfig,
    ) -> Result<Box<dyn Stream<Item = StreamEvent> + Send + Unpin>, LlmError> {
        Err(LlmError::UnsupportedFeature {
            feature: "Anthropic Direct API streaming not yet implemented".to_string(),
            provider: ProviderType::Anthropic,
        })
    }

    async fn chat_streaming_with_tools(
        &self,
        _model_id: &str,
        _messages: &Messages,
        _tools: &[Tool],
        _config: &ChatConfig,
    ) -> Result<Box<dyn Stream<Item = StreamEvent> + Send + Unpin>, LlmError> {
        Err(LlmError::UnsupportedFeature {
            feature: "Anthropic Direct API streaming with tools not yet implemented".to_string(),
            provider: ProviderType::Anthropic,
        })
    }

    async fn health_check(&self) -> Result<HealthStatus, LlmError> {
        // Test basic connectivity with a minimal request
        let start = std::time::Instant::now();

        let test_request = serde_json::json!({
            "model": "claude-3-5-haiku-20241022",
            "max_tokens": 1,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hi"}]
                }
            ]
        });

        let result = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("Content-Type", "application/json")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&test_request)
            .send()
            .await;

        let latency = start.elapsed().as_millis() as u64;

        match result {
            Ok(response) if response.status().is_success() => Ok(HealthStatus {
                healthy: true,
                provider: ProviderType::Anthropic,
                latency_ms: Some(latency),
                error: None,
            }),
            Ok(response) => {
                let status = response.status();
                let error_text = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "Unknown error".to_string());
                Ok(HealthStatus {
                    healthy: false,
                    provider: ProviderType::Anthropic,
                    latency_ms: Some(latency),
                    error: Some(format!("HTTP {}: {}", status, error_text)),
                })
            }
            Err(e) => Ok(HealthStatus {
                healthy: false,
                provider: ProviderType::Anthropic,
                latency_ms: None,
                error: Some(format!("Connection failed: {}", e)),
            }),
        }
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supports_streaming: false, // TODO: Implement streaming
            supports_tools: true, // Basic chat with tools implemented via convert_messages_to_anthropic
            supports_thinking: false, // TODO: Add thinking mode support
            supports_vision: false, // TODO: Add vision support
            supports_prompt_caching: true, // Anthropic API supports prompt caching
            supports_tool_caching: true,   // Claude supports tool caching
            max_tokens: Some(8192), // Anthropic allows up to 8192 output tokens
            available_models: vec![
                "claude-3-5-sonnet-20241022".to_string(),
                "claude-3-5-haiku-20241022".to_string(),
                "claude-3-opus-20240229".to_string(),
            ],
        }
    }

    fn provider_type(&self) -> ProviderType {
        ProviderType::Anthropic
    }

    fn supported_models(&self) -> Vec<&'static str> {
        vec![
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ]
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl AnthropicProvider {
    /// Convert Stood Messages format to Anthropic API format
    fn convert_messages_to_anthropic(
        &self,
        messages: &Messages,
    ) -> Result<(Vec<serde_json::Value>, Option<String>), LlmError> {
        use crate::types::{ContentBlock, MessageRole};

        let mut anthropic_messages = Vec::new();
        let mut system_message = None;

        for message in &messages.messages {
            match message.role {
                MessageRole::System => {
                    // Extract system message text
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
                        system_message = Some(text);
                    }
                }
                MessageRole::User | MessageRole::Assistant => {
                    let mut content = Vec::new();

                    for block in &message.content {
                        match block {
                            ContentBlock::Text { text } => {
                                content.push(serde_json::json!({
                                    "type": "text",
                                    "text": text
                                }));
                            }
                            ContentBlock::ToolUse { id, name, input } => {
                                content.push(serde_json::json!({
                                    "type": "tool_use",
                                    "id": id,
                                    "name": name,
                                    "input": input
                                }));
                            }
                            ContentBlock::ToolResult {
                                tool_use_id,
                                content: tool_content,
                                is_error,
                            } => {
                                content.push(serde_json::json!({
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "content": tool_content.to_display_string(),
                                    "is_error": is_error
                                }));
                            }
                            _ => {} // Skip other content types for now
                        }
                    }

                    if !content.is_empty() {
                        anthropic_messages.push(serde_json::json!({
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

        Ok((anthropic_messages, system_message))
    }

    /// Convert Anthropic API response to ChatResponse
    fn convert_anthropic_response_to_chat_response(
        &self,
        response: serde_json::Value,
    ) -> Result<ChatResponse, LlmError> {
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
                // Anthropic API uses same field names as Claude on Bedrock
                cache_read_tokens: usage["cache_read_input_tokens"].as_u64().map(|t| t as u32),
                cache_write_tokens: usage["cache_creation_input_tokens"].as_u64().map(|t| t as u32),
            });

        // Create metadata
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("stop_reason".to_string(), response["stop_reason"].clone());
        metadata.insert("model".to_string(), response["model"].clone());
        metadata.insert("id".to_string(), response["id"].clone());

        Ok(ChatResponse {
            content,
            tool_calls,
            thinking: None, // Anthropic doesn't have thinking mode in the API response yet
            usage,
            metadata,
        })
    }
}
