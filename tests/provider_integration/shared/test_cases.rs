//! Generic test case implementations for LLM Client verification
//!
//! These test cases are provider-agnostic and can be run across all
//! LLM providers to ensure consistent behavior.

use super::*;
use crate::agent::Agent;
// Model strings used in tests
const HAIKU: &str = "us.anthropic.claude-haiku-4-5-20251001-v1:0";
const NOVA_MICRO: &str = "us.amazon.nova-micro-v1:0";
const LM_GEMMA3_12B: &str = "google/gemma-3-12b";
use crate::tools::builtin::{CalculatorTool, CurrentTimeTool};
use std::time::Instant;

// =============================================================================
// MILESTONE 1: Core Provider Functionality
// =============================================================================

/// Test basic chat functionality
pub struct BasicChatTest;

#[async_trait::async_trait]
impl VerificationTest for BasicChatTest {
    fn test_name(&self) -> &'static str {
        "basic_chat"
    }
    fn description(&self) -> &'static str {
        "Verify basic chat functionality works"
    }
    fn category(&self) -> TestCategory {
        TestCategory::Core
    }
    fn required_features(&self) -> Vec<ProviderFeature> {
        vec![ProviderFeature::BasicChat]
    }

    async fn execute(&self, config: &TestConfig) -> VerificationResult {
        let start = Instant::now();
        let mut metadata = std::collections::HashMap::new();

        let result = async {
            // Create agent based on provider
            let mut agent = match config.provider {
                ProviderType::LmStudio => {
                    Agent::builder()
                        .provider("lm_studio").model(LM_GEMMA3_12B)
                        .system_prompt("You are a helpful assistant. Keep responses brief.")
                        .build()
                        .await?
                }
                ProviderType::Bedrock => {
                    Agent::builder()
                        .provider("bedrock").model(HAIKU)
                        .system_prompt("You are a helpful assistant. Keep responses brief.")
                        .build()
                        .await?
                }
                ProviderType::Anthropic => {
                    Agent::builder()
                        .provider("anthropic").model("claude-haiku-4-5-20251001")
                        .system_prompt("You are a helpful assistant. Keep responses brief.")
                        .build()
                        .await?
                }
                _ => return Err("Unsupported provider".into()),
            };

            // Execute simple chat
            let response = agent
                .execute("What is 2+2? Answer with just the number.")
                .await?;

            // Verify response
            if response.response.is_empty() {
                return Err("Empty response".into());
            }

            if !response.response.contains("4") {
                return Err(format!("Expected '4' in response, got: {}", response.response).into());
            }

            metadata.insert(
                "response_length".to_string(),
                serde_json::Value::Number(response.response.len().into()),
            );
            metadata.insert(
                "response_content".to_string(),
                serde_json::Value::String(response.response.clone()),
            );

            Ok(())
        }
        .await;

        VerificationResult {
            test_name: self.test_name().to_string(),
            provider: config.provider,
            model_id: config.model_id.clone(),
            success: result.is_ok(),
            duration: start.elapsed(),
            error: result
                .err()
                .map(|e: Box<dyn std::error::Error>| e.to_string()),
            metadata,
        }
    }
}

/// Test multi-turn conversation
pub struct MultiTurnConversationTest;

#[async_trait::async_trait]
impl VerificationTest for MultiTurnConversationTest {
    fn test_name(&self) -> &'static str {
        "multi_turn_conversation"
    }
    fn description(&self) -> &'static str {
        "Verify conversation context is maintained across turns"
    }
    fn category(&self) -> TestCategory {
        TestCategory::Core
    }
    fn required_features(&self) -> Vec<ProviderFeature> {
        vec![ProviderFeature::BasicChat]
    }

    async fn execute(&self, config: &TestConfig) -> VerificationResult {
        let start = Instant::now();
        let mut metadata = std::collections::HashMap::new();

        let result = async {
            let mut agent = match config.provider {
                ProviderType::LmStudio => {
                    Agent::builder()
                        .provider("lm_studio").model(LM_GEMMA3_12B)
                        .system_prompt("You are a helpful assistant.")
                        .build()
                        .await?
                }
                ProviderType::Bedrock => {
                    Agent::builder()
                        .provider("bedrock").model(HAIKU)
                        .system_prompt("You are a helpful assistant.")
                        .build()
                        .await?
                }
                ProviderType::Anthropic => {
                    Agent::builder()
                        .provider("anthropic").model("claude-haiku-4-5-20251001")
                        .system_prompt("You are a helpful assistant.")
                        .build()
                        .await?
                }
                _ => return Err("Unsupported provider".into()),
            };

            // First message: establish context
            let response1 = agent.execute("My favorite number is 42.").await?;
            metadata.insert(
                "response1".to_string(),
                serde_json::Value::String(response1.response.clone()),
            );

            // Second message: reference context
            let response2 = agent.execute("What number did I just mention?").await?;
            metadata.insert(
                "response2".to_string(),
                serde_json::Value::String(response2.response.clone()),
            );

            // Verify context was maintained
            if !response2.response.contains("42") {
                return Err(format!(
                    "Context not maintained. Expected '42' in: {}",
                    response2.response
                )
                .into());
            }

            // Check conversation length
            let conversation_count = agent.get_performance_summary().conversation_length;
            metadata.insert(
                "message_count".to_string(),
                serde_json::Value::Number(conversation_count.into()),
            );

            if conversation_count < 4 {
                // system + user1 + assistant1 + user2 + assistant2
                return Err(
                    format!("Expected at least 4 messages, got {}", conversation_count).into(),
                );
            }

            Ok(())
        }
        .await;

        VerificationResult {
            test_name: self.test_name().to_string(),
            provider: config.provider,
            model_id: config.model_id.clone(),
            success: result.is_ok(),
            duration: start.elapsed(),
            error: result
                .err()
                .map(|e: Box<dyn std::error::Error>| e.to_string()),
            metadata,
        }
    }
}

// =============================================================================
// MILESTONE 2: Tool System Integration
// =============================================================================

/// Test basic tool calling
pub struct BasicToolCallingTest;

#[async_trait::async_trait]
impl VerificationTest for BasicToolCallingTest {
    fn test_name(&self) -> &'static str {
        "basic_tool_calling"
    }
    fn description(&self) -> &'static str {
        "Verify basic tool calling functionality"
    }
    fn category(&self) -> TestCategory {
        TestCategory::Tools
    }
    fn required_features(&self) -> Vec<ProviderFeature> {
        vec![ProviderFeature::ToolCalling]
    }

    async fn execute(&self, config: &TestConfig) -> VerificationResult {
        let start = Instant::now();
        let mut metadata = std::collections::HashMap::new();

        let result = async {
            let mut agent = match config.provider {
                ProviderType::LmStudio => {
                    Agent::builder()
                        .provider("lm_studio").model(LM_GEMMA3_12B)
                        .system_prompt("You are a helpful assistant with access to tools. Use the calculator tool for math problems.")
                        .tool(Box::new(CalculatorTool))
                        .build()
                        .await?
                }
                ProviderType::Bedrock => {
                    Agent::builder()
                        .provider("bedrock").model(HAIKU)
                        .system_prompt("You are a helpful assistant with access to tools. Use the calculator tool for math problems.")
                        .tool(Box::new(CalculatorTool))
                        .build()
                        .await?
                }
                ProviderType::Anthropic => {
                    Agent::builder()
                        .provider("anthropic").model("claude-haiku-4-5-20251001")
                        .system_prompt("You are a helpful assistant with access to tools. Use the calculator tool for math problems.")
                        .tool(Box::new(CalculatorTool))
                        .build()
                        .await?
                }
                _ => return Err("Unsupported provider".into()),
            };

            // Execute a request that should use the calculator tool
            let response = agent.execute("What is 23 * 47? Use the calculator tool.").await?;

            metadata.insert("response".to_string(), serde_json::Value::String(response.response.clone()));
            metadata.insert("tools_called_count".to_string(), serde_json::Value::Number(response.tools_called.len().into()));
            metadata.insert("used_tools".to_string(), serde_json::Value::Bool(response.used_tools));

            // Verify tool was used
            if !response.used_tools || response.tools_called.is_empty() {
                return Err("No tools were used".into());
            }

            // Verify calculator tool was used
            let calculator_used = response.tools_called.iter()
                .any(|tool_name| tool_name.contains("calculator") || tool_name.contains("calc"));

            if !calculator_used {
                return Err(format!("Calculator tool not used. Tools used: {:?}", response.tools_called).into());
            }

            // Verify result contains the correct answer (23 * 47 = 1081)
            if !response.response.contains("1081") {
                return Err(format!("Expected '1081' in response: {}", response.response).into());
            }

            Ok(())
        }.await;

        VerificationResult {
            test_name: self.test_name().to_string(),
            provider: config.provider,
            model_id: config.model_id.clone(),
            success: result.is_ok(),
            duration: start.elapsed(),
            error: result
                .err()
                .map(|e: Box<dyn std::error::Error>| e.to_string()),
            metadata,
        }
    }
}

/// Test multiple tool usage
pub struct MultipleToolsTest;

#[async_trait::async_trait]
impl VerificationTest for MultipleToolsTest {
    fn test_name(&self) -> &'static str {
        "multiple_tools"
    }
    fn description(&self) -> &'static str {
        "Verify agent can use multiple different tools"
    }
    fn category(&self) -> TestCategory {
        TestCategory::Tools
    }
    fn required_features(&self) -> Vec<ProviderFeature> {
        vec![ProviderFeature::ToolCalling]
    }

    async fn execute(&self, config: &TestConfig) -> VerificationResult {
        let start = Instant::now();
        let mut metadata = std::collections::HashMap::new();

        let result = async {
            let mut agent = match config.provider {
                ProviderType::LmStudio => {
                    Agent::builder()
                        .provider("lm_studio").model(LM_GEMMA3_12B)
                        .system_prompt("You are a helpful assistant with access to tools.")
                        .tool(Box::new(CalculatorTool))
                        .tool(Box::new(CurrentTimeTool))
                        .build()
                        .await?
                }
                ProviderType::Bedrock => {
                    Agent::builder()
                        .provider("bedrock").model(HAIKU)
                        .system_prompt("You are a helpful assistant with access to tools.")
                        .tool(Box::new(CalculatorTool))
                        .tool(Box::new(CurrentTimeTool))
                        .build()
                        .await?
                }
                ProviderType::Anthropic => {
                    Agent::builder()
                        .provider("anthropic").model("claude-haiku-4-5-20251001")
                        .system_prompt("You are a helpful assistant with access to tools.")
                        .tool(Box::new(CalculatorTool))
                        .tool(Box::new(CurrentTimeTool))
                        .build()
                        .await?
                }
                _ => return Err("Unsupported provider".into()),
            };

            // Request that should use both tools
            let response = agent.execute(
                "Please tell me what time it is, then calculate 12 * 34. Use the appropriate tools."
            ).await?;

            metadata.insert(
                "response".to_string(),
                serde_json::Value::String(response.response.clone()),
            );
            metadata.insert(
                "tools_called_count".to_string(),
                serde_json::Value::Number(response.tools_called.len().into()),
            );
            metadata.insert(
                "used_tools".to_string(),
                serde_json::Value::Bool(response.used_tools),
            );

            // Verify multiple tools were used
            if response.tools_called.len() < 2 {
                return Err(format!(
                    "Expected at least 2 tool calls, got {}",
                    response.tools_called.len()
                )
                .into());
            }

            // Verify different tools were used
            let tool_names: std::collections::HashSet<_> = response.tools_called.iter().collect();

            if tool_names.len() < 2 {
                return Err("Expected different tools to be used".into());
            }

            metadata.insert(
                "unique_tools".to_string(),
                serde_json::Value::Array(
                    tool_names
                        .iter()
                        .map(|name| serde_json::Value::String((**name).clone()))
                        .collect(),
                ),
            );

            Ok(())
        }
        .await;

        VerificationResult {
            test_name: self.test_name().to_string(),
            provider: config.provider,
            model_id: config.model_id.clone(),
            success: result.is_ok(),
            duration: start.elapsed(),
            error: result
                .err()
                .map(|e: Box<dyn std::error::Error>| e.to_string()),
            metadata,
        }
    }
}

// =============================================================================
// MILESTONE 3: Streaming Features (only for providers that support it)
// =============================================================================

/// Test basic streaming functionality
pub struct BasicStreamingTest;

#[async_trait::async_trait]
impl VerificationTest for BasicStreamingTest {
    fn test_name(&self) -> &'static str {
        "basic_streaming"
    }
    fn description(&self) -> &'static str {
        "Verify streaming responses work correctly"
    }
    fn category(&self) -> TestCategory {
        TestCategory::Streaming
    }
    fn required_features(&self) -> Vec<ProviderFeature> {
        vec![ProviderFeature::Streaming]
    }

    async fn execute(&self, config: &TestConfig) -> VerificationResult {
        let start = Instant::now();
        let mut metadata = std::collections::HashMap::new();

        let result = async {
            use crate::agent::streaming::{AgentExecutionMode, StreamEvent};
            use tokio_stream::StreamExt;

            // Create agent with streaming enabled based on provider and model
            let mut agent = match (config.provider, config.model_id.as_str()) {
                (ProviderType::LmStudio, "gemma-3-12b") => {
                    Agent::builder()
                        .provider("lm_studio").model(LM_GEMMA3_12B)
                        .system_prompt("You are a helpful assistant. Keep responses brief.")
                        .execution_mode(AgentExecutionMode::Streaming)
                        .build()
                        .await?
                }
                (ProviderType::Bedrock, "claude-3-5-haiku") => {
                    Agent::builder()
                        .provider("bedrock").model(HAIKU)
                        .system_prompt("You are a helpful assistant. Keep responses brief.")
                        .execution_mode(AgentExecutionMode::Streaming)
                        .build()
                        .await?
                }
                (ProviderType::Bedrock, "amazon-nova-micro") => {
                    Agent::builder()
                        .provider("bedrock").model(NOVA_MICRO)
                        .system_prompt("You are a helpful assistant. Keep responses brief.")
                        .execution_mode(AgentExecutionMode::Streaming)
                        .build()
                        .await?
                }
                _ => {
                    return Err(format!(
                        "Streaming not supported for provider {:?} with model {}",
                        config.provider, config.model_id
                    )
                    .into())
                }
            };

            // Execute streaming request
            let mut stream = agent
                .execute_streaming("Count from 1 to 5, one number per sentence.")
                .await?;

            let mut events = Vec::new();
            let mut content_chunks = Vec::new();
            let mut final_response = String::new();

            // Collect all stream events
            while let Some(event) = stream.next().await {
                match event {
                    Ok(StreamEvent::Delta { content }) => {
                        content_chunks.push(content.clone());
                        final_response.push_str(&content);
                        events.push("Delta".to_string());
                    }
                    Ok(StreamEvent::Done { response }) => {
                        events.push("Done".to_string());
                        final_response = response.response;
                        break;
                    }
                    Ok(StreamEvent::Error { error }) => {
                        return Err(format!("Stream error: {}", error).into());
                    }
                    Err(e) => {
                        return Err(format!("Stream iteration error: {}", e).into());
                    }
                }
            }

            // Verify streaming worked correctly
            if events.is_empty() {
                return Err("No stream events received".into());
            }

            if !events.contains(&"Done".to_string()) {
                return Err("Stream did not complete with Done event".into());
            }

            if final_response.is_empty() {
                return Err("No final response content".into());
            }

            // Verify content makes sense (should contain numbers 1-5)
            let contains_numbers = ["1", "2", "3", "4", "5"]
                .iter()
                .all(|num| final_response.contains(num));

            if !contains_numbers {
                metadata.insert(
                    "warning".to_string(),
                    serde_json::Value::String(
                        "Response may not contain all requested numbers 1-5".to_string(),
                    ),
                );
            }

            metadata.insert(
                "stream_events_count".to_string(),
                serde_json::Value::Number(events.len().into()),
            );
            metadata.insert(
                "content_chunks_count".to_string(),
                serde_json::Value::Number(content_chunks.len().into()),
            );
            metadata.insert(
                "final_response_length".to_string(),
                serde_json::Value::Number(final_response.len().into()),
            );
            metadata.insert(
                "received_done_event".to_string(),
                serde_json::Value::Bool(events.contains(&"Done".to_string())),
            );
            metadata.insert(
                "final_response".to_string(),
                serde_json::Value::String(final_response.clone()),
            );

            Ok(())
        }
        .await;

        VerificationResult {
            test_name: self.test_name().to_string(),
            provider: config.provider,
            model_id: config.model_id.clone(),
            success: result.is_ok(),
            duration: start.elapsed(),
            error: result
                .err()
                .map(|e: Box<dyn std::error::Error>| e.to_string()),
            metadata,
        }
    }
}

// =============================================================================
// Helper functions to create test suites
// =============================================================================

/// Create the core functionality test suite (Milestone 1)
pub fn create_core_test_suite() -> VerificationSuite {
    VerificationSuite::new("Core Provider Functionality")
        .add_test(Box::new(BasicChatTest))
        .add_test(Box::new(MultiTurnConversationTest))
}

/// Create the tool system test suite (Milestone 2)
pub fn create_tools_test_suite() -> VerificationSuite {
    VerificationSuite::new("Tool System Integration")
        .add_test(Box::new(BasicToolCallingTest))
        .add_test(Box::new(MultipleToolsTest))
}

/// Test streaming with tools functionality
pub struct StreamingWithToolsTest;

#[async_trait::async_trait]
impl VerificationTest for StreamingWithToolsTest {
    fn test_name(&self) -> &'static str {
        "streaming_with_tools"
    }
    fn description(&self) -> &'static str {
        "Verify streaming responses work correctly with tool usage"
    }
    fn category(&self) -> TestCategory {
        TestCategory::Streaming
    }
    fn required_features(&self) -> Vec<ProviderFeature> {
        vec![ProviderFeature::Streaming, ProviderFeature::ToolCalling]
    }

    async fn execute(&self, config: &TestConfig) -> VerificationResult {
        let start = Instant::now();
        let mut metadata = std::collections::HashMap::new();

        let result = async {
            use crate::agent::streaming::{StreamEvent, AgentExecutionMode};
            use tokio_stream::StreamExt;

            // Create agent with streaming and tools enabled
            let mut agent = match (config.provider, config.model_id.as_str()) {
                (ProviderType::LmStudio, "gemma-3-12b") => {
                    Agent::builder()
                        .provider("lm_studio").model(LM_GEMMA3_12B)
                        .system_prompt("You are a helpful assistant with access to tools. Use the calculator tool for math problems.")
                        .tool(Box::new(CalculatorTool))
                        .execution_mode(AgentExecutionMode::Streaming)
                        .build()
                        .await?
                }
                (ProviderType::Bedrock, "claude-3-5-haiku") => {
                    Agent::builder()
                        .provider("bedrock").model(HAIKU)
                        .system_prompt("You are a helpful assistant with access to tools. Use the calculator tool for math problems.")
                        .tool(Box::new(CalculatorTool))
                        .execution_mode(AgentExecutionMode::Streaming)
                        .build()
                        .await?
                }
                (ProviderType::Bedrock, "amazon-nova-micro") => {
                    Agent::builder()
                        .provider("bedrock").model(NOVA_MICRO)
                        .system_prompt("You are a helpful assistant with access to tools. Use the calculator tool for math problems.")
                        .tool(Box::new(CalculatorTool))
                        .execution_mode(AgentExecutionMode::Streaming)
                        .build()
                        .await?
                }
                _ => return Err(format!("Streaming with tools not supported for provider {:?} with model {}", config.provider, config.model_id).into()),
            };

            // Execute streaming request that requires tool usage
            let mut stream = agent.execute_streaming("Calculate 17 * 29 using the calculator tool.").await?;

            let mut events = Vec::new();
            let mut tool_calls = Vec::new();
            let mut final_response = String::new();
            let mut used_tools = false;

            // Collect all stream events
            while let Some(event) = stream.next().await {
                match event {
                    Ok(StreamEvent::Delta { content }) => {
                        final_response.push_str(&content);
                        events.push("Delta".to_string());
                    }
                    Ok(StreamEvent::ToolCall { tool_name, .. }) => {
                        tool_calls.push(tool_name.clone());
                        events.push("ToolCall".to_string());
                        used_tools = true;
                    }
                    Ok(StreamEvent::ToolResult { tool_name, result }) => {
                        events.push("ToolResult".to_string());
                        // Verify calculator result (17 * 29 = 493)
                        if tool_name.contains("calculator") && result.contains("493") {
                            metadata.insert("correct_calculation".to_string(), serde_json::Value::Bool(true));
                        }
                    }
                    Ok(StreamEvent::Done { response }) => {
                        events.push("Done".to_string());
                        final_response = response.response;
                        used_tools = response.used_tools;
                        tool_calls = response.tools_called;
                        break;
                    }
                    Ok(StreamEvent::Error { error }) => {
                        return Err(format!("Stream error: {}", error).into());
                    }
                    Err(e) => {
                        return Err(format!("Stream iteration error: {}", e).into());
                    }
                }
            }

            // Verify streaming with tools worked correctly
            if events.is_empty() {
                return Err("No stream events received".into());
            }

            if !events.contains(&"Done".to_string()) {
                return Err("Stream did not complete with Done event".into());
            }

            if !used_tools || tool_calls.is_empty() {
                return Err("No tools were used in streaming response".into());
            }

            // Verify calculator tool was used
            let calculator_used = tool_calls.iter()
                .any(|tool_name| tool_name.contains("calculator") || tool_name.contains("calc"));

            if !calculator_used {
                return Err(format!("Calculator tool not used. Tools used: {:?}", tool_calls).into());
            }

            // Verify result contains the correct answer (17 * 29 = 493)
            if !final_response.contains("493") {
                return Err(format!("Expected '493' in response: {}", final_response).into());
            }

            metadata.insert("stream_events_count".to_string(), serde_json::Value::Number(events.len().into()));
            metadata.insert("tools_called_count".to_string(), serde_json::Value::Number(tool_calls.len().into()));
            metadata.insert("used_tools".to_string(), serde_json::Value::Bool(used_tools));
            metadata.insert("tool_events_received".to_string(), serde_json::Value::Bool(
                events.contains(&"ToolCall".to_string()) || events.contains(&"ToolResult".to_string())
            ));
            metadata.insert("final_response".to_string(), serde_json::Value::String(final_response.clone()));

            Ok(())
        }.await;

        VerificationResult {
            test_name: self.test_name().to_string(),
            provider: config.provider,
            model_id: config.model_id.clone(),
            success: result.is_ok(),
            duration: start.elapsed(),
            error: result
                .err()
                .map(|e: Box<dyn std::error::Error>| e.to_string()),
            metadata,
        }
    }
}

/// Create the streaming test suite (Milestone 3)
pub fn create_streaming_test_suite() -> VerificationSuite {
    VerificationSuite::new("Streaming and Real-time Features")
        .add_test(Box::new(BasicStreamingTest))
        .add_test(Box::new(StreamingWithToolsTest))
}

/// Create the token counting test suite (Telemetry)
pub fn create_token_counting_test_suite() -> VerificationSuite {
    use super::token_counting_tests::*;
    create_token_counting_test_suite()
}

/// Create a comprehensive test suite with all tests including token counting
pub fn create_comprehensive_test_suite() -> VerificationSuite {
    use super::token_counting_tests::*;

    VerificationSuite::new("Comprehensive LLM Client Verification")
        .add_test(Box::new(BasicChatTest))
        .add_test(Box::new(MultiTurnConversationTest))
        .add_test(Box::new(BasicToolCallingTest))
        .add_test(Box::new(MultipleToolsTest))
        .add_test(Box::new(BasicStreamingTest))
        .add_test(Box::new(StreamingWithToolsTest))
        .add_test(Box::new(StreamingTokenCountingTest))
        .add_test(Box::new(NonStreamingTokenCountingTest))
        .add_test(Box::new(StreamingTokenCountingWithToolsTest))
        .add_test(Box::new(TokenCountingConsistencyTest))
}
