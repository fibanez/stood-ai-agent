//! Core functionality tests for LM Studio provider
//!
//! This module tests basic provider functionality including health checks,
//! model registration, basic chat, and agent integration.

use crate::agent::Agent;
use crate::llm::registry::PROVIDER_REGISTRY;
use crate::llm::traits::{LlmModel, ProviderType};
use crate::llm::string_model::StringModel;
use crate::verification::shared::*;

/// Test LM Studio provider health check
pub struct LMStudioHealthTest;

#[async_trait::async_trait]
impl VerificationTest for LMStudioHealthTest {
    fn test_name(&self) -> &'static str {
        "lm_studio_health_check"
    }
    fn description(&self) -> &'static str {
        "Verify LM Studio provider health check"
    }
    fn category(&self) -> TestCategory {
        TestCategory::Core
    }
    fn required_features(&self) -> Vec<ProviderFeature> {
        vec![]
    }

    async fn execute(&self, config: &TestConfig) -> VerificationResult {
        let start = std::time::Instant::now();
        let mut metadata = std::collections::HashMap::new();

        let result = async {
            // Test health check by creating an agent (this verifies provider works)
            let agent_result = Agent::builder()
                .provider("lm_studio")
                .model_str("google/gemma-3-12b")
                .system_prompt("Test")
                .build()
                .await;

            match agent_result {
                Ok(_) => {
                    // Agent creation succeeded, provider is healthy
                    metadata.insert("healthy".to_string(), serde_json::Value::Bool(true));
                    metadata.insert(
                        "provider".to_string(),
                        serde_json::Value::String("LmStudio".to_string()),
                    );
                    Ok(())
                }
                Err(e) => {
                    metadata.insert("healthy".to_string(), serde_json::Value::Bool(false));
                    metadata.insert(
                        "health_error".to_string(),
                        serde_json::Value::String(e.to_string()),
                    );
                    Err(format!("Health check failed: {}", e).into())
                }
            }
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

/// Test LM Studio model metadata
pub struct LMStudioModelMetadataTest;

#[async_trait::async_trait]
impl VerificationTest for LMStudioModelMetadataTest {
    fn test_name(&self) -> &'static str {
        "lm_studio_model_metadata"
    }
    fn description(&self) -> &'static str {
        "Verify LM Studio model metadata is correct"
    }
    fn category(&self) -> TestCategory {
        TestCategory::Core
    }
    fn required_features(&self) -> Vec<ProviderFeature> {
        vec![]
    }

    async fn execute(&self, config: &TestConfig) -> VerificationResult {
        let start = std::time::Instant::now();
        let mut metadata = std::collections::HashMap::new();

        let result = async {
            // Test Gemma 3 12B model metadata
            let model = StringModel::new("google/gemma-3-12b", ProviderType::LmStudio);

            // Verify provider type
            if model.provider() != ProviderType::LmStudio {
                return Err("Model provider type mismatch".into());
            }

            // Verify model ID
            let model_id = model.model_id();
            metadata.insert(
                "model_id".to_string(),
                serde_json::Value::String(model_id.to_string()),
            );

            if model_id.is_empty() {
                return Err("Model ID is empty".into());
            }

            // Verify context window
            let context_window = model.context_window();
            metadata.insert(
                "context_window".to_string(),
                serde_json::Value::Number(context_window.into()),
            );

            if context_window == 0 {
                return Err("Context window is zero".into());
            }

            // Verify max output tokens
            let max_output = model.max_output_tokens();
            metadata.insert(
                "max_output_tokens".to_string(),
                serde_json::Value::Number(max_output.into()),
            );

            if max_output == 0 {
                return Err("Max output tokens is zero".into());
            }

            // Verify capabilities
            let supports_tools = model.supports_tool_use();
            let supports_streaming = model.supports_streaming();

            metadata.insert(
                "supports_tools".to_string(),
                serde_json::Value::Bool(supports_tools),
            );
            metadata.insert(
                "supports_streaming".to_string(),
                serde_json::Value::Bool(supports_streaming),
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

/// Test basic chat directly with LM Studio provider
pub struct LMStudioDirectChatTest;

#[async_trait::async_trait]
impl VerificationTest for LMStudioDirectChatTest {
    fn test_name(&self) -> &'static str {
        "lm_studio_direct_chat"
    }
    fn description(&self) -> &'static str {
        "Test direct chat with LM Studio provider"
    }
    fn category(&self) -> TestCategory {
        TestCategory::Core
    }
    fn required_features(&self) -> Vec<ProviderFeature> {
        vec![ProviderFeature::BasicChat]
    }

    async fn execute(&self, config: &TestConfig) -> VerificationResult {
        let start = std::time::Instant::now();
        let mut metadata = std::collections::HashMap::new();

        let result = async {
            // Create agent with LM Studio model
            let mut agent = Agent::builder()
                .provider("lm_studio")
                .model_str("google/gemma-3-12b")
                .system_prompt("You are a helpful assistant. Keep responses brief.")
                .temperature(0.0)
                .max_tokens(50)
                .build()
                .await?;

            // Execute simple query
            let response = agent
                .execute("What is 2+2? Answer with just the number.")
                .await?;

            metadata.insert(
                "response_content".to_string(),
                serde_json::Value::String(response.response.clone()),
            );
            metadata.insert(
                "response_length".to_string(),
                serde_json::Value::Number(response.response.len().into()),
            );

            // Verify response
            if response.response.is_empty() {
                return Err("Empty response from agent".into());
            }

            // Check if response contains expected answer
            if !response.response.contains("4") {
                return Err(format!("Expected '4' in response, got: {}", response.response).into());
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

/// Test agent creation and basic execution with LM Studio
pub struct LMStudioAgentTest;

#[async_trait::async_trait]
impl VerificationTest for LMStudioAgentTest {
    fn test_name(&self) -> &'static str {
        "lm_studio_agent_execution"
    }
    fn description(&self) -> &'static str {
        "Test agent creation and execution with LM Studio"
    }
    fn category(&self) -> TestCategory {
        TestCategory::Core
    }
    fn required_features(&self) -> Vec<ProviderFeature> {
        vec![ProviderFeature::BasicChat]
    }

    async fn execute(&self, config: &TestConfig) -> VerificationResult {
        let start = std::time::Instant::now();
        let mut metadata = std::collections::HashMap::new();

        let result = async {
            // Create agent with LM Studio model
            let mut agent = Agent::builder()
                .provider("lm_studio")
                .model_str("google/gemma-3-12b")
                .system_prompt("You are a helpful assistant. Keep responses very brief.")
                .temperature(0.0)
                .max_tokens(100)
                .build()
                .await?;

            metadata.insert("agent_created".to_string(), serde_json::Value::Bool(true));

            // Execute simple query
            let response = agent
                .execute("What is the capital of France? One word answer.")
                .await?;

            metadata.insert(
                "response_content".to_string(),
                serde_json::Value::String(response.response.clone()),
            );
            metadata.insert(
                "response_length".to_string(),
                serde_json::Value::Number(response.response.len().into()),
            );
            metadata.insert(
                "tools_called_count".to_string(),
                serde_json::Value::Number(response.tools_called.len().into()),
            );
            metadata.insert(
                "used_tools".to_string(),
                serde_json::Value::Bool(response.used_tools),
            );

            // Verify response
            if response.response.is_empty() {
                return Err("Empty response from agent".into());
            }

            // Check for expected answer (Paris)
            let response_lower = response.response.to_lowercase();
            if !response_lower.contains("paris") {
                return Err(
                    format!("Expected 'Paris' in response, got: {}", response.response).into(),
                );
            }

            // Verify no tools were used (this is basic chat)
            if response.used_tools || !response.tools_called.is_empty() {
                return Err("Tools were used when none were expected".into());
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

/// Test LM Studio Gemma 3 27B model specifically
pub struct LMStudioGemma27BTest;

#[async_trait::async_trait]
impl VerificationTest for LMStudioGemma27BTest {
    fn test_name(&self) -> &'static str {
        "lm_studio_gemma_27b_test"
    }
    fn description(&self) -> &'static str {
        "Test LM Studio with Gemma 3 27B model"
    }
    fn category(&self) -> TestCategory {
        TestCategory::Core
    }
    fn required_features(&self) -> Vec<ProviderFeature> {
        vec![ProviderFeature::BasicChat]
    }

    async fn execute(&self, config: &TestConfig) -> VerificationResult {
        let start = std::time::Instant::now();
        let mut metadata = std::collections::HashMap::new();

        let result = async {
            // Create agent with Gemma 3 27B model
            let mut agent = Agent::builder()
                .provider("lm_studio")
                .model_str("google/gemma-3-27b")
                .system_prompt("You are a helpful assistant. Keep responses brief.")
                .temperature(0.0)
                .max_tokens(100)
                .build()
                .await?;

            metadata.insert(
                "model_id".to_string(),
                serde_json::Value::String("google/gemma-3-27b".to_string()),
            );
            metadata.insert(
                "model_display_name".to_string(),
                serde_json::Value::String("Gemma 3 27B (Local)".to_string()),
            );
            metadata.insert(
                "supports_tools".to_string(),
                serde_json::Value::Bool(agent.model().supports_tool_use()),
            );
            metadata.insert(
                "max_output_tokens".to_string(),
                serde_json::Value::Number(agent.model().max_output_tokens().into()),
            );

            // Execute test query
            let response = agent
                .execute("What is 5+7? Answer with just the number.")
                .await?;

            metadata.insert(
                "response_content".to_string(),
                serde_json::Value::String(response.response.clone()),
            );
            metadata.insert(
                "response_length".to_string(),
                serde_json::Value::Number(response.response.len().into()),
            );
            metadata.insert(
                "used_tools".to_string(),
                serde_json::Value::Bool(response.used_tools),
            );

            // Verify response contains expected answer
            if !response.response.contains("12") {
                return Err(
                    format!("Expected '12' in response, got: {}", response.response).into(),
                );
            }

            // Verify model capabilities
            if agent.model().max_output_tokens() == 0 {
                return Err(format!(
                    "Expected max_output_tokens to be non-zero, got: {}",
                    agent.model().max_output_tokens()
                )
                .into());
            }

            if !agent.model().supports_tool_use() {
                return Err("Expected Gemma 3 27B to support tools".into());
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

/// Create LM Studio core test suite
pub fn create_lm_studio_core_suite() -> VerificationSuite {
    VerificationSuite::new("LM Studio Core Functionality")
        .add_test(Box::new(LMStudioHealthTest))
        .add_test(Box::new(LMStudioModelMetadataTest))
        .add_test(Box::new(LMStudioDirectChatTest))
        .add_test(Box::new(LMStudioAgentTest))
        .add_test(Box::new(LMStudioGemma27BTest))
}
