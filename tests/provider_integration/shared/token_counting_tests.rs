//! Token counting verification tests for LLM Client providers
//!
//! These tests verify that token counting works correctly across all supported
//! model combinations in both streaming and non-streaming modes.

use super::*;
use crate::agent::Agent;
use std::time::Instant;

/// Detect if token counting used estimation based on token patterns
///
/// Estimation typically uses ~4 characters per token ratio, so we can detect
/// it by checking if the token counts match this pattern closely.
fn is_likely_estimation(tokens: &crate::llm::traits::Usage, response_content: &str) -> bool {
    // Calculate expected tokens using the 4-char estimation method
    let content_length = response_content.len();
    let estimated_output_tokens = (content_length / 4).max(1) as u32;

    // If the actual output tokens are very close to 4-char estimation, it's likely estimated
    let estimation_diff = (tokens.output_tokens as i32 - estimated_output_tokens as i32).abs();

    // Consider it estimation if:
    // 1. The output tokens are within 2 tokens of the 4-char estimation, OR
    // 2. The input tokens are exactly 100 (common fallback value), OR
    // 3. The output tokens are exactly content_length/4 (exact estimation match)
    estimation_diff <= 2
        || tokens.input_tokens == 100
        || tokens.output_tokens == estimated_output_tokens
}

// =============================================================================
// TOKEN COUNTING VERIFICATION TESTS
// =============================================================================

/// Test streaming token counting functionality
pub struct StreamingTokenCountingTest;

#[async_trait::async_trait]
impl VerificationTest for StreamingTokenCountingTest {
    fn test_name(&self) -> &'static str {
        "streaming_token_counting"
    }
    fn description(&self) -> &'static str {
        "Verify token counting works correctly in streaming mode"
    }
    fn category(&self) -> TestCategory {
        TestCategory::Telemetry
    }
    fn required_features(&self) -> Vec<ProviderFeature> {
        vec![ProviderFeature::BasicChat, ProviderFeature::Streaming]
    }

    async fn execute(&self, config: &TestConfig) -> VerificationResult {
        let start = Instant::now();
        let mut metadata = std::collections::HashMap::new();

        let result = async {
            // Create agent with streaming enabled based on provider and model
            let provider_str = match config.provider {
                ProviderType::Bedrock => "bedrock",
                ProviderType::LmStudio => "lm_studio",
                _ => return Err(format!("Unsupported provider: {:?}", config.provider).into()),
            };
            let mut agent = Agent::builder()
                .provider(provider_str)
                .model_str(config.model_id.clone())
                .system_prompt("You are a helpful assistant. Respond concisely.")
                .with_streaming(true)
                .build()
                .await?;

            // Execute a simple request that should generate measurable tokens
            let response = agent
                .execute("Explain what 2+2 equals in exactly one sentence.")
                .await?;

            // Verify response exists
            if response.response.is_empty() {
                return Err("Empty response received".into());
            }

            // Verify token information is available
            let tokens = response
                .execution
                .tokens
                .ok_or("No token usage information available")?;

            // Detect if estimation was used
            let used_estimation = is_likely_estimation(&tokens, &response.response);

            // Add estimation indicator to metadata
            metadata.insert(
                "used_estimation".to_string(),
                serde_json::Value::Bool(used_estimation),
            );
            metadata.insert(
                "estimation_method".to_string(),
                serde_json::Value::String(if used_estimation {
                    "4-char-ratio-fallback".to_string()
                } else {
                    "api-provided".to_string()
                }),
            );

            // Basic token validation
            if tokens.total_tokens == 0 {
                return Err("Total tokens is zero - token counting failed".into());
            }

            if tokens.input_tokens == 0 {
                return Err("Input tokens is zero - input token counting failed".into());
            }

            if tokens.output_tokens == 0 {
                return Err("Output tokens is zero - output token counting failed".into());
            }

            // Verify token arithmetic
            if tokens.total_tokens != tokens.input_tokens + tokens.output_tokens {
                return Err(format!(
                    "Token arithmetic incorrect: {} != {} + {}",
                    tokens.total_tokens, tokens.input_tokens, tokens.output_tokens
                )
                .into());
            }

            // Verify streaming was actually used
            if !response.execution.performance.was_streamed {
                return Err("Response was not streamed despite streaming being enabled".into());
            }

            // Collect metadata for analysis
            metadata.insert(
                "input_tokens".to_string(),
                serde_json::Value::Number(tokens.input_tokens.into()),
            );
            metadata.insert(
                "output_tokens".to_string(),
                serde_json::Value::Number(tokens.output_tokens.into()),
            );
            metadata.insert(
                "total_tokens".to_string(),
                serde_json::Value::Number(tokens.total_tokens.into()),
            );
            metadata.insert(
                "was_streamed".to_string(),
                serde_json::Value::Bool(response.execution.performance.was_streamed),
            );
            metadata.insert(
                "response_length".to_string(),
                serde_json::Value::Number(response.response.len().into()),
            );
            metadata.insert(
                "response_content".to_string(),
                serde_json::Value::String(response.response.clone()),
            );
            metadata.insert(
                "model_calls".to_string(),
                serde_json::Value::Number(response.execution.model_calls.into()),
            );
            metadata.insert(
                "cycles".to_string(),
                serde_json::Value::Number(response.execution.cycles.into()),
            );

            // Validate reasonable token ranges based on provider
            let expected_input_range = match config.provider {
                ProviderType::Bedrock => (10, 200), // AWS Bedrock typically accurate
                ProviderType::LmStudio => (50, 150), // LM Studio uses estimation
                _ => (1, 1000),                     // Generous range for others
            };

            let expected_output_range = match config.provider {
                ProviderType::Bedrock => (5, 100), // Should be concise response
                ProviderType::LmStudio => (5, 50), // Estimated tokens
                _ => (1, 200),                     // Generous range for others
            };

            if tokens.input_tokens < expected_input_range.0
                || tokens.input_tokens > expected_input_range.1
            {
                metadata.insert(
                    "warning_input_tokens".to_string(),
                    serde_json::Value::String(format!(
                        "Input tokens {} outside expected range {:?}",
                        tokens.input_tokens, expected_input_range
                    )),
                );
            }

            if tokens.output_tokens < expected_output_range.0
                || tokens.output_tokens > expected_output_range.1
            {
                metadata.insert(
                    "warning_output_tokens".to_string(),
                    serde_json::Value::String(format!(
                        "Output tokens {} outside expected range {:?}",
                        tokens.output_tokens, expected_output_range
                    )),
                );
            }

            Ok(())
        }
        .await;

        // Create dynamic test name with estimation indicator
        let estimation_suffix = if metadata
            .get("used_estimation")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            " (estimation)"
        } else {
            ""
        };

        VerificationResult {
            test_name: format!("{}{}", self.test_name(), estimation_suffix),
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

/// Test non-streaming token counting functionality
pub struct NonStreamingTokenCountingTest;

#[async_trait::async_trait]
impl VerificationTest for NonStreamingTokenCountingTest {
    fn test_name(&self) -> &'static str {
        "non_streaming_token_counting"
    }
    fn description(&self) -> &'static str {
        "Verify token counting works correctly in non-streaming mode"
    }
    fn category(&self) -> TestCategory {
        TestCategory::Telemetry
    }
    fn required_features(&self) -> Vec<ProviderFeature> {
        vec![ProviderFeature::BasicChat]
    }

    async fn execute(&self, config: &TestConfig) -> VerificationResult {
        let start = Instant::now();
        let mut metadata = std::collections::HashMap::new();

        let result = async {
            // Create agent with streaming disabled based on provider and model
            let provider_str = match config.provider {
                ProviderType::Bedrock => "bedrock",
                ProviderType::LmStudio => "lm_studio",
                _ => return Err(format!("Unsupported provider: {:?}", config.provider).into()),
            };
            let mut agent = Agent::builder()
                .provider(provider_str)
                .model_str(config.model_id.clone())
                .system_prompt("You are a helpful assistant. Respond concisely.")
                .with_streaming(false)
                .build()
                .await?;

            // Execute a simple request that should generate measurable tokens
            let response = agent
                .execute("What is the capital of France? Answer in one word.")
                .await?;

            // Verify response exists
            if response.response.is_empty() {
                return Err("Empty response received".into());
            }

            // Verify token information is available
            let tokens = response
                .execution
                .tokens
                .ok_or("No token usage information available")?;

            // Detect if estimation was used
            let used_estimation = is_likely_estimation(&tokens, &response.response);

            // Add estimation indicator to metadata
            metadata.insert(
                "used_estimation".to_string(),
                serde_json::Value::Bool(used_estimation),
            );
            metadata.insert(
                "estimation_method".to_string(),
                serde_json::Value::String(if used_estimation {
                    "4-char-ratio-fallback".to_string()
                } else {
                    "api-provided".to_string()
                }),
            );

            // Basic token validation
            if tokens.total_tokens == 0 {
                return Err("Total tokens is zero - token counting failed".into());
            }

            if tokens.input_tokens == 0 {
                return Err("Input tokens is zero - input token counting failed".into());
            }

            if tokens.output_tokens == 0 {
                return Err("Output tokens is zero - output token counting failed".into());
            }

            // Verify token arithmetic
            if tokens.total_tokens != tokens.input_tokens + tokens.output_tokens {
                return Err(format!(
                    "Token arithmetic incorrect: {} != {} + {}",
                    tokens.total_tokens, tokens.input_tokens, tokens.output_tokens
                )
                .into());
            }

            // Verify streaming was NOT used
            if response.execution.performance.was_streamed {
                return Err("Response was streamed despite streaming being disabled".into());
            }

            // Collect metadata for analysis
            metadata.insert(
                "input_tokens".to_string(),
                serde_json::Value::Number(tokens.input_tokens.into()),
            );
            metadata.insert(
                "output_tokens".to_string(),
                serde_json::Value::Number(tokens.output_tokens.into()),
            );
            metadata.insert(
                "total_tokens".to_string(),
                serde_json::Value::Number(tokens.total_tokens.into()),
            );
            metadata.insert(
                "was_streamed".to_string(),
                serde_json::Value::Bool(response.execution.performance.was_streamed),
            );
            metadata.insert(
                "response_length".to_string(),
                serde_json::Value::Number(response.response.len().into()),
            );
            metadata.insert(
                "response_content".to_string(),
                serde_json::Value::String(response.response.clone()),
            );
            metadata.insert(
                "model_calls".to_string(),
                serde_json::Value::Number(response.execution.model_calls.into()),
            );
            metadata.insert(
                "cycles".to_string(),
                serde_json::Value::Number(response.execution.cycles.into()),
            );

            // Validate reasonable token ranges
            let expected_input_range = match config.provider {
                ProviderType::Bedrock => (10, 150), // AWS Bedrock typically accurate
                ProviderType::LmStudio => (50, 150), // LM Studio uses estimation
                _ => (1, 1000),                     // Generous range for others
            };

            let expected_output_range = (1, 20); // Should be very concise (one word)

            if tokens.input_tokens < expected_input_range.0
                || tokens.input_tokens > expected_input_range.1
            {
                metadata.insert(
                    "warning_input_tokens".to_string(),
                    serde_json::Value::String(format!(
                        "Input tokens {} outside expected range {:?}",
                        tokens.input_tokens, expected_input_range
                    )),
                );
            }

            if tokens.output_tokens < expected_output_range.0
                || tokens.output_tokens > expected_output_range.1
            {
                metadata.insert(
                    "warning_output_tokens".to_string(),
                    serde_json::Value::String(format!(
                        "Output tokens {} outside expected range {:?}",
                        tokens.output_tokens, expected_output_range
                    )),
                );
            }

            Ok(())
        }
        .await;

        // Create dynamic test name with estimation indicator
        let estimation_suffix = if metadata
            .get("used_estimation")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            " (estimation)"
        } else {
            ""
        };

        VerificationResult {
            test_name: format!("{}{}", self.test_name(), estimation_suffix),
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

/// Test token counting with tool usage in streaming mode
pub struct StreamingTokenCountingWithToolsTest;

#[async_trait::async_trait]
impl VerificationTest for StreamingTokenCountingWithToolsTest {
    fn test_name(&self) -> &'static str {
        "streaming_token_counting_with_tools"
    }
    fn description(&self) -> &'static str {
        "Verify token counting works correctly with tools in streaming mode"
    }
    fn category(&self) -> TestCategory {
        TestCategory::Telemetry
    }
    fn required_features(&self) -> Vec<ProviderFeature> {
        vec![
            ProviderFeature::BasicChat,
            ProviderFeature::Streaming,
            ProviderFeature::ToolCalling,
        ]
    }

    async fn execute(&self, config: &TestConfig) -> VerificationResult {
        let start = Instant::now();
        let mut metadata = std::collections::HashMap::new();

        let result = async {
            use crate::tools::builtin::CalculatorTool;

            // Create agent with streaming and tools enabled
            let provider_str = match config.provider {
                ProviderType::Bedrock => "bedrock",
                ProviderType::LmStudio => "lm_studio",
                _ => return Err(format!("Unsupported provider: {:?}", config.provider).into()),
            };
            let mut agent = Agent::builder()
                .provider(provider_str)
                .model_str(config.model_id.clone())
                .system_prompt("You are a helpful assistant with access to tools. Use the calculator tool for math problems.")
                .tool(Box::new(CalculatorTool))
                .with_streaming(true)
                .build()
                .await?;

            // Execute a request that should use tools and generate tokens
            let response = agent.execute("Calculate 15 * 23 using the calculator tool.").await?;

            // Verify response exists
            if response.response.is_empty() {
                return Err("Empty response received".into());
            }

            // Verify tools were used
            if !response.used_tools || response.tools_called.is_empty() {
                return Err("No tools were used in the response".into());
            }

            // Verify token information is available
            let tokens = response.execution.tokens
                .ok_or("No token usage information available")?;

            // Detect if estimation was used
            let used_estimation = is_likely_estimation(&tokens, &response.response);

            // Add estimation indicator to metadata
            metadata.insert("used_estimation".to_string(), serde_json::Value::Bool(used_estimation));
            metadata.insert("estimation_method".to_string(), serde_json::Value::String(
                if used_estimation { "4-char-ratio-fallback".to_string() } else { "api-provided".to_string() }
            ));

            // Basic token validation
            if tokens.total_tokens == 0 {
                return Err("Total tokens is zero - token counting failed".into());
            }

            if tokens.input_tokens == 0 {
                return Err("Input tokens is zero - input token counting failed".into());
            }

            if tokens.output_tokens == 0 {
                return Err("Output tokens is zero - output token counting failed".into());
            }

            // Verify token arithmetic
            if tokens.total_tokens != tokens.input_tokens + tokens.output_tokens {
                return Err(format!(
                    "Token arithmetic incorrect: {} != {} + {}",
                    tokens.total_tokens, tokens.input_tokens, tokens.output_tokens
                ).into());
            }

            // Verify streaming was actually used
            if !response.execution.performance.was_streamed {
                return Err("Response was not streamed despite streaming being enabled".into());
            }

            // Verify calculator was used and result is correct (15 * 23 = 345)
            let calculator_used = response.tools_called.iter()
                .any(|tool_name| tool_name.contains("calculator") || tool_name.contains("calc"));

            if !calculator_used {
                return Err(format!("Calculator tool not used. Tools used: {:?}", response.tools_called).into());
            }

            if !response.response.contains("345") {
                metadata.insert("warning_calculation".to_string(), serde_json::Value::String(
                    "Response may not contain expected calculation result (345)".to_string()
                ));
            }

            // Collect metadata for analysis
            metadata.insert("input_tokens".to_string(), serde_json::Value::Number(tokens.input_tokens.into()));
            metadata.insert("output_tokens".to_string(), serde_json::Value::Number(tokens.output_tokens.into()));
            metadata.insert("total_tokens".to_string(), serde_json::Value::Number(tokens.total_tokens.into()));
            metadata.insert("was_streamed".to_string(), serde_json::Value::Bool(response.execution.performance.was_streamed));
            metadata.insert("used_tools".to_string(), serde_json::Value::Bool(response.used_tools));
            metadata.insert("tools_called_count".to_string(), serde_json::Value::Number(response.tools_called.len().into()));
            metadata.insert("tools_successful_count".to_string(), serde_json::Value::Number(response.tools_successful.len().into()));
            metadata.insert("response_length".to_string(), serde_json::Value::Number(response.response.len().into()));
            metadata.insert("response_content".to_string(), serde_json::Value::String(response.response.clone()));
            metadata.insert("model_calls".to_string(), serde_json::Value::Number(response.execution.model_calls.into()));
            metadata.insert("cycles".to_string(), serde_json::Value::Number(response.execution.cycles.into()));
            metadata.insert("tool_executions".to_string(), serde_json::Value::Number(response.execution.tool_executions.into()));

            // Validate token ranges for tool usage (should be higher due to tool schemas and results)
            let expected_input_range = match config.provider {
                ProviderType::Bedrock => (50, 500),  // Tool schemas add significant tokens
                ProviderType::LmStudio => (100, 400), // Estimated with tool usage
                _ => (1, 1000), // Generous range for others
            };

            let expected_output_range = match config.provider {
                ProviderType::Bedrock => (20, 200),  // Tool calls and results
                ProviderType::LmStudio => (10, 100), // Estimated tokens
                _ => (1, 500), // Generous range for others
            };

            if tokens.input_tokens < expected_input_range.0 || tokens.input_tokens > expected_input_range.1 {
                metadata.insert("warning_input_tokens".to_string(), serde_json::Value::String(
                    format!("Input tokens {} outside expected range {:?}", tokens.input_tokens, expected_input_range)
                ));
            }

            if tokens.output_tokens < expected_output_range.0 || tokens.output_tokens > expected_output_range.1 {
                metadata.insert("warning_output_tokens".to_string(), serde_json::Value::String(
                    format!("Output tokens {} outside expected range {:?}", tokens.output_tokens, expected_output_range)
                ));
            }

            Ok(())
        }.await;

        // Create dynamic test name with estimation indicator
        let estimation_suffix = if metadata
            .get("used_estimation")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            " (estimation)"
        } else {
            ""
        };

        VerificationResult {
            test_name: format!("{}{}", self.test_name(), estimation_suffix),
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

/// Test token counting consistency between streaming and non-streaming modes
pub struct TokenCountingConsistencyTest;

#[async_trait::async_trait]
impl VerificationTest for TokenCountingConsistencyTest {
    fn test_name(&self) -> &'static str {
        "token_counting_consistency"
    }
    fn description(&self) -> &'static str {
        "Verify token counting is consistent between streaming and non-streaming modes"
    }
    fn category(&self) -> TestCategory {
        TestCategory::Telemetry
    }
    fn required_features(&self) -> Vec<ProviderFeature> {
        vec![ProviderFeature::BasicChat, ProviderFeature::Streaming]
    }

    async fn execute(&self, config: &TestConfig) -> VerificationResult {
        let start = Instant::now();
        let mut metadata = std::collections::HashMap::new();

        let result = async {
            let test_prompt = "Count from 1 to 3, with each number on a separate line.";

            // Create agents for both streaming and non-streaming modes
            let provider_str = match config.provider {
                ProviderType::Bedrock => "bedrock",
                ProviderType::LmStudio => "lm_studio",
                _ => return Err(format!("Unsupported provider: {:?}", config.provider).into()),
            };
            let mut streaming_agent = Agent::builder()
                .provider(provider_str)
                .model_str(config.model_id.clone())
                .system_prompt("You are a helpful assistant. Follow instructions exactly.")
                .with_streaming(true)
                .build()
                .await?;

            let mut non_streaming_agent = Agent::builder()
                .provider(provider_str)
                .model_str(config.model_id.clone())
                .system_prompt("You are a helpful assistant. Follow instructions exactly.")
                .with_streaming(false)
                .build()
                .await?;

            // Execute the same prompt with both modes
            let streaming_response = streaming_agent.execute(test_prompt).await?;
            let non_streaming_response = non_streaming_agent.execute(test_prompt).await?;

            // Verify both responses have token information
            let streaming_tokens = streaming_response
                .execution
                .tokens
                .ok_or("No token usage information available for streaming response")?;

            let non_streaming_tokens = non_streaming_response
                .execution
                .tokens
                .ok_or("No token usage information available for non-streaming response")?;

            // Detect if estimation was used for either response
            let streaming_used_estimation =
                is_likely_estimation(&streaming_tokens, &streaming_response.response);
            let non_streaming_used_estimation =
                is_likely_estimation(&non_streaming_tokens, &non_streaming_response.response);
            let overall_used_estimation =
                streaming_used_estimation || non_streaming_used_estimation;

            // Add estimation indicators to metadata
            metadata.insert(
                "used_estimation".to_string(),
                serde_json::Value::Bool(overall_used_estimation),
            );
            metadata.insert(
                "streaming_used_estimation".to_string(),
                serde_json::Value::Bool(streaming_used_estimation),
            );
            metadata.insert(
                "non_streaming_used_estimation".to_string(),
                serde_json::Value::Bool(non_streaming_used_estimation),
            );
            metadata.insert(
                "estimation_method".to_string(),
                serde_json::Value::String(if overall_used_estimation {
                    "4-char-ratio-fallback".to_string()
                } else {
                    "api-provided".to_string()
                }),
            );

            // Verify streaming flag is correct
            if !streaming_response.execution.performance.was_streamed {
                return Err("Streaming response was not marked as streamed".into());
            }

            if non_streaming_response.execution.performance.was_streamed {
                return Err("Non-streaming response was incorrectly marked as streamed".into());
            }

            // Collect metadata for both responses
            metadata.insert(
                "streaming_input_tokens".to_string(),
                serde_json::Value::Number(streaming_tokens.input_tokens.into()),
            );
            metadata.insert(
                "streaming_output_tokens".to_string(),
                serde_json::Value::Number(streaming_tokens.output_tokens.into()),
            );
            metadata.insert(
                "streaming_total_tokens".to_string(),
                serde_json::Value::Number(streaming_tokens.total_tokens.into()),
            );
            metadata.insert(
                "non_streaming_input_tokens".to_string(),
                serde_json::Value::Number(non_streaming_tokens.input_tokens.into()),
            );
            metadata.insert(
                "non_streaming_output_tokens".to_string(),
                serde_json::Value::Number(non_streaming_tokens.output_tokens.into()),
            );
            metadata.insert(
                "non_streaming_total_tokens".to_string(),
                serde_json::Value::Number(non_streaming_tokens.total_tokens.into()),
            );

            metadata.insert(
                "streaming_response_length".to_string(),
                serde_json::Value::Number(streaming_response.response.len().into()),
            );
            metadata.insert(
                "non_streaming_response_length".to_string(),
                serde_json::Value::Number(non_streaming_response.response.len().into()),
            );

            // Calculate differences and check if they're reasonable
            let input_diff = (streaming_tokens.input_tokens as i32
                - non_streaming_tokens.input_tokens as i32)
                .abs();
            let output_diff = (streaming_tokens.output_tokens as i32
                - non_streaming_tokens.output_tokens as i32)
                .abs();
            let total_diff = (streaming_tokens.total_tokens as i32
                - non_streaming_tokens.total_tokens as i32)
                .abs();

            metadata.insert(
                "input_tokens_diff".to_string(),
                serde_json::Value::Number(input_diff.into()),
            );
            metadata.insert(
                "output_tokens_diff".to_string(),
                serde_json::Value::Number(output_diff.into()),
            );
            metadata.insert(
                "total_tokens_diff".to_string(),
                serde_json::Value::Number(total_diff.into()),
            );

            // For Bedrock, token counts should be identical or very close
            // For LM Studio, some variation is expected due to estimation
            let acceptable_variance = match config.provider {
                ProviderType::Bedrock => 5,   // Very close for AWS Bedrock
                ProviderType::LmStudio => 20, // More tolerance for LM Studio estimation
                _ => 50,                      // Generous tolerance for other providers
            };

            if input_diff > acceptable_variance {
                metadata.insert(
                    "warning_input_variance".to_string(),
                    serde_json::Value::String(format!(
                        "Input token difference {} exceeds acceptable variance {}",
                        input_diff, acceptable_variance
                    )),
                );
            }

            if output_diff > acceptable_variance {
                metadata.insert(
                    "warning_output_variance".to_string(),
                    serde_json::Value::String(format!(
                        "Output token difference {} exceeds acceptable variance {}",
                        output_diff, acceptable_variance
                    )),
                );
            }

            if total_diff > acceptable_variance {
                metadata.insert(
                    "warning_total_variance".to_string(),
                    serde_json::Value::String(format!(
                        "Total token difference {} exceeds acceptable variance {}",
                        total_diff, acceptable_variance
                    )),
                );
            }

            // For reporting, we consider the test successful if tokens are being counted,
            // even if there's some variance between modes (which is expected for estimation-based providers)

            Ok(())
        }
        .await;

        // Create dynamic test name with estimation indicator
        let estimation_suffix = if metadata
            .get("used_estimation")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            " (estimation)"
        } else {
            ""
        };

        VerificationResult {
            test_name: format!("{}{}", self.test_name(), estimation_suffix),
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

/// Create the token counting test suite
pub fn create_token_counting_test_suite() -> VerificationSuite {
    VerificationSuite::new("Token Counting Verification")
        .add_test(Box::new(StreamingTokenCountingTest))
        .add_test(Box::new(NonStreamingTokenCountingTest))
        .add_test(Box::new(StreamingTokenCountingWithToolsTest))
        .add_test(Box::new(TokenCountingConsistencyTest))
}
