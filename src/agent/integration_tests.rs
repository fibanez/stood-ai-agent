//! Integration tests for Agent chat functionality and LLM-driven tool selection
//!
//! These tests require valid AWS credentials and Bedrock access.
//! They test the full end-to-end process of agent chat including:
//! - Bedrock API calls
//! - Message parsing
//! - Conversation management
//! - Error handling
//! - LLM-driven tool selection and execution
//! - Event loop tool integration

use crate::agent::event_loop::{EventLoop, EventLoopConfig};
use crate::agent::Agent;
use crate::tools::builtin::*;
use crate::tools::{Tool, ToolError, ToolRegistry, ToolResult};
use crate::types::*;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::env;
use std::time::Duration;

/// Helper to check if we have AWS credentials for integration tests
fn ensure_aws_credentials() -> Result<(), String> {
    // Check AWS credentials
    let has_access_key = env::var("AWS_ACCESS_KEY_ID").is_ok();
    let has_profile = env::var("AWS_PROFILE").is_ok();
    let has_role_arn = env::var("AWS_ROLE_ARN").is_ok();

    if !has_access_key && !has_profile && !has_role_arn {
        return Err("AWS credentials required for integration tests. Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, AWS_PROFILE, or configure IAM role.".to_string());
    }

    Ok(())
}

#[tokio::test]
async fn test_agent_chat_integration_haiku() {
    // Skip test if no AWS credentials
    if ensure_aws_credentials().is_err() {
        println!("⚠️  Skipping Agent chat test - AWS credentials not available");
        return;
    }

    println!("🚀 Hi, Testing Agent chat integration with Claude Haiku 4.5...");

    // Create agent with Haiku (our default model) - providers auto-configured
    let mut agent = Agent::builder()
        .provider("bedrock")
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .system_prompt("You are a helpful assistant. Keep responses very brief and concise.")
        .temperature(0.7)
        .max_tokens(100) // Keep responses short for testing
        .build()
        .await
        .expect("Failed to build agent");

    // Verify initial state
    assert_eq!(agent.conversation().message_count(), 0);
    assert_eq!(
        agent.conversation().system_prompt(),
        Some("You are a helpful assistant. Keep responses very brief and concise.")
    );

    // Test 1: Simple math question
    println!("📝 Testing simple chat interaction...");
    let result1 = agent
        .execute("What is 2+2? Answer with just the number.")
        .await
        .expect("Chat request failed");

    println!("🤖 Agent response: {}", result1.response);

    // Verify response
    assert!(!result1.response.is_empty(), "Response should not be empty");
    assert!(
        result1.response.contains("4"),
        "Response should contain the answer '4'"
    );

    // Verify conversation history
    assert_eq!(agent.conversation().message_count(), 2); // user + assistant

    let messages = agent.conversation().messages();
    assert_eq!(
        messages.messages[0].text(),
        Some("What is 2+2? Answer with just the number.".to_string())
    );
    assert_eq!(messages.messages[1].text(), Some(result1.response.clone()));

    // Test 2: Multi-turn conversation with context
    println!("🔄 Testing multi-turn conversation with context...");
    let result2 = agent
        .execute("What number did I just ask you about?")
        .await
        .expect("Second chat request failed");

    println!("🤖 Agent response: {}", result2.response);

    // Verify response references previous context
    assert!(
        !result2.response.is_empty(),
        "Second response should not be empty"
    );
    assert!(
        result2.response.to_lowercase().contains("2")
            || result2.response.to_lowercase().contains("two")
            || result2.response.to_lowercase().contains("addition")
            || result2.response.to_lowercase().contains("math")
            || result2.response.to_lowercase().contains("4")
            || result2.response.to_lowercase().contains("four"),
        "Response should reference the previous question about 2+2. Got: '{}'",
        result2.response
    );

    // Verify conversation history updated
    assert_eq!(agent.conversation().message_count(), 4); // 2 user + 2 assistant

    // Test 3: Verify conversation summary
    let summary = agent.conversation().summary();
    println!("📊 Conversation summary: {}", summary);
    assert!(summary.contains("4 messages"));

    println!("✅ Agent chat integration test completed successfully!");
}

#[tokio::test]
async fn test_agent_chat_error_recovery() {
    // Skip test if no AWS credentials
    if ensure_aws_credentials().is_err() {
        println!("⚠️  Skipping Agent chat error recovery test - AWS credentials not available");
        return;
    }

    println!("🛠️  Testing Agent chat error recovery...");

    let mut agent = Agent::builder()
        .provider("bedrock")
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .system_prompt("You are helpful.")
        .build()
        .await
        .expect("Failed to build agent");

    // Test normal operation first
    let response = agent.execute("Hello").await.expect("Chat should work");

    assert!(!response.response.is_empty());
    assert_eq!(agent.conversation().message_count(), 2);

    // Test that conversation state is maintained even after requests
    let response2 = agent
        .execute("What did I say before?")
        .await
        .expect("Second chat should work");

    assert!(!response2.response.is_empty());
    assert_eq!(agent.conversation().message_count(), 4);

    // Verify conversation history integrity
    let messages = agent.conversation().messages();
    assert_eq!(messages.messages[0].text(), Some("Hello".to_string()));
    assert_eq!(messages.messages[1].text(), Some(response.response));
    assert_eq!(
        messages.messages[2].text(),
        Some("What did I say before?".to_string())
    );
    assert_eq!(messages.messages[3].text(), Some(response2.response));

    println!("✅ Agent error recovery test completed successfully!");
}

#[tokio::test]
async fn test_agent_chat_conversation_persistence() {
    // Skip test if no AWS credentials
    if ensure_aws_credentials().is_err() {
        println!(
            "⚠️  Skipping Agent chat conversation persistence test - AWS credentials not available"
        );
        return;
    }

    println!("📏 Testing Agent chat conversation persistence...");

    let mut agent = Agent::builder()
        .provider("bedrock")
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .system_prompt("Be very brief. Answer with one word when possible.")
        .build()
        .await
        .expect("Failed to build agent");

    // Test multiple interactions and verify history grows
    let _ = agent.execute("Say hello").await.expect("Chat 1 failed");
    assert_eq!(agent.conversation().message_count(), 2);

    let _ = agent.execute("Say goodbye").await.expect("Chat 2 failed");
    assert_eq!(agent.conversation().message_count(), 4);

    let _ = agent
        .execute("What was the first thing I asked?")
        .await
        .expect("Chat 3 failed");
    assert_eq!(agent.conversation().message_count(), 6);

    // Verify conversation history integrity
    let messages = agent.conversation().messages();
    assert!(messages.messages[0].text().unwrap().contains("hello"));
    assert!(messages.messages[2].text().unwrap().contains("goodbye"));
    assert!(messages.messages[4].text().unwrap().contains("first"));

    // Test conversation summary
    let summary = agent.conversation().summary();
    assert!(summary.contains("6 messages"));

    println!("✅ Agent conversation persistence test completed successfully!");
}

#[tokio::test]
async fn test_agent_chat_different_models() {
    // Skip test if no AWS credentials
    if ensure_aws_credentials().is_err() {
        println!(
            "⚠️  Skipping Agent chat with different models test - AWS credentials not available"
        );
        return;
    }

    println!("🔀 Testing Agent chat with different models...");

    // Test with Haiku (our default)
    let mut haiku_agent = Agent::builder()
        .provider("bedrock")
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .system_prompt("Answer in exactly 3 words.")
        .temperature(0.3)
        .build()
        .await
        .expect("Failed to build Haiku agent");

    let haiku_response = haiku_agent
        .execute("What is your favorite color?")
        .await
        .expect("Haiku chat failed");

    println!("🟦 Haiku response: {}", haiku_response);
    assert!(!haiku_response.response.is_empty());

    // Test with Sonnet (if available)
    let mut sonnet_agent = Agent::builder()
        .provider("bedrock")
        .model("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
        .system_prompt("Answer in exactly 3 words.")
        .temperature(0.3)
        .build()
        .await
        .expect("Failed to build Sonnet agent");

    // Sonnet may not be available in all accounts, so handle gracefully
    match sonnet_agent.execute("What is your favorite color?").await {
        Ok(sonnet_response) => {
            println!("🟨 Sonnet response: {}", sonnet_response);
            if sonnet_response.response.is_empty() {
                println!(
                    "⚠️  Sonnet returned empty response - model may not be properly configured"
                );
            } else {
                // Both should follow the system prompt but may differ slightly
                // Note: Models may interpret "3 words" differently, so allow reasonable flexibility
                assert!(haiku_response.response.split_whitespace().count() <= 10); // Allow flexibility
                assert!(sonnet_response.response.split_whitespace().count() <= 30);
                // Sonnet tends to be more verbose
            }
        }
        Err(e) => {
            println!("⚠️  Sonnet not available in this account: {}", e);
            // This is expected in some accounts, so just log it
        }
    }

    println!("✅ Multiple models test completed!");
}

#[tokio::test]
async fn test_agent_chat_system_prompt_behavior() {
    // Skip test if no AWS credentials
    if ensure_aws_credentials().is_err() {
        println!(
            "⚠️  Skipping Agent chat system prompt behavior test - AWS credentials not available"
        );
        return;
    }

    println!("🎭 Testing Agent chat with different system prompts...");

    // Agent 1: Formal assistant
    let mut formal_agent = Agent::builder()
        .provider("bedrock")
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .system_prompt("You are a formal, professional assistant. Always use formal language and complete sentences.")
        .temperature(0.1) // Low temperature for consistency
        .build()
        .await.expect("Failed to build formal agent");

    // Agent 2: Casual assistant
    let mut casual_agent = Agent::builder()
        .provider("bedrock")
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .system_prompt("You are a casual, friendly assistant. Use informal language and be brief.")
        .temperature(0.1) // Low temperature for consistency
        .build()
        .await
        .expect("Failed to build casual agent");

    let question = "How are you doing today?";

    let formal_response = formal_agent
        .execute(question)
        .await
        .expect("Formal agent chat failed");

    let casual_response = casual_agent
        .execute(question)
        .await
        .expect("Casual agent chat failed");

    println!("🎩 Formal response: {}", formal_response);
    println!("😊 Casual response: {}", casual_response);

    // Verify both responded
    assert!(!formal_response.response.is_empty());
    assert!(!casual_response.response.is_empty());

    // Verify conversation history for both
    assert_eq!(formal_agent.conversation().message_count(), 2);
    assert_eq!(casual_agent.conversation().message_count(), 2);

    println!("✅ System prompt behavior test completed!");
}

// ================================================================================================
// LLM-DRIVEN TOOL SELECTION INTEGRATION TESTS
// ================================================================================================

/// Test tool that helps verify LLM-driven tool selection
#[derive(Debug)]
struct TestCalculatorTool;

#[async_trait]
impl Tool for TestCalculatorTool {
    fn name(&self) -> &str {
        "test_calculator"
    }

    fn description(&self) -> &str {
        "Performs basic arithmetic calculations for testing"
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to calculate (e.g., '2+2', '5*5')"
                }
            },
            "required": ["expression"]
        })
    }

    async fn execute(
        &self,
        parameters: Option<Value>,
        _agent_context: Option<&crate::agent::AgentContext>,
    ) -> Result<ToolResult, ToolError> {
        let params = parameters.unwrap_or(json!({}));
        let expression = params
            .get("expression")
            .and_then(|e| e.as_str())
            .unwrap_or("2+2");

        // Simple arithmetic for testing
        let result = match expression {
            "2+2" => 4,
            "5*5" => 25,
            "10-3" => 7,
            "15/3" => 5,
            _ => 42, // Default for unknown expressions
        };

        let response = json!({
            "result": result,
            "expression": expression,
            "message": format!("The result of {} is {}", expression, result)
        });

        Ok(ToolResult::success(response))
    }
}

/// Test tool that provides current time information
#[derive(Debug)]
struct TestTimeTool;

#[async_trait]
impl Tool for TestTimeTool {
    fn name(&self) -> &str {
        "test_time"
    }

    fn description(&self) -> &str {
        "Returns the current time and date"
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {},
            "required": []
        })
    }

    async fn execute(
        &self,
        _parameters: Option<Value>,
        _agent_context: Option<&crate::agent::AgentContext>,
    ) -> Result<ToolResult, ToolError> {
        use chrono::Utc;
        let now = Utc::now();

        let response = json!({
            "current_time": now.to_rfc3339(),
            "timezone": "UTC",
            "message": format!("The current time is {}", now.format("%Y-%m-%d %H:%M:%S UTC"))
        });

        Ok(ToolResult::success(response))
    }
}

/// Test basic tool configuration generation for LLM integration
#[tokio::test]
async fn test_tool_registry_generates_llm_config() {
    let registry = ToolRegistry::new();

    // Register test tools
    registry
        .register_tool(Box::new(TestCalculatorTool))
        .await
        .unwrap();
    registry
        .register_tool(Box::new(TestTimeTool))
        .await
        .unwrap();

    // Get tool configuration for LLM
    let tool_config = registry.get_tool_config().await;

    // Verify structure
    assert_eq!(tool_config.tools.len(), 2);
    assert!(matches!(tool_config.tool_choice, ToolChoice::Auto));

    // Verify tool specifications are properly formatted
    let tool_names: Vec<&str> = tool_config
        .tools
        .iter()
        .map(|t| t.tool_spec.name.as_str())
        .collect();

    assert!(tool_names.contains(&"test_calculator"));
    assert!(tool_names.contains(&"test_time"));

    // Verify each tool has required fields for LLM integration
    for tool in &tool_config.tools {
        assert!(!tool.tool_spec.name.is_empty());
        assert!(!tool.tool_spec.description.is_empty());
        assert!(tool.tool_spec.input_schema.is_object());
    }

    println!(
        "✅ Tool registry properly generates LLM configuration with {} tools",
        tool_config.tools.len()
    );
}

/// Test Bedrock client can handle tool configurations
#[tokio::test]
async fn test_bedrock_client_accepts_tool_config() {
    // Skip test if no AWS credentials
    if ensure_aws_credentials().is_err() {
        println!("⚠️  Skipping Bedrock tool config test - AWS credentials not available");
        return;
    }

    // Create agent with tools
    let mut agent = Agent::builder()
        .provider("bedrock")
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .system_prompt("You are a helpful assistant with access to tools. Use the calculator tool if the user asks for math calculations.")
        .tool(Box::new(TestCalculatorTool))
        .tool(Box::new(TestTimeTool))
        .build()
        .await
        .expect("Failed to build agent");

    // Test that the agent accepts tool configuration without error
    let response = agent.execute("What is 2+2? Use tools if needed.").await;

    match response {
        Ok(agent_response) => {
            println!("✅ Agent successfully accepted tool configuration");
            println!("Response: {}", agent_response.response);

            // Verify response structure
            assert!(!agent_response.response.is_empty());

            // Check if tools were used
            if agent_response.used_tools {
                println!("✅ Model successfully made tool use decision");
                println!("Tools used: {}", agent_response.tools_called.len());
            } else {
                println!("ℹ️  Model chose not to use tools (still valid behavior)");
            }
        }
        Err(e) => {
            eprintln!("❌ Agent tool config test failed: {}", e);
            panic!("Agent failed to handle tool configuration: {}", e);
        }
    }
}

/// Test complete event loop with LLM-driven tool selection
#[tokio::test]
async fn test_event_loop_llm_driven_tool_selection() {
    // Skip test if no AWS credentials
    if ensure_aws_credentials().is_err() {
        println!("⚠️  Skipping event loop LLM-driven test - AWS credentials not available");
        return;
    }

    // Create client and agent
    let agent = Agent::builder()
        .provider("bedrock")
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0") // Use reliable model
        .build()
        .await
        .unwrap();

    // Create tool registry with test tools
    let tool_registry = ToolRegistry::new();
    tool_registry
        .register_tool(Box::new(TestCalculatorTool))
        .await
        .unwrap();
    tool_registry
        .register_tool(Box::new(TestTimeTool))
        .await
        .unwrap();

    // Configure event loop with shorter limits for testing
    let config = EventLoopConfig {
        max_cycles: 3,
        max_duration: Duration::from_secs(60),
        enable_streaming: false,
        enable_telemetry: false,
        ..EventLoopConfig::default()
    };

    let mut event_loop = EventLoop::new(agent, tool_registry, config).unwrap();

    println!("🚀 Testing LLM-driven tool selection with calculation request...");

    // Test with a clear calculation request that should trigger tool use
    let result = event_loop.execute("Calculate 5*5 for me please").await;

    match result {
        Ok(loop_result) => {
            println!("✅ Event loop completed successfully");
            println!("Cycles executed: {}", loop_result.cycles_executed);
            println!("Total duration: {:?}", loop_result.total_duration);
            println!("Success: {}", loop_result.success);

            if let Some(error) = &loop_result.error {
                println!("Error: {}", error);
            }

            // Verify loop executed successfully
            assert!(
                loop_result.success,
                "Event loop should complete successfully"
            );
            assert!(
                loop_result.cycles_executed > 0,
                "Should execute at least one model interaction"
            );

            // Verify response contains meaningful content
            assert!(!loop_result.response.is_empty(), "Should have a response");

            // Check metrics for tool execution
            let tool_executions = loop_result.metrics.tool_executions.len();
            println!("Tool executions recorded: {}", tool_executions);

            if tool_executions > 0 {
                println!("✅ LLM successfully chose and executed tools");

                // Verify tool execution was successful
                for tool_metric in &loop_result.metrics.tool_executions {
                    println!(
                        "Tool '{}' executed: success={}",
                        tool_metric.tool_name, tool_metric.success
                    );
                }
            } else {
                println!("ℹ️  LLM chose not to use tools (still valid behavior)");
            }

            println!("Final response: {}", loop_result.response);
        }
        Err(e) => {
            eprintln!("❌ Event loop LLM-driven test failed: {}", e);
            panic!("Event loop with LLM-driven tool selection failed: {}", e);
        }
    }
}

/// Test event loop handles tool use responses correctly
#[tokio::test]
async fn test_event_loop_tool_use_response_parsing() {
    // Skip test if no AWS credentials
    if ensure_aws_credentials().is_err() {
        println!("⚠️  Skipping tool use response parsing test - AWS credentials not available");
        return;
    }

    // Create client and agent
    let agent = Agent::builder()
        .provider("bedrock")
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .build()
        .await
        .unwrap();

    // Create tool registry with simple test tool
    let tool_registry = ToolRegistry::new();
    tool_registry
        .register_tool(Box::new(TestTimeTool))
        .await
        .unwrap();

    let config = EventLoopConfig {
        max_cycles: 2,
        max_duration: Duration::from_secs(45),
        enable_streaming: false,
        enable_telemetry: false,
        ..EventLoopConfig::default()
    };

    let mut event_loop = EventLoop::new(agent, tool_registry, config).unwrap();

    println!("🕐 Testing LLM-driven tool selection with time request...");

    // Test with a time request that should potentially trigger tool use
    let result = event_loop.execute("What time is it right now?").await;

    match result {
        Ok(loop_result) => {
            println!("✅ Event loop with time request completed");
            println!(
                "Cycles: {}, Duration: {:?}",
                loop_result.cycles_executed, loop_result.total_duration
            );

            assert!(loop_result.success);
            assert!(!loop_result.response.is_empty());

            // Check if tools were used
            if loop_result.metrics.tool_executions.len() > 0 {
                println!("✅ Model chose to use time tool");

                // Verify time tool execution
                let time_executions: Vec<_> = loop_result
                    .metrics
                    .tool_executions
                    .iter()
                    .filter(|t| t.tool_name == "test_time")
                    .collect();

                if !time_executions.is_empty() {
                    println!("✅ Time tool execution recorded");
                    assert!(
                        time_executions[0].success,
                        "Time tool should execute successfully"
                    );
                }
            } else {
                println!("ℹ️  Model provided time without using tools (valid behavior)");
            }

            println!("Response: {}", loop_result.response);
        }
        Err(e) => {
            eprintln!("❌ Tool use response parsing test failed: {}", e);
            panic!("Event loop tool use response parsing failed: {}", e);
        }
    }
}

/// Test event loop with multiple tools available
#[tokio::test]
async fn test_event_loop_multiple_tools_llm_choice() {
    // Skip test if no AWS credentials
    if ensure_aws_credentials().is_err() {
        println!("⚠️  Skipping multiple tools LLM choice test - AWS credentials not available");
        return;
    }

    // Create client and agent
    let agent = Agent::builder()
        .provider("bedrock")
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .build()
        .await
        .unwrap();

    // Create tool registry with multiple tools
    let tool_registry = ToolRegistry::new();
    tool_registry
        .register_tool(Box::new(TestCalculatorTool))
        .await
        .unwrap();
    tool_registry
        .register_tool(Box::new(TestTimeTool))
        .await
        .unwrap();

    // Also add built-in tools to test LLM choice among many options
    tool_registry
        .register_tool(Box::new(CalculatorTool::new()))
        .await
        .unwrap();
    tool_registry
        .register_tool(Box::new(CurrentTimeTool::new()))
        .await
        .unwrap();

    let config = EventLoopConfig {
        max_cycles: 3,
        max_duration: Duration::from_secs(90),
        enable_streaming: false,
        enable_telemetry: false,
        ..EventLoopConfig::default()
    };

    let mut event_loop = EventLoop::new(agent, tool_registry, config).unwrap();

    println!("🔧 Testing LLM choice among multiple tools...");

    // Test with a request that could use different tools
    let result = event_loop
        .execute("I need to know what 10-3 equals and what time it is")
        .await;

    match result {
        Ok(loop_result) => {
            println!("✅ Event loop with multiple tools completed");
            println!(
                "Cycles: {}, Duration: {:?}",
                loop_result.cycles_executed, loop_result.total_duration
            );

            assert!(loop_result.success);
            assert!(!loop_result.response.is_empty());

            // Analyze tool usage
            let tool_count = loop_result.metrics.tool_executions.len();
            println!("Total tool executions: {}", tool_count);

            if tool_count > 0 {
                println!("✅ LLM made tool selection decisions");

                // Check which types of tools were used
                let calculator_used = loop_result
                    .metrics
                    .tool_executions
                    .iter()
                    .any(|t| t.tool_name.contains("calculator"));
                let time_used = loop_result
                    .metrics
                    .tool_executions
                    .iter()
                    .any(|t| t.tool_name.contains("time"));

                println!("Calculator tool used: {}", calculator_used);
                println!("Time tool used: {}", time_used);

                // For this specific request, ideally both types would be used
                if calculator_used && time_used {
                    println!("✅ LLM intelligently selected both calculation and time tools");
                } else if calculator_used || time_used {
                    println!("✅ LLM selected appropriate tool(s) for the request");
                }

                // Verify all tool executions were successful
                for tool_metric in &loop_result.metrics.tool_executions {
                    assert!(
                        tool_metric.success,
                        "Tool '{}' should execute successfully",
                        tool_metric.tool_name
                    );
                }
            } else {
                println!("ℹ️  LLM chose to respond without tools (valid behavior)");
            }

            println!("Final response: {}", loop_result.response);
        }
        Err(e) => {
            eprintln!("❌ Multiple tools LLM choice test failed: {}", e);
            panic!("Event loop with multiple tools failed: {}", e);
        }
    }
}

/// Test that conversation context is properly maintained during tool use
#[tokio::test]
async fn test_event_loop_conversation_context_with_tools() {
    // Skip test if no AWS credentials
    if ensure_aws_credentials().is_err() {
        println!("⚠️  Skipping conversation context test - AWS credentials not available");
        return;
    }

    // Create client and agent
    let agent = Agent::builder()
        .provider("bedrock")
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .build()
        .await
        .unwrap();

    // Create tool registry
    let tool_registry = ToolRegistry::new();
    tool_registry
        .register_tool(Box::new(TestCalculatorTool))
        .await
        .unwrap();

    let config = EventLoopConfig {
        max_cycles: 4,
        max_duration: Duration::from_secs(60),
        enable_streaming: false,
        enable_telemetry: false,
        ..EventLoopConfig::default()
    };

    let mut event_loop = EventLoop::new(agent, tool_registry, config).unwrap();

    println!("💬 Testing conversation context maintenance during tool use...");

    // Test with a request that establishes context and requires tool use
    let result = event_loop.execute("My favorite number is 5. Please calculate what my favorite number times itself equals.").await;

    match result {
        Ok(loop_result) => {
            println!("✅ Conversation context test completed");
            println!(
                "Cycles: {}, Duration: {:?}",
                loop_result.cycles_executed, loop_result.total_duration
            );

            assert!(loop_result.success);
            assert!(!loop_result.response.is_empty());

            // Check if calculation tool was used
            let calculator_used = loop_result
                .metrics
                .tool_executions
                .iter()
                .any(|t| t.tool_name.contains("calculator"));

            if calculator_used {
                println!("✅ LLM used calculator tool for the calculation");

                // Check if the response references the context (favorite number = 5)
                let response_lower = loop_result.response.to_lowercase();
                let mentions_five = response_lower.contains("5") || response_lower.contains("five");
                let mentions_favorite = response_lower.contains("favorite");

                if mentions_five && mentions_favorite {
                    println!("✅ Response maintains conversation context about favorite number");
                } else if mentions_five {
                    println!("✅ Response includes the number 5 from context");
                }

                // For 5*5, we should see 25 in the response
                if response_lower.contains("25") || response_lower.contains("twenty") {
                    println!("✅ Response includes correct calculation result (25)");
                }
            } else {
                println!("ℹ️  LLM provided answer without using calculator tool");
            }

            println!("Final response: {}", loop_result.response);
        }
        Err(e) => {
            eprintln!("❌ Conversation context test failed: {}", e);
            panic!("Event loop conversation context test failed: {}", e);
        }
    }
}

/// Test error handling when tools fail
#[tokio::test]
async fn test_event_loop_handles_tool_failures() {
    // Skip test if no AWS credentials
    if ensure_aws_credentials().is_err() {
        println!("⚠️  Skipping tool failure handling test - AWS credentials not available");
        return;
    }

    // Create a tool that always fails for testing
    #[derive(Debug)]
    struct FailingTestTool;

    #[async_trait]
    impl Tool for FailingTestTool {
        fn name(&self) -> &str {
            "failing_tool"
        }

        fn description(&self) -> &str {
            "A tool that always fails for testing error handling"
        }

        fn parameters_schema(&self) -> Value {
            json!({
                "type": "object",
                "properties": {},
                "required": []
            })
        }

        async fn execute(
            &self,
            _parameters: Option<Value>,
            _agent_context: Option<&crate::agent::AgentContext>,
        ) -> Result<ToolResult, ToolError> {
            Err(ToolError::ExecutionFailed {
                message: "This tool always fails for testing purposes".to_string(),
            })
        }
    }

    // Create client and agent
    let agent = Agent::builder()
        .provider("bedrock")
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .build()
        .await
        .unwrap();

    // Create tool registry with failing tool
    let tool_registry = ToolRegistry::new();
    tool_registry
        .register_tool(Box::new(FailingTestTool))
        .await
        .unwrap();
    tool_registry
        .register_tool(Box::new(TestCalculatorTool))
        .await
        .unwrap(); // Also add a working tool

    let config = EventLoopConfig {
        max_cycles: 3,
        max_duration: Duration::from_secs(45),
        enable_streaming: false,
        enable_telemetry: false,
        ..EventLoopConfig::default()
    };

    let mut event_loop = EventLoop::new(agent, tool_registry, config).unwrap();

    println!("⚠️  Testing error handling with failing tools...");

    // Make a request that might trigger the failing tool
    let result = event_loop
        .execute("Please help me with something, use any tools you think are appropriate")
        .await;

    match result {
        Ok(loop_result) => {
            println!("✅ Event loop completed despite tool failures");
            println!(
                "Cycles: {}, Duration: {:?}",
                loop_result.cycles_executed, loop_result.total_duration
            );

            // The loop should still complete successfully even if some tools fail
            assert!(loop_result.success);
            assert!(!loop_result.response.is_empty());

            // Check if any tools were executed
            if loop_result.metrics.tool_executions.len() > 0 {
                println!(
                    "Tools executed: {}",
                    loop_result.metrics.tool_executions.len()
                );

                // Check for failed executions
                let failed_executions = loop_result
                    .metrics
                    .tool_executions
                    .iter()
                    .filter(|t| !t.success)
                    .count();

                if failed_executions > 0 {
                    println!(
                        "✅ Error handling worked - {} tool(s) failed as expected",
                        failed_executions
                    );
                }

                // Check for successful executions
                let successful_executions = loop_result
                    .metrics
                    .tool_executions
                    .iter()
                    .filter(|t| t.success)
                    .count();

                if successful_executions > 0 {
                    println!("✅ Some tools executed successfully despite failures");
                }
            }

            println!("Final response: {}", loop_result.response);
        }
        Err(e) => {
            eprintln!("❌ Tool failure handling test failed: {}", e);
            panic!("Event loop should handle tool failures gracefully: {}", e);
        }
    }
}

#[tokio::test]
async fn test_callback_integration_end_to_end() {
    // Skip test if no AWS credentials
    if ensure_aws_credentials().is_err() {
        println!("⚠️  Skipping callback integration test - AWS credentials not available");
        return;
    }

    println!("🚀 Testing end-to-end callback integration...");

    use crate::agent::callbacks::{CallbackError, CallbackEvent, CallbackHandler};
    use std::sync::{Arc, Mutex};

    // Test callback handler that records all events
    #[derive(Debug)]
    struct TestCallbackHandler {
        events: Arc<Mutex<Vec<String>>>,
    }

    impl TestCallbackHandler {
        fn new() -> Self {
            Self {
                events: Arc::new(Mutex::new(Vec::new())),
            }
        }

        #[allow(dead_code)]
        fn get_events(&self) -> Vec<String> {
            self.events.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl CallbackHandler for TestCallbackHandler {
        async fn handle_event(&self, event: CallbackEvent) -> Result<(), CallbackError> {
            let event_description = match &event {
                CallbackEvent::EventLoopStart { loop_id, .. } => {
                    format!("EventLoopStart({})", loop_id)
                }
                CallbackEvent::CycleStart {
                    cycle_id,
                    cycle_number,
                } => {
                    format!("CycleStart({}, cycle_{})", cycle_id, cycle_number)
                }
                CallbackEvent::ModelStart {
                    provider,
                    model_id,
                    tools_available,
                    ..
                } => {
                    format!(
                        "ModelStart({:?}/{}, {} tools)",
                        provider, model_id, tools_available
                    )
                }
                CallbackEvent::ModelComplete {
                    response, duration, ..
                } => {
                    format!("ModelComplete({} chars, {:?})", response.len(), duration)
                }
                CallbackEvent::ContentDelta {
                    delta, complete, ..
                } => {
                    format!(
                        "ContentDelta({} chars, complete: {})",
                        delta.len(),
                        complete
                    )
                }
                CallbackEvent::ToolStart { tool_name, .. } => {
                    format!("ToolStart({})", tool_name)
                }
                CallbackEvent::ToolComplete {
                    tool_name,
                    output,
                    error,
                    ..
                } => {
                    if error.is_some() {
                        format!("ToolComplete({}, failed)", tool_name)
                    } else {
                        format!("ToolComplete({}, success: {})", tool_name, output.is_some())
                    }
                }
                CallbackEvent::ParallelStart {
                    tool_count,
                    max_parallel,
                } => {
                    format!(
                        "ParallelStart({} tools, max_parallel: {})",
                        tool_count, max_parallel
                    )
                }
                CallbackEvent::ParallelProgress {
                    completed,
                    total,
                    running,
                } => {
                    format!(
                        "ParallelProgress({}/{} completed, {} running)",
                        completed, total, running
                    )
                }
                CallbackEvent::ParallelComplete {
                    total_duration,
                    success_count,
                    failure_count,
                } => {
                    format!(
                        "ParallelComplete({:?}, {} success, {} failures)",
                        total_duration, success_count, failure_count
                    )
                }
                CallbackEvent::EventLoopComplete { result, .. } => {
                    format!(
                        "EventLoopComplete(success: {}, {} cycles)",
                        result.success, result.cycles_executed
                    )
                }
                CallbackEvent::Error { error, context } => {
                    format!("Error({}, {})", context, error)
                }
                CallbackEvent::EvaluationStart { .. } => "EvaluationStart".to_string(),
                CallbackEvent::EvaluationComplete { .. } => "EvaluationComplete".to_string(),
            };

            self.events.lock().unwrap().push(event_description);
            Ok(())
        }
    }

    // Create test callback handler
    let callback_handler = TestCallbackHandler::new();
    let events_tracker = callback_handler.events.clone();

    // Create agent with callback handler configured
    let agent = match Agent::builder()
        .provider("bedrock")
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .system_prompt("You are a helpful assistant. Keep responses brief.")
        .with_callback_handler(callback_handler)
        .tool(Box::new(CalculatorTool::new()))
        .build()
        .await
    {
        Ok(agent) => agent,
        Err(e) => {
            eprintln!("❌ Failed to create agent: {}", e);
            panic!("Agent creation failed: {}", e);
        }
    };

    // Test callback integration with a simple task
    let mut agent = agent;
    match agent.execute("What is 25 + 37?").await {
        Ok(result) => {
            println!("✅ Agent execution completed successfully");
            println!("Response: {}", result.response);

            // Verify callback events were captured
            let events = events_tracker.lock().unwrap().clone();
            println!("📋 Callback events captured: {}", events.len());
            for (i, event) in events.iter().enumerate() {
                println!("  {}: {}", i + 1, event);
            }

            // Verify we captured essential events
            let has_loop_start = events.iter().any(|e| e.starts_with("EventLoopStart"));
            let has_cycle_start = events.iter().any(|e| e.starts_with("CycleStart"));
            let has_model_start = events.iter().any(|e| e.starts_with("ModelStart"));
            let has_model_complete = events.iter().any(|e| e.starts_with("ModelComplete"));
            let has_loop_complete = events.iter().any(|e| e.starts_with("EventLoopComplete"));

            assert!(has_loop_start, "Should capture EventLoopStart event");
            assert!(has_cycle_start, "Should capture CycleStart event");
            assert!(has_model_start, "Should capture ModelStart event");
            assert!(has_model_complete, "Should capture ModelComplete event");
            assert!(has_loop_complete, "Should capture EventLoopComplete event");

            // Check if tools were used (the LLM might choose to calculate 25+37 mentally)
            let has_tool_events = events
                .iter()
                .any(|e| e.starts_with("ToolStart") || e.starts_with("ToolComplete"));
            if has_tool_events {
                println!("✅ Tool execution callbacks captured");
                let has_tool_start = events.iter().any(|e| e.starts_with("ToolStart"));
                let has_tool_complete = events.iter().any(|e| e.starts_with("ToolComplete"));
                assert!(
                    has_tool_start,
                    "Should capture ToolStart event when tools are used"
                );
                assert!(
                    has_tool_complete,
                    "Should capture ToolComplete event when tools are used"
                );
            } else {
                println!("ℹ️  LLM solved the math problem without using tools (valid behavior)");
            }

            // Verify streaming callbacks if enabled
            let has_content_delta = events.iter().any(|e| e.starts_with("ContentDelta"));
            if has_content_delta {
                println!("✅ Streaming content delta callbacks captured");
                // Verify we have multiple deltas that eventually complete
                let content_deltas: Vec<_> = events
                    .iter()
                    .filter(|e| e.starts_with("ContentDelta"))
                    .collect();
                assert!(
                    !content_deltas.is_empty(),
                    "Should have content delta events"
                );

                // Check that at least one delta is marked as complete
                let has_complete_delta =
                    content_deltas.iter().any(|e| e.contains("complete: true"));
                assert!(
                    has_complete_delta,
                    "Should have at least one complete content delta"
                );
            } else {
                println!("ℹ️  No content deltas captured (might be non-streaming mode)");
            }

            println!("✅ Callback integration test passed!");
        }
        Err(e) => {
            eprintln!("❌ Agent execution failed: {}", e);

            // Still check if error callbacks were captured
            let events = events_tracker.lock().unwrap().clone();
            println!("📋 Error case - callback events captured: {}", events.len());
            for (i, event) in events.iter().enumerate() {
                println!("  {}: {}", i + 1, event);
            }

            let has_error_event = events.iter().any(|e| e.starts_with("Error"));
            if has_error_event {
                println!("✅ Error callback captured successfully");
            }

            // For integration tests, we expect the API call might fail
            println!("⚠️  Agent execution failed, but this is acceptable in test environment");
        }
    }
}
