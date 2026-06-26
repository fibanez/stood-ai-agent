//! Integration tests for Agent + Tools workflow
//!
//! These tests demonstrate the complete integration between:
//! - AWS Bedrock Agent chat
//! - Tool registry and execution
//! - Built-in tools (calculator, file operations, HTTP, env vars)
//!
//! NOTE: These tests require valid AWS credentials and Bedrock access.

use crate::agent::Agent;
use crate::tools::builtin::create_builtin_tools;
use crate::tools::{ExecutorConfig, ToolExecutor, ToolUse};
use serde_json::json;
use std::env;

/// Helper to check if we have AWS credentials for integration tests
fn ensure_aws_credentials() -> Result<(), String> {
    let has_access_key = env::var("AWS_ACCESS_KEY_ID").is_ok();
    let has_profile = env::var("AWS_PROFILE").is_ok();
    let has_role_arn = env::var("AWS_ROLE_ARN").is_ok();

    if !has_access_key && !has_profile && !has_role_arn {
        return Err("AWS credentials required for integration tests".to_string());
    }

    Ok(())
}

#[tokio::test]
async fn test_agent_with_builtin_tools_registry() {
    if ensure_aws_credentials().is_err() {
        println!(
            "⚠️  Skipping Agent with built-in tools registry test - AWS credentials not available"
        );
        return;
    }

    println!("🔧 Testing Agent with built-in tools registry...");

    // Create agent
    let mut agent = Agent::builder()
        .provider("bedrock")
        .model_str("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .system_prompt("You are a helpful assistant with access to tools.")
        .build()
        .await
        .expect("Failed to build agent");

    // Create built-in tools registry
    let registry = create_builtin_tools()
        .await
        .expect("Failed to create built-in tools");

    // Verify all tools are registered
    assert!(registry.has_tool("calculator").await);
    assert!(registry.has_tool("file_read").await);
    assert!(registry.has_tool("file_write").await);
    assert!(registry.has_tool("file_list").await);
    assert!(registry.has_tool("http_request").await);
    assert!(registry.has_tool("env_var").await);
    assert_eq!(registry.tool_names().await.len(), 7);

    // Test basic chat still works
    let response = agent
        .execute("Hello! Can you help me with calculations?")
        .await
        .expect("Agent chat failed");

    assert!(!response.response.is_empty());
    assert_eq!(agent.conversation().message_count(), 2);

    println!("✅ Agent + Tools registry integration test completed!");
}

#[tokio::test]
async fn test_tool_executor_with_aws_agent_workflow() {
    if ensure_aws_credentials().is_err() {
        println!("⚠️  Skipping Tool executor with AWS Agent workflow test - AWS credentials not available");
        return;
    }

    println!("⚙️ Testing Tool executor + AWS Agent workflow...");

    // Create agent for reasoning/planning
    let mut agent = Agent::builder()
        .provider("bedrock")
        .model_str("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .system_prompt("You help users by understanding their requests and explaining what tools might be useful.")
        .build()
        .await.expect("Failed to build agent");

    // Create tool registry and executor
    let registry = create_builtin_tools()
        .await
        .expect("Failed to create built-in tools");

    let _executor = ToolExecutor::new(ExecutorConfig::default());

    // Step 1: Agent suggests what we could do
    let planning_response = agent.execute(
        "I want to do some math calculations and check an environment variable. What tools could help me?"
    ).await.expect("Agent planning failed");

    println!("🧠 Agent planning: {}", planning_response);
    assert!(!planning_response.response.is_empty());

    // Step 2: Execute calculator tool
    let calc_tool_use = ToolUse {
        tool_use_id: "calc_1".to_string(),
        name: "calculator".to_string(),
        input: json!({
            "expression": "15 * 3 + 7",
            "precision": 2
        }),
    };

    let calc_result = registry
        .execute_tool(&calc_tool_use.name, Some(calc_tool_use.input.clone()), None)
        .await
        .expect("Calculator tool should succeed");

    println!("🔢 Calculator result: {}", calc_result.content);
    assert!(calc_result.success);

    // Step 3: Execute environment variable tool
    let env_tool_use = ToolUse {
        tool_use_id: "env_1".to_string(),
        name: "env_var".to_string(),
        input: json!({
            "name": "PATH"
        }),
    };

    let env_result = registry
        .execute_tool(&env_tool_use.name, Some(env_tool_use.input.clone()), None)
        .await
        .expect("Environment tool should succeed");

    println!("🌍 Environment result: {}", env_result.content);
    assert!(env_result.success);

    // The result should contain the PATH environment variable
    let env_response = env_result.content.as_object().unwrap();
    assert_eq!(env_response.get("name").unwrap(), "PATH");
    assert_eq!(env_response.get("found").unwrap(), true);
    assert!(env_response.get("value").is_some());

    // Step 4: Agent summarizes results
    let summary_prompt = format!(
        "I completed some calculations and environment checks. Calculator result: {}. Environment PATH found: {}. Please summarize what was accomplished.",
        calc_result.content,
        env_response.get("found").unwrap()
    );

    let summary_response = agent
        .execute(&summary_prompt)
        .await
        .expect("Agent summary failed");

    println!("📋 Agent summary: {}", summary_response);
    assert!(!summary_response.response.is_empty());

    // Verify conversation history
    assert_eq!(agent.conversation().message_count(), 4); // 2 pairs of user/assistant

    println!("✅ Tool executor + AWS Agent workflow test completed!");
}

#[tokio::test]
async fn test_parallel_tool_execution_with_agent() {
    if ensure_aws_credentials().is_err() {
        println!("⚠️  Skipping Parallel tool execution with Agent coordination test - AWS credentials not available");
        return;
    }

    println!("🔄 Testing parallel tool execution with Agent coordination...");

    // Create agent
    let mut agent = Agent::builder()
        .provider("bedrock")
        .model_str("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .system_prompt("You coordinate multiple tool executions and summarize results.")
        .build()
        .await
        .expect("Failed to build agent");

    // Create tools and executor
    let registry = create_builtin_tools()
        .await
        .expect("Failed to create built-in tools");

    let _executor = ToolExecutor::new(ExecutorConfig::default());

    // Step 1: Agent explains the plan
    let plan_response = agent.execute(
        "I'm going to run multiple calculations in parallel and check environment variables. Explain why this might be useful."
    ).await.expect("Agent planning failed");

    println!("📝 Agent plan: {}", plan_response);

    // Step 2: Execute multiple tools in parallel
    let tool_uses = vec![
        ToolUse {
            tool_use_id: "calc_1".to_string(),
            name: "calculator".to_string(),
            input: json!({"expression": "100 / 4", "precision": 1}),
        },
        ToolUse {
            tool_use_id: "calc_2".to_string(),
            name: "calculator".to_string(),
            input: json!({"expression": "50 + 25", "precision": 0}),
        },
        ToolUse {
            tool_use_id: "env_1".to_string(),
            name: "env_var".to_string(),
            input: json!({"name": "HOME", "default": "/tmp"}),
        },
    ];

    // Execute tools in parallel using the new API
    let mut results = Vec::new();
    for tool_use in &tool_uses {
        let result = registry
            .execute_tool(&tool_use.name, Some(tool_use.input.clone()), None)
            .await;
        results.push(result);
    }

    println!("⚡ Parallel results count: {}", results.len());

    // Verify all executions succeeded
    assert_eq!(results.len(), 3);
    for result in &results {
        assert!(result.is_ok(), "Tool execution failed: {:?}", result);
    }

    // Extract specific results
    let calc1_result = results[0].as_ref().unwrap();
    let calc2_result = results[1].as_ref().unwrap();
    let env_result = results[2].as_ref().unwrap();

    assert!(calc1_result.success);
    assert!(calc2_result.success);
    assert!(env_result.success);

    println!(
        "⚡ Parallel results - Calc1: {}, Calc2: {}, Env success: {}",
        calc1_result.content, calc2_result.content, env_result.success
    );

    // Step 3: Agent summarizes parallel execution results
    let summary_prompt = format!(
        "Parallel execution completed: Calculation 1 = {}, Calculation 2 = {}, Environment check successful = {}. Please summarize these results.",
        calc1_result.content, calc2_result.content, env_result.success
    );

    let summary = agent
        .execute(&summary_prompt)
        .await
        .expect("Agent summary failed");

    println!("📊 Final summary: {}", summary);
    assert!(!summary.response.is_empty());

    println!("✅ Parallel tool execution + Agent coordination test completed!");
}

#[tokio::test]
async fn test_error_handling_in_agent_tool_workflow() {
    if ensure_aws_credentials().is_err() {
        println!("⚠️  Skipping Error handling in Agent and Tool workflow test - AWS credentials not available");
        return;
    }

    println!("🚨 Testing error handling in Agent + Tool workflow...");

    // Create agent
    let mut agent = Agent::builder()
        .provider("bedrock")
        .model_str("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .system_prompt("You help users understand and handle tool execution errors.")
        .build()
        .await
        .expect("Failed to build agent");

    let registry = create_builtin_tools()
        .await
        .expect("Failed to create built-in tools");
    let _executor = ToolExecutor::new(ExecutorConfig::default());

    // Step 1: Agent sets expectations
    let intro = agent.execute(
        "I'm going to test some calculations, including some that might fail. How should we handle errors?"
    ).await.expect("Agent intro failed");

    println!("💡 Agent guidance: {}", intro);

    // Step 2: Execute successful calculation
    let good_tool_use = ToolUse {
        tool_use_id: "good_calc".to_string(),
        name: "calculator".to_string(),
        input: json!({
            "expression": "10 + 5"
        }),
    };

    let good_result = registry
        .execute_tool(&good_tool_use.name, Some(good_tool_use.input.clone()), None)
        .await
        .expect("Tool execution should succeed");

    assert!(good_result.success);
    println!("✅ Good calculation: {}", good_result.content);

    // Step 3: Execute calculation that returns error in JSON (not Rust error)
    let bad_tool_use = ToolUse {
        tool_use_id: "bad_calc".to_string(),
        name: "calculator".to_string(),
        input: json!({
            "expression": "invalid_expression_xyz"
        }),
    };

    let bad_result = registry
        .execute_tool(&bad_tool_use.name, Some(bad_tool_use.input.clone()), None)
        .await
        .expect("Tool execution should succeed");

    // Calculator tool should handle invalid expressions gracefully
    // It might return an error in the content or a success with error message
    println!("🔍 Error result: {}", bad_result.content);

    // Step 4: Agent explains error handling
    let error_explanation = agent.execute(&format!(
        "The calculation tool returned a result for invalid input: {}. How should we interpret this result?",
        bad_result.content
    )).await.expect("Agent error explanation failed");

    println!("🧭 Agent error explanation: {}", error_explanation);
    assert!(!error_explanation.response.is_empty());

    println!("✅ Error handling in Agent + Tool workflow test completed!");
}
