use std::env;
use stood::{
    agent::{Agent, LogLevel},
    tool,
};

#[tool]
async fn simple_calculator(a: f64, b: f64, operation: String) -> Result<f64, String> {
    println!(
        "🔧 Tool called: simple_calculator({}, {}, {})",
        a, b, operation
    );
    match operation.as_str() {
        "add" => Ok(a + b),
        "multiply" => Ok(a * b),
        _ => Err(format!("Unknown operation: {}", operation)),
    }
}

#[tokio::test]
async fn test_nova_tool_execution_minimal() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Nova Tool Execution Minimal Test");
    println!("===================================");

    // Disable telemetry completely to avoid timeout - use correct env var
    env::set_var("OTEL_ENABLED", "false");

    // Force trace logging
    env::set_var("RUST_LOG", "stood=trace");
    tracing_subscriber::fmt()
        .with_env_filter("stood=trace")
        .with_target(true)
        .try_init()
        .ok();

    // Configure providers
    use stood::llm::registry::{ProviderRegistry, PROVIDER_REGISTRY};
    ProviderRegistry::configure().await?;

    // Check Bedrock availability
    if !PROVIDER_REGISTRY
        .is_configured(stood::llm::traits::ProviderType::Bedrock)
        .await
    {
        eprintln!("❌ AWS Bedrock not available - skipping test");
        return Ok(());
    }

    println!("✅ Bedrock provider configured");

    let tools = vec![simple_calculator()];

    // Create agent with Nova - non-streaming to simplify
    let mut agent = Agent::builder()
        .provider("bedrock")
        .model("us.amazon.nova-lite-v1:0")
        .system_prompt("You are a helpful assistant with access to tools. When asked to do math, use the simple_calculator tool.")
        .with_streaming(false)
        .tools(tools)
        .with_log_level(LogLevel::Trace)
        .build()
        .await?;

    println!("🤖 Agent created with Nova Lite (non-streaming)");

    // Simple tool request
    println!("\n=== Tool Execution Test ===");
    let question = "Please calculate 5 + 3 using the simple_calculator tool with operation 'add'.";
    println!("Question: {}", question);
    println!("Expected: Tool should be called and return 8");

    let result = agent.execute(question).await?;

    println!("\n=== Results ===");
    println!("Response: {}", result.response);
    println!("Used tools: {}", result.used_tools);
    println!("Tools called: {:?}", result.tools_called);
    println!("Execution cycles: {}", result.execution.cycles);

    // Assertions
    assert!(result.used_tools, "Nova should have used tools");
    assert!(
        !result.tools_called.is_empty(),
        "Nova should have called at least one tool"
    );
    assert!(
        result
            .tools_called
            .contains(&"simple_calculator".to_string()),
        "Nova should have called simple_calculator, but called: {:?}",
        result.tools_called
    );
    assert!(
        result.response.contains("8"),
        "Response should contain the result '8', but was: {}",
        result.response
    );

    println!("✅ Nova tool execution test passed!");
    Ok(())
}

#[tokio::test]
async fn test_claude_tool_execution_control() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Claude Tool Execution Control Test");
    println!("====================================");

    // Disable telemetry completely to avoid timeout - use correct env var
    env::set_var("OTEL_ENABLED", "false");

    // Same setup but with Claude
    env::set_var("RUST_LOG", "stood=trace");
    tracing_subscriber::fmt()
        .with_env_filter("stood=trace")
        .with_target(true)
        .try_init()
        .ok();

    use stood::llm::registry::{ProviderRegistry, PROVIDER_REGISTRY};
    ProviderRegistry::configure().await?;

    if !PROVIDER_REGISTRY
        .is_configured(stood::llm::traits::ProviderType::Bedrock)
        .await
    {
        eprintln!("❌ AWS Bedrock not available - skipping test");
        return Ok(());
    }

    let tools = vec![simple_calculator()];

    // Create agent with Claude - non-streaming
    let mut agent = Agent::builder()
        .provider("bedrock")
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .system_prompt("You are a helpful assistant with access to tools. When asked to do math, use the simple_calculator tool.")
        .with_streaming(false)
        .tools(tools)
        .with_log_level(LogLevel::Trace)
        .build()
        .await?;

    println!("🤖 Agent created with Claude Haiku 4.5 (non-streaming)");

    // Same tool request
    println!("\n=== Tool Execution Test ===");
    let question = "Please calculate 5 + 3 using the simple_calculator tool with operation 'add'.";
    println!("Question: {}", question);
    println!("Expected: Tool should be called and return 8");

    let result = agent.execute(question).await?;

    println!("\n=== Results ===");
    println!("Response: {}", result.response);
    println!("Used tools: {}", result.used_tools);
    println!("Tools called: {:?}", result.tools_called);
    println!("Execution cycles: {}", result.execution.cycles);

    // Same assertions
    assert!(result.used_tools, "Claude should have used tools");
    assert!(
        !result.tools_called.is_empty(),
        "Claude should have called at least one tool"
    );
    assert!(
        result
            .tools_called
            .contains(&"simple_calculator".to_string()),
        "Claude should have called simple_calculator, but called: {:?}",
        result.tools_called
    );
    assert!(
        result.response.contains("8"),
        "Response should contain the result '8', but was: {}",
        result.response
    );

    println!("✅ Claude tool execution test passed!");
    Ok(())
}
