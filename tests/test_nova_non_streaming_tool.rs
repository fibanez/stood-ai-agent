use std::env;
use std::fs;
use stood::{
    agent::{Agent, LogLevel},
    tools::builtin::FileReadTool,
};

#[tokio::test]
async fn test_nova_non_streaming_tool_execution() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Nova Non-Streaming Tool Execution Test");
    println!("========================================");

    // Disable telemetry
    env::set_var("OTEL_ENABLED", "false");
    env::set_var("RUST_LOG", "stood=debug");
    tracing_subscriber::fmt()
        .with_env_filter("stood=debug")
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

    // Create test file
    let temp_dir = std::env::temp_dir();
    let temp_path = temp_dir.join("nova_non_streaming_test.txt");
    let test_content = "Content for Nova non-streaming test";
    fs::write(&temp_path, test_content)?;
    println!("📁 Created test file: {}", temp_path.display());

    // Create agent with Nova - disable streaming
    let mut agent = Agent::builder()
        .provider("bedrock")
        .model("us.amazon.nova-micro-v1:0")
        .system_prompt(
            "You are a helpful assistant. When asked to read a file, use the file_read tool.",
        )
        .tool(Box::new(FileReadTool::new()))
        .with_log_level(LogLevel::Debug)
        .with_streaming(false) // Disable streaming
        .build()
        .await?;

    println!("\n🤖 Agent created with Nova Micro (streaming disabled)");

    // Test with explicit tool instruction
    println!("\n=== Testing Nova Non-Streaming Tool Execution ===");
    let prompt = format!(
        "Use the file_read tool to read the file at path '{}'",
        temp_path.display()
    );
    println!("Prompt: {}", prompt);

    let response = agent.execute(&prompt).await?;

    println!("\n📊 Execution Results:");
    println!("Success: {}", response.success);
    println!("Used tools: {}", response.used_tools);
    println!("Tools called: {:?}", response.tools_called);
    println!("Response: {}", response.response);

    if let Some(error) = &response.error {
        println!("Error: {}", error);
    }

    // Clean up
    fs::remove_file(&temp_path).ok();

    // Verify the test results
    if !response.success {
        println!("\n❌ Nova non-streaming execution failed");
        return Err(format!("Execution failed: {}", response.error.unwrap_or_default()).into());
    }

    if !response.used_tools {
        println!("\n❌ Nova agent did not use tools");
        return Err("Nova agent should have used the file_read tool".into());
    }

    if !response.response.contains(test_content) {
        println!("\n❌ Response doesn't contain file content");
        println!("Expected content: '{}'", test_content);
        println!("Actual response: '{}'", response.response);
        return Err("Response should contain the file content".into());
    }

    println!("\n✅ Nova non-streaming tool execution test completed successfully");
    Ok(())
}
