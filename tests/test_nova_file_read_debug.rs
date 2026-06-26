use std::env;
use std::fs;
use stood::{
    agent::{Agent, LogLevel},
    tools::builtin::FileReadTool,
};

#[tokio::test]
async fn test_nova_file_read_debug() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Nova File Read Debug Test");
    println!("===========================");

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
    let temp_path = temp_dir.join("nova_file_read_test.txt");
    let test_content = "Hello from Nova file read test!";
    fs::write(&temp_path, test_content)?;
    println!("📁 Created test file: {}", temp_path.display());

    // Create agent with Nova
    let mut agent = Agent::builder()
        .provider("bedrock")
        .model("us.amazon.nova-lite-v1:0")
        .system_prompt(
            "You are a helpful assistant. When asked to read a file, use the file_read tool.",
        )
        .tool(Box::new(FileReadTool::new()))
        .with_log_level(LogLevel::Debug)
        .build()
        .await?;

    println!("\n🤖 Agent created with Nova Lite");

    // Test 1: Direct path request
    println!("\n=== Test 1: Direct Path Request ===");
    let prompt1 = format!(
        "Please read the file at '{}' and tell me what it contains.",
        temp_path.display()
    );
    println!("Prompt: {}", prompt1);

    let response1 = agent.execute(&prompt1).await?;
    println!("\nResponse: {}", response1.response);
    println!("Used tools: {}", response1.used_tools);
    println!("Tools called: {:?}", response1.tools_called);

    // Test 2: Explicit tool instruction
    println!("\n=== Test 2: Explicit Tool Instruction ===");
    let prompt2 = format!(
        "Use the file_read tool with path '{}' to read the file content.",
        temp_path.display()
    );
    println!("Prompt: {}", prompt2);

    let response2 = agent.execute(&prompt2).await?;
    println!("\nResponse: {}", response2.response);
    println!("Used tools: {}", response2.used_tools);
    println!("Tools called: {:?}", response2.tools_called);

    // Clean up
    fs::remove_file(&temp_path).ok();

    // Check if either test succeeded
    let test1_success = response1.used_tools && response1.response.contains(test_content);
    let test2_success = response2.used_tools && response2.response.contains(test_content);

    if !test1_success && !test2_success {
        println!("\n❌ Nova failed to read file in both tests");
        return Err("Nova file read failed".into());
    }

    println!("\n✅ Nova file read test completed");
    Ok(())
}

#[tokio::test]
async fn test_claude_file_read_control() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Claude File Read Control Test");
    println!("================================");

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
    let temp_path = temp_dir.join("claude_file_read_test.txt");
    let test_content = "Hello from Claude file read test!";
    fs::write(&temp_path, test_content)?;
    println!("📁 Created test file: {}", temp_path.display());

    // Create agent with Claude
    let mut agent = Agent::builder()
        .provider("bedrock")
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .system_prompt(
            "You are a helpful assistant. When asked to read a file, use the file_read tool.",
        )
        .tool(Box::new(FileReadTool::new()))
        .with_log_level(LogLevel::Debug)
        .build()
        .await?;

    println!("\n🤖 Agent created with Claude Haiku 4.5");

    // Test: Direct path request (same as Nova test 1)
    println!("\n=== Claude Direct Path Request ===");
    let prompt = format!(
        "Please read the file at '{}' and tell me what it contains.",
        temp_path.display()
    );
    println!("Prompt: {}", prompt);

    let response = agent.execute(&prompt).await?;
    println!("\nResponse: {}", response.response);
    println!("Used tools: {}", response.used_tools);
    println!("Tools called: {:?}", response.tools_called);

    // Clean up
    fs::remove_file(&temp_path).ok();

    // Check success
    if !response.used_tools || !response.response.contains(test_content) {
        println!("\n❌ Claude failed to read file");
        return Err("Claude file read failed".into());
    }

    println!("\n✅ Claude file read test completed");
    Ok(())
}
