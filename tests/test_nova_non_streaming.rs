use std::env;
use stood::tools::builtin::CalculatorTool;
use stood::{
    agent::{Agent, LogLevel},
    tool,
};

#[tool]
async fn get_weather(location: String) -> Result<String, String> {
    let weather_info = format!(
        "The weather in {} is sunny, 72°F with light winds.",
        location
    );
    Ok(weather_info)
}

#[tokio::test]
async fn test_nova_non_streaming() -> Result<(), Box<dyn std::error::Error>> {
    println!("🛠️  Nova Non-Streaming Testing");
    println!("=============================");

    // Force trace logging
    env::set_var("RUST_LOG", "stood=trace");
    tracing_subscriber::fmt()
        .with_env_filter("stood=trace")
        .with_target(true)
        .try_init()
        .ok(); // Ignore if already initialized

    println!("✅ Trace logging enabled");

    // Configure providers
    use stood::llm::registry::{ProviderRegistry, PROVIDER_REGISTRY};
    ProviderRegistry::configure().await?;

    // Check Bedrock availability
    if !PROVIDER_REGISTRY
        .is_configured(stood::llm::traits::ProviderType::Bedrock)
        .await
    {
        println!("❌ AWS Bedrock not available");
        return Err("AWS Bedrock not available".into());
    }

    let tools = vec![
        get_weather(),
        Box::new(CalculatorTool::new()) as Box<dyn stood::tools::Tool>,
    ];

    // Create agent with Nova Lite - DISABLE STREAMING
    let mut agent = Agent::builder()
        .provider("bedrock")
        .model("us.amazon.nova-lite-v1:0")
        .system_prompt("You are a helpful assistant.")
        .with_streaming(false) // DISABLE STREAMING
        .tools(tools)
        .with_log_level(LogLevel::Trace)
        .build()
        .await?;

    println!("🤖 Agent created with Nova Lite (non-streaming)");

    // Simple test
    println!("\n=== Simple Test ===");
    let question = "What is 2+3?";
    println!("Question: {}", question);

    let result = agent.execute(question).await?;
    println!("Agent: {}", result.response);
    println!("Duration: {:?}", result.duration);
    println!("Used tools: {}", result.used_tools);

    // Verify we got a response
    assert!(!result.response.is_empty(), "Nova should return a response");
    println!("✅ Nova non-streaming test passed!");

    Ok(())
}
