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

#[tool]
async fn calculate_percentage(value: f64, percentage: f64) -> Result<f64, String> {
    if percentage < 0.0 || percentage > 100.0 {
        return Err("Percentage must be between 0 and 100".to_string());
    }
    Ok(value * percentage / 100.0)
}

#[tokio::test]
async fn test_nova_debug() -> Result<(), Box<dyn std::error::Error>> {
    println!("🛠️  Nova Testing with Trace Logging");
    println!("==================================");

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

    let provider = PROVIDER_REGISTRY
        .get_provider(stood::llm::traits::ProviderType::Bedrock)
        .await?;
    let health = provider.health_check().await;
    match health {
        Ok(status) if status.healthy => {
            println!("✅ AWS Bedrock connected successfully!");
        }
        Ok(status) => {
            println!(
                "⚠️  AWS Bedrock connected but unhealthy: {:?}",
                status.error
            );
        }
        Err(e) => {
            println!("❌ AWS Bedrock connection failed: {}", e);
            return Err(e.into());
        }
    }

    let tools = vec![
        get_weather(),
        calculate_percentage(),
        Box::new(CalculatorTool::new()) as Box<dyn stood::tools::Tool>,
    ];

    // Create agent with Nova Lite
    let mut agent = Agent::builder()
        .provider("bedrock")
        .model("us.amazon.nova-lite-v1:0")
        .system_prompt("You are a helpful assistant.")
        .with_streaming(true)
        .tools(tools)
        .with_log_level(LogLevel::Trace)
        .build()
        .await?;

    println!("🤖 Agent created with Nova Lite");

    // Simple test
    println!("\n=== Simple Test ===");
    let question = "What is 2+3?";
    println!("Question: {}", question);

    let result = agent.execute(question).await?;
    println!("Agent: {}", result.response);
    println!("Duration: {:?}", result.duration);
    println!("Used tools: {}", result.used_tools);

    // Tool test
    println!("\n=== Tool Test ===");
    let task = "Calculate 15% of 67 using the calculator.";
    println!("Task: {}", task);

    let result = agent.execute(task).await?;
    println!("Agent: {}", result.response);
    println!("Duration: {:?}", result.duration);
    println!("Used tools: {}", result.used_tools);
    if result.used_tools {
        println!("Tools called: {}", result.tools_called.join(", "));
    }

    Ok(())
}
