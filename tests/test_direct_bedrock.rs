use std::env;
use stood::llm::providers::bedrock::BedrockProvider;
use stood::llm::traits::{ChatConfig, LlmProvider, Tool};
use stood::types::{ContentBlock, Message, MessageRole, Messages};

#[tokio::test]
async fn test_direct_nova_vs_claude() -> Result<(), Box<dyn std::error::Error>> {
    println!("🛠️  Direct Bedrock Provider Testing");
    println!("==================================");

    // Force trace logging for Bedrock provider only
    env::set_var("RUST_LOG", "stood::llm::providers::bedrock=trace");
    tracing_subscriber::fmt()
        .with_env_filter("stood::llm::providers::bedrock=trace")
        .with_target(true)
        .try_init()
        .ok(); // Ignore if already initialized

    println!("✅ Trace logging enabled for Bedrock provider only");

    // Create Bedrock provider
    let provider = BedrockProvider::new(None).await?;

    println!("✅ Bedrock provider created");

    // Test simple message
    let messages = Messages {
        messages: vec![Message::user("What is 2+3?")],
        system_prompt: None,
    };

    let config = ChatConfig {
        max_tokens: Some(100),
        temperature: Some(0.1),
        ..Default::default()
    };

    // Test tools
    let tools = vec![Tool {
        name: "calculator".to_string(),
        description: "Basic calculator operations".to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"]
                },
                "a": { "type": "number" },
                "b": { "type": "number" }
            },
            "required": ["operation", "a", "b"]
        }),
    }];

    // Test Nova Lite
    println!("\n=== Testing Nova Lite ===");
    let nova_model = "us.amazon.nova-micro-v1:0";
    println!("🤖 Calling Nova with model: {}", nova_model);

    match provider
        .chat_with_tools(nova_model, &messages, &tools[..], &config)
        .await
    {
        Ok(response) => {
            println!("✅ Nova response received:");
            println!("   Content: {}", response.content);
            println!("   Tool calls: {}", response.tool_calls.len());
        }
        Err(e) => {
            println!("❌ Nova failed: {}", e);
        }
    }

    // Test Claude 3.5 Haiku
    println!("\n=== Testing Claude 3.5 Haiku ===");
    let claude_model = "us.anthropic.claude-3-5-haiku-20241022-v1:0";
    println!("🤖 Calling Claude with model: {}", claude_model);

    match provider
        .chat_with_tools(claude_model, &messages, &tools[..], &config)
        .await
    {
        Ok(response) => {
            println!("✅ Claude response received:");
            println!("   Content: {}", response.content);
            println!("   Tool calls: {}", response.tool_calls.len());
        }
        Err(e) => {
            println!("❌ Claude failed: {}", e);
        }
    }

    Ok(())
}

#[tokio::test]
async fn test_bedrock_provider_with_credentials() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔑 Testing BedrockProvider with custom credentials");
    println!("================================================");

    // This test validates that the with_credentials method:
    // 1. Accepts credential parameters without error
    // 2. Creates a properly configured provider
    // 3. The provider has the same interface as the standard provider

    let provider = BedrockProvider::with_credentials(
        Some("us-east-1".to_string()),
        "dummy_access_key".to_string(),
        "dummy_secret_key".to_string(),
        Some("dummy_session_token".to_string()),
    )
    .await?;

    println!("✅ BedrockProvider created successfully with custom credentials");

    // Validate the provider has the expected interface
    assert!(provider.supported_models().len() > 0);
    println!(
        "✅ Provider reports {} supported models",
        provider.supported_models().len()
    );

    // Note: We don't actually call the API with dummy credentials
    // This test only validates that the credential injection mechanism works
    println!("✅ Credential injection mechanism validated");

    Ok(())
}
