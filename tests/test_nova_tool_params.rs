use std::env;
use stood::{
    llm::{
        registry::{ProviderRegistry, PROVIDER_REGISTRY},
        traits::{ChatConfig, LlmProvider, ProviderType, Tool},
    },
    types::Messages,
};

#[tokio::test]
async fn test_nova_tool_parameter_interpretation() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Nova Tool Parameter Interpretation Test");
    println!("=========================================");

    // Disable telemetry
    env::set_var("OTEL_ENABLED", "false");
    env::set_var("RUST_LOG", "stood=trace");
    tracing_subscriber::fmt()
        .with_env_filter("stood=trace")
        .with_target(true)
        .try_init()
        .ok();

    // Configure providers
    ProviderRegistry::configure().await?;

    // Get Bedrock provider
    let provider_arc = PROVIDER_REGISTRY
        .get_provider(ProviderType::Bedrock)
        .await
        .map_err(|e| format!("Failed to get Bedrock provider: {}", e))?;

    let provider = provider_arc.as_ref();

    // Create file_read tool definition
    let tools = vec![Tool {
        name: "file_read".to_string(),
        description: "Read the contents of a text file".to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                }
            },
            "required": ["path"]
        }),
    }];

    // Create messages with different prompting styles
    let test_cases = vec![
        // Test 1: Direct path in message
        (
            "Direct path",
            "Please read the file at '/tmp/test.txt' and tell me what it contains.",
        ),
        // Test 2: Explicit tool instruction
        (
            "Explicit tool",
            "Use the file_read tool with path '/tmp/test.txt' to read the file.",
        ),
        // Test 3: Path parameter format
        (
            "Path parameter",
            "Use the file_read tool with parameter path='/tmp/test.txt'",
        ),
        // Test 4: JSON-style instruction
        (
            "JSON style",
            "Use the file_read tool with {\"path\": \"/tmp/test.txt\"}",
        ),
    ];

    // Test Nova
    println!("\n=== Testing Nova ===");
    for (name, prompt) in &test_cases {
        println!("\nTest: {}", name);
        println!("Prompt: {}", prompt);

        let mut messages = Messages::new();
        messages.add_user_message(prompt);

        let config = ChatConfig {
            temperature: Some(0.0),
            max_tokens: Some(200),
            model_id: "us.amazon.nova-lite-v1:0".to_string(),
            provider: ProviderType::Bedrock,
            enable_thinking: false,
            cache_strategy: stood::llm::traits::CacheStrategy::default(),
            additional_params: std::collections::HashMap::new(),
        };

        let response = provider
            .chat_with_tools("us.amazon.nova-lite-v1:0", &messages, &tools, &config)
            .await?;

        println!(
            "Response has tool calls: {}",
            !response.tool_calls.is_empty()
        );

        if !response.tool_calls.is_empty() {
            for (i, call) in response.tool_calls.iter().enumerate() {
                println!(
                    "Tool call {}: {} with input: {}",
                    i + 1,
                    call.name,
                    serde_json::to_string_pretty(&call.input)?
                );
            }
        } else {
            println!("No tool calls made. Response: {}", response.content);
        }
    }

    // Test Claude for comparison
    println!("\n\n=== Testing Claude (for comparison) ===");
    let prompt = "Please read the file at '/tmp/test.txt' and tell me what it contains.";
    println!("Prompt: {}", prompt);

    let mut messages = Messages::new();
    messages.add_user_message(prompt);

    let config = ChatConfig {
        temperature: Some(0.0),
        max_tokens: Some(200),
        model_id: "anthropic.claude-3-5-haiku-20241022-v1:0".to_string(),
        provider: ProviderType::Bedrock,
        enable_thinking: false,
        cache_strategy: stood::llm::traits::CacheStrategy::default(),
        additional_params: std::collections::HashMap::new(),
    };

    let response = provider
        .chat_with_tools(
            "anthropic.claude-3-5-haiku-20241022-v1:0",
            &messages,
            &tools,
            &config,
        )
        .await?;

    println!(
        "Claude response has tool calls: {}",
        !response.tool_calls.is_empty()
    );

    if !response.tool_calls.is_empty() {
        for (i, call) in response.tool_calls.iter().enumerate() {
            println!(
                "Tool call {}: {} with input: {}",
                i + 1,
                call.name,
                serde_json::to_string_pretty(&call.input)?
            );
        }
    }

    Ok(())
}
