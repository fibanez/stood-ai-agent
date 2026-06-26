use std::env;
use stood::{
    llm::{
        registry::{ProviderRegistry, PROVIDER_REGISTRY},
        traits::{ChatConfig, LlmProvider, ProviderType, Tool},
    },
    types::Messages,
};

#[tokio::test]
async fn test_nova_direct_tool_call() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Nova Direct Tool Call Test");
    println!("===========================");

    // Disable telemetry completely
    env::set_var("STOOD_TELEMETRY_ENABLED", "false");
    env::set_var("RUST_LOG", "stood=trace");
    tracing_subscriber::fmt()
        .with_env_filter("stood=trace")
        .with_target(true)
        .try_init()
        .ok();

    // Configure providers
    ProviderRegistry::configure().await?;

    // Get Bedrock provider directly
    let provider_arc = PROVIDER_REGISTRY
        .get_provider(ProviderType::Bedrock)
        .await
        .map_err(|e| format!("Failed to get Bedrock provider: {}", e))?;

    let provider = provider_arc.as_ref();

    // Create tool definition
    let tools = vec![Tool {
        name: "simple_calculator".to_string(),
        description: "Performs simple arithmetic operations".to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
                "operation": {"type": "string", "description": "Operation to perform", "enum": ["add", "multiply"]}
            },
            "required": ["a", "b", "operation"]
        }),
    }];

    // Create messages
    let mut messages = Messages::new();
    messages.add_user_message(
        "Please calculate 5 + 3 using the simple_calculator tool with operation 'add'.",
    );

    // Call Nova directly
    println!("\n📞 Calling Nova directly with tools...");
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

    println!("\n📤 Response:");
    println!("Content: {}", response.content);
    println!("Tool calls: {:?}", response.tool_calls);

    // Verify tool was called
    assert!(
        !response.tool_calls.is_empty(),
        "Nova should have made tool calls"
    );
    assert_eq!(response.tool_calls[0].name, "simple_calculator");

    println!("\n✅ Nova direct tool call test passed!");
    Ok(())
}

#[tokio::test]
async fn test_claude_direct_tool_call() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Claude Direct Tool Call Test");
    println!("=============================");

    // Disable telemetry completely
    env::set_var("STOOD_TELEMETRY_ENABLED", "false");
    env::set_var("RUST_LOG", "stood=trace");
    tracing_subscriber::fmt()
        .with_env_filter("stood=trace")
        .with_target(true)
        .try_init()
        .ok();

    // Configure providers
    ProviderRegistry::configure().await?;

    // Get Bedrock provider directly
    let provider_arc = PROVIDER_REGISTRY
        .get_provider(ProviderType::Bedrock)
        .await
        .map_err(|e| format!("Failed to get Bedrock provider: {}", e))?;

    let provider = provider_arc.as_ref();

    // Create tool definition
    let tools = vec![Tool {
        name: "simple_calculator".to_string(),
        description: "Performs simple arithmetic operations".to_string(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
                "operation": {"type": "string", "description": "Operation to perform", "enum": ["add", "multiply"]}
            },
            "required": ["a", "b", "operation"]
        }),
    }];

    // Create messages
    let mut messages = Messages::new();
    messages.add_user_message(
        "Please calculate 5 + 3 using the simple_calculator tool with operation 'add'.",
    );

    // Call Claude directly
    println!("\n📞 Calling Claude directly with tools...");
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

    println!("\n📤 Response:");
    println!("Content: {}", response.content);
    println!("Tool calls: {:?}", response.tool_calls);

    // Verify tool was called
    assert!(
        !response.tool_calls.is_empty(),
        "Claude should have made tool calls"
    );
    assert_eq!(response.tool_calls[0].name, "simple_calculator");

    println!("\n✅ Claude direct tool call test passed!");
    Ok(())
}
