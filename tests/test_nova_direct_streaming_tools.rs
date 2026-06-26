use futures::StreamExt;
use std::env;
use stood::{
    llm::{
        registry::{ProviderRegistry, PROVIDER_REGISTRY},
        traits::{ChatConfig, ProviderType, StreamEvent, Tool},
    },
    types::Messages,
};

#[tokio::test]
async fn test_nova_direct_streaming_with_tools() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Nova Direct Streaming with Tools Test");
    println!("=======================================");

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

    // Test streaming with tools
    println!("\n=== Testing Nova Streaming with Tools ===");
    let mut messages = Messages::new();
    messages.add_user_message("Please read the file at '/tmp/test.txt' using the file_read tool");

    let config = ChatConfig {
        temperature: Some(0.0),
        max_tokens: Some(200),
        model_id: "us.amazon.nova-lite-v1:0".to_string(),
        provider: ProviderType::Bedrock,
        enable_thinking: false,
        cache_strategy: stood::llm::traits::CacheStrategy::default(),
        additional_params: std::collections::HashMap::new(),
    };

    println!("\nStarting streaming request...");
    let mut stream = provider
        .chat_streaming_with_tools("us.amazon.nova-lite-v1:0", &messages, &tools, &config)
        .await?;

    println!("\nReceiving stream events:");
    let mut event_count = 0;
    let mut tool_calls = Vec::new();
    let mut content_parts = Vec::new();

    while let Some(event) = stream.next().await {
        event_count += 1;
        match &event {
            StreamEvent::ContentDelta { delta, .. } => {
                print!("{}", delta);
                content_parts.push(delta.clone());
            }
            StreamEvent::ToolCallStart { tool_call } => {
                println!(
                    "\n🔧 Tool call start: {} ({})",
                    tool_call.name, tool_call.id
                );
                tool_calls.push(tool_call.clone());
            }
            StreamEvent::ToolCallDelta {
                tool_call_id,
                delta,
            } => {
                println!("🔧 Tool call delta for {}: {}", tool_call_id, delta);
            }
            StreamEvent::Done { .. } => {
                println!("\n✅ Stream completed");
                break;
            }
            StreamEvent::Error { error } => {
                println!("\n❌ Stream error: {}", error);
                return Err(error.clone().into());
            }
            _ => {
                println!("\n📨 Other event: {:?}", event);
            }
        }
    }

    println!("\n\n=== Stream Results ===");
    println!("Total events: {}", event_count);
    println!("Content: {}", content_parts.join(""));
    println!("Tool calls: {}", tool_calls.len());
    for (i, tool_call) in tool_calls.iter().enumerate() {
        println!(
            "  Tool {}: {} with input: {}",
            i + 1,
            tool_call.name,
            serde_json::to_string_pretty(&tool_call.input)?
        );
    }

    Ok(())
}
