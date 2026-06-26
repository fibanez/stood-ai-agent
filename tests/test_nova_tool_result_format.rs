use std::env;
use stood::{
    llm::{
        registry::{ProviderRegistry, PROVIDER_REGISTRY},
        traits::{ChatConfig, ProviderType, Tool},
    },
    types::{ContentBlock, MessageRole, Messages},
};

#[tokio::test]
async fn test_nova_tool_result_format() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Nova Tool Result Format Test");
    println!("==============================");

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

    // Test 1: Initial request with tool
    println!("\n=== Test 1: Initial Tool Request ===");
    let mut messages = Messages::new();
    messages.add_user_message("Please read the file at '/tmp/test.txt'");

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
        // Add assistant message with tool use
        let mut content_blocks = vec![];
        if !response.content.is_empty() {
            content_blocks.push(ContentBlock::text(&response.content));
        }

        for tool_call in &response.tool_calls {
            println!(
                "Tool call: {} with input: {}",
                tool_call.name,
                serde_json::to_string_pretty(&tool_call.input)?
            );

            content_blocks.push(ContentBlock::tool_use(
                &tool_call.id,
                &tool_call.name,
                tool_call.input.clone(),
            ));
        }

        messages.messages.push(stood::types::Message::new(
            MessageRole::Assistant,
            content_blocks,
        ));

        // Add tool result
        let tool_result_content = ContentBlock::ToolResult {
            tool_use_id: response.tool_calls[0].id.clone(),
            content: stood::types::ToolResultContent::json(serde_json::json!({
                "content": "Hello from test file!",
                "path": "/tmp/test.txt"
            })),
            is_error: false,
        };

        messages.messages.push(stood::types::Message::new(
            MessageRole::User,
            vec![tool_result_content],
        ));

        // Test 2: Follow-up request with tool result
        println!("\n=== Test 2: Follow-up with Tool Result ===");
        println!("Messages in conversation: {}", messages.messages.len());

        // Debug: Print the messages being sent
        for (i, msg) in messages.messages.iter().enumerate() {
            println!("\nMessage {}: {:?}", i + 1, msg.role);
            for (j, block) in msg.content.iter().enumerate() {
                match block {
                    ContentBlock::Text { text } => {
                        println!("  Block {}: Text: '{}'", j + 1, text);
                    }
                    ContentBlock::ToolUse { id, name, input } => {
                        println!("  Block {}: ToolUse: {} ({})", j + 1, name, id);
                        println!("    Input: {}", serde_json::to_string_pretty(input)?);
                    }
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } => {
                        println!(
                            "  Block {}: ToolResult for {} (error: {})",
                            j + 1,
                            tool_use_id,
                            is_error
                        );
                        println!("    Content: {}", content.to_display_string());
                    }
                    _ => {
                        println!("  Block {}: Other", j + 1);
                    }
                }
            }
        }

        // This should fail with the validation error
        match provider
            .chat_with_tools("us.amazon.nova-lite-v1:0", &messages, &tools, &config)
            .await
        {
            Ok(response2) => {
                println!("\n✅ Follow-up succeeded!");
                println!("Response: {}", response2.content);
            }
            Err(e) => {
                println!("\n❌ Follow-up failed with error:");
                println!("{}", e);
                println!("\nThis is the expected validation error we need to fix");
            }
        }
    }

    Ok(())
}
