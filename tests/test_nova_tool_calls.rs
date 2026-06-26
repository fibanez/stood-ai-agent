use std::env;
use stood::llm::providers::bedrock::BedrockProvider;
use stood::llm::traits::{ChatConfig, LlmProvider, Tool};
use stood::types::{Message, Messages};

#[tokio::test]
async fn test_nova_tool_calls() -> Result<(), Box<dyn std::error::Error>> {
    println!("🛠️  Testing Nova Tool Calls");
    println!("===========================");

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

    // Test message that should trigger tool use
    let messages = Messages {
        messages: vec![Message::user(
            "Please use the calculator tool to compute 15 * 8",
        )],
        system_prompt: None,
    };

    let config = ChatConfig {
        max_tokens: Some(200),
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

    // Test Nova with tool calls
    println!("\n=== Testing Nova Tool Usage ===");
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
            if !response.tool_calls.is_empty() {
                for (i, tool_call) in response.tool_calls.iter().enumerate() {
                    println!(
                        "   Tool call {}: name={}, id={}",
                        i + 1,
                        tool_call.name,
                        tool_call.id
                    );
                    println!(
                        "   Input: {}",
                        serde_json::to_string_pretty(&tool_call.input)?
                    );
                }
            }

            // Verify we got some content
            assert!(!response.content.is_empty(), "Nova should return content");

            println!("✅ Nova tool call test passed!");
        }
        Err(e) => {
            println!("❌ Nova tool call test failed: {}", e);
            return Err(e.into());
        }
    }

    Ok(())
}
