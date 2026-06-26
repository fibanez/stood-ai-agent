// Test to verify Nova fix works in provider integration style test
use std::env;
use std::fs;
use stood::{agent::Agent, tools::builtin::FileReadTool};

#[tokio::test]
async fn test_nova_provider_integration_style() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Nova Provider Integration Style Test");
    println!("======================================");

    // Disable telemetry
    env::set_var("OTEL_ENABLED", "false");

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
    let temp_path = temp_dir.join("test_file_read_nova_streaming.txt");
    let test_content = "Hello from Nova Micro streaming file reading test!";
    fs::write(&temp_path, test_content)?;

    // Test Nova file reading via Agent with streaming (this will test tool streaming)
    let mut agent = Agent::builder()
        .provider("bedrock")
        .model_str("us.amazon.nova-micro-v1:0")
        .system_prompt("You are a helpful assistant. When asked to read a file, use the file_read tool and then summarize the content you found.")
        .tool(Box::new(FileReadTool::new()))
        .build()
        .await
        .map_err(|e| format!("Failed to build Nova Micro agent: {}", e))?;

    // Request file reading via streaming - this should trigger Nova tool streaming
    let response = agent
        .execute(&format!(
            "Please read the file at '{}' and tell me what it contains.",
            temp_path.to_str().unwrap()
        ))
        .await
        .map_err(|e| format!("Failed to execute Nova Micro streaming request: {}", e))?;

    if !response.success {
        return Err(format!(
            "Nova Micro streaming agent execution failed: {}",
            response.error.unwrap_or_default()
        )
        .into());
    }

    if response.response.trim().is_empty() {
        return Err("Empty response from Nova Micro streaming agent with file read tool".into());
    }

    // Verify the response mentions the file content (the agent should have used the tool)
    if !response.used_tools {
        return Err("Nova Micro agent should have used the file_read tool".into());
    }

    // Verify the response actually contains content from the file
    let response_lower = response.response.to_lowercase();
    if !response_lower.contains("hello from nova micro")
        && !response_lower.contains("streaming file reading test")
    {
        return Err(format!(
            "Nova Micro response doesn't contain expected file content. Response: {}",
            response.response
        )
        .into());
    }

    println!("\n✅ Nova provider integration style test passed!");
    println!("Response: {}", response.response);

    // Clean up
    fs::remove_file(&temp_path).ok();

    Ok(())
}
