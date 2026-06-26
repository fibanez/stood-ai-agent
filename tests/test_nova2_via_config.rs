// Test that Nova 2 Pro works when agent is cloned via create_model_from_config()
// This mirrors the actual execution path in agent.execute()

use stood::agent::Agent;

#[tokio::test]
async fn test_nova2_pro_via_execute() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("AWS_ACCESS_KEY_ID").is_err() && std::env::var("AWS_PROFILE").is_err() {
        println!("Skipping test - no AWS credentials");
        return Ok(());
    }

    println!("Building agent with Nova2Pro model...");

    // Build agent with Nova2Pro - this works fine
    let mut agent = Agent::builder()
        .provider("bedrock")
        .model("us.amazon.nova-2-pro-v1:0")
        .system_prompt("You are a helpful assistant.")
        .build()
        .await?;

    println!("Agent built successfully");
    println!("Model ID from config: {}", agent.config().model_id);
    println!("Model ID from model: {}", agent.model().model_id());

    // This is where the bug happens - execute() calls create_model_from_config()
    // with the model_id string from agent.config()
    println!("\nCalling agent.execute() - this triggers create_model_from_config()...");

    let response = agent.execute("What is 2+2? Answer in one word.").await?;

    println!("\n=== Test Results ===");
    println!("Success: {}", response.success);
    println!("Response: {}", response.response);

    if !response.success {
        if let Some(error) = response.error {
            println!("ERROR: {}", error);
            return Err(format!("Agent execution failed: {}", error).into());
        }
    }

    // Verify response
    let response_lower = response.response.to_lowercase();
    if !response_lower.contains("4") && !response_lower.contains("four") {
        return Err(format!("Response doesn't contain expected answer: {}", response.response).into());
    }

    println!("\n✓ Nova 2 Pro works correctly via create_model_from_config()!");
    Ok(())
}

#[tokio::test]
async fn test_nova2_lite_via_execute() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("AWS_ACCESS_KEY_ID").is_err() && std::env::var("AWS_PROFILE").is_err() {
        println!("Skipping test - no AWS credentials");
        return Ok(());
    }

    println!("Building agent with Nova2Lite model...");

    let mut agent = Agent::builder()
        .provider("bedrock")
        .model("us.amazon.nova-2-lite-v1:0")
        .system_prompt("You are a helpful assistant.")
        .build()
        .await?;

    println!("Agent built successfully");
    println!("Model ID: {}", agent.model().model_id());

    println!("\nCalling agent.execute()...");
    let response = agent.execute("What is 2+2? Answer in one word.").await?;

    println!("Success: {}", response.success);

    if !response.success {
        if let Some(error) = response.error {
            return Err(format!("Agent execution failed: {}", error).into());
        }
    }

    println!("✓ Nova 2 Lite works correctly!");
    Ok(())
}
