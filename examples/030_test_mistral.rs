//! Test Mistral Large 2 via AWS Bedrock
//!
//! This example tests the Mistral Large 2 model implementation.

use stood::agent::Agent;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the agent with Mistral Large 2
    println!("Creating agent with Mistral Large 2...");
    let mut agent = Agent::builder()
        .provider("bedrock")
        .model("mistral.mistral-large-2407-v1:0")
        .build()
        .await?;

    // Test basic chat
    println!("\n==== Testing Basic Chat ====");
    let response = agent
        .execute("What is 2+2? Please answer with just the number.")
        .await?;

    println!("Response: {}", response.response);

    // Validate response contains "4"
    if response.response.contains("4") {
        println!("\n✅ SUCCESS: Mistral Large 2 is working correctly!");
    } else {
        println!("\n❌ FAIL: Expected '4' in response");
        return Err("Validation failed".into());
    }

    Ok(())
}
