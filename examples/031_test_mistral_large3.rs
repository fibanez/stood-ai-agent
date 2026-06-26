//! Test Mistral Large 3 via AWS Bedrock
//!
//! This example tests the Mistral Large 3 model implementation.

use stood::agent::Agent;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the agent with Mistral Large 3
    println!("Creating agent with Mistral Large 3...");
    let mut agent = Agent::builder()
        .provider("bedrock")
        .model("mistral.mistral-large-3-675b-instruct")
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
        println!("\n✅ SUCCESS: Mistral Large 3 is working correctly!");
    } else {
        println!("\n❌ FAIL: Expected '4' in response");
        return Err("Validation failed".into());
    }

    // Test a more complex query
    println!("\n==== Testing Complex Reasoning ====");
    let response2 = agent
        .execute("If a train leaves at 2pm and travels for 3 hours, what time does it arrive? Give me just the time.")
        .await?;

    println!("Response: {}", response2.response);

    Ok(())
}
