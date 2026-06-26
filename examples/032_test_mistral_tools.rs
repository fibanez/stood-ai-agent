//! Test Mistral Large 3 with tool calling via AWS Bedrock
//!
//! This example tests tool calling with Mistral models.

use stood::agent::Agent;
use stood::tool;

/// Simple calculator tool for testing
#[tool]
async fn calculator(operation: String, a: f64, b: f64) -> Result<f64, String> {
    let result = match operation.as_str() {
        "add" => a + b,
        "subtract" => a - b,
        "multiply" => a * b,
        "divide" => {
            if b == 0.0 {
                return Err("Division by zero".to_string());
            }
            a / b
        }
        _ => return Err("Unknown operation".to_string()),
    };

    Ok(result)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the agent with Mistral Large 3 and calculator tool
    println!("Creating agent with Mistral Large 3 and calculator tool...");
    let mut agent = Agent::builder()
        .provider("bedrock")
        .model_str("mistral.mistral-large-3-675b-instruct")
        .tool(calculator())
        .build()
        .await?;

    // Test tool calling
    println!("\n==== Testing Tool Calling ====");
    let response = agent
        .execute("What is 15 multiplied by 7? Use the calculator tool.")
        .await?;

    println!("Response: {}", response.response);

    // Check if tool was called
    if !response.tools_called.is_empty() {
        println!("\n✅ SUCCESS: Tool calling works!");
        println!("Tools called: {}", response.tools_called.join(", "));
        println!("Tools successful: {}", response.tools_successful.join(", "));
    } else {
        println!("\n❌ FAIL: Expected tool to be called");
        return Err("Tool not called".into());
    }

    // Validate response contains correct answer (105)
    if response.response.contains("105") {
        println!("✅ Correct answer in response!");
    } else {
        println!("⚠️  Answer might be in different format: {}", response.response);
    }

    Ok(())
}
