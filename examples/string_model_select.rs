//! Example: String-based model selection
//!
//! Demonstrates the new runtime string API introduced in this release.
//! Instead of importing compile-time typed structs (`Bedrock::ClaudeHaiku45`),
//! you pass provider and model-id as plain strings.
//!
//! New providers or models announced by AWS (or any other provider) are
//! immediately usable without a library update.
//!
//! # Run
//!
//! ```bash
//! cargo run --example string_model_select
//! ```
//!
//! Requires AWS credentials with Bedrock access (`AWS_REGION`, `AWS_ACCESS_KEY_ID`,
//! `AWS_SECRET_ACCESS_KEY`, or a configured AWS profile).

use stood::agent::Agent;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== String-based model selection ===\n");

    // Build an agent using plain strings for provider and model id.
    // No compile-time model registry is needed — the library passes the
    // model id directly to the AWS Bedrock API.
    let mut agent = Agent::builder()
        .provider("bedrock")
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .system_prompt("You are a helpful assistant. Keep responses concise.")
        .build()
        .await?;

    println!("Agent built successfully.");
    println!("  provider : {}", agent.config().provider);
    println!("  model_id : {}", agent.config().model_id);
    println!();

    // Run a single chat turn.
    let result = agent.execute("What is 2 + 2? Reply in one word.").await?;

    println!("Response: {}", result.response);

    Ok(())
}
