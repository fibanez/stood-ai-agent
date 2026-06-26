//! Prompt Caching Example
//!
//! This example demonstrates how to use AWS Bedrock's prompt caching feature
//! to reduce latency by up to 85% and costs by up to 90% for repeated prompts.
//!
//! Prompt caching works by caching frequently used content (system prompts,
//! tool definitions) across API calls. The cache has a 5-minute TTL that
//! resets with each successful cache hit.
//!
//! # Supported Models
//! - **Claude models**: Full support (system prompt + tool definitions)
//! - **Nova models**: Partial support (system prompt only)
//! - **Mistral models**: Not supported
//!
//! # Running this example
//! ```bash
//! cargo run --example 033_prompt_caching
//! ```
//!
//! # Prerequisites
//! - AWS credentials configured (via environment, profile, or IAM role)
//! - Access to AWS Bedrock Claude or Nova models

use stood::agent::Agent;
use stood::llm::providers::bedrock::BedrockProvider;
use stood::tools::builtin::CalculatorTool;
use stood::CacheStrategy;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging to see cache-related debug messages
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    println!("=== Prompt Caching Example ===\n");

    // Example 1: Basic system prompt caching
    println!("--- Example 1: System Prompt Caching ---");
    basic_system_caching().await?;

    // Example 2: Full caching with tools
    println!("\n--- Example 2: Full Caching (System + Tools) ---");
    full_caching_with_tools().await?;

    // Example 3: Cache performance comparison
    println!("\n--- Example 3: Cache Performance Demo ---");
    cache_performance_demo().await?;

    // Example 4: Checking model cache support
    println!("\n--- Example 4: Model Cache Support ---");
    check_model_support();

    Ok(())
}

/// Example 1: Basic system prompt caching
///
/// This is the most common use case - caching a long system prompt
/// that remains constant across multiple queries.
async fn basic_system_caching() -> Result<(), Box<dyn std::error::Error>> {
    // A detailed system prompt that benefits from caching
    let system_prompt = r#"
You are an expert Rust developer assistant. Your responsibilities include:

1. Code Review: Analyze code for correctness, performance, and idiomatic Rust patterns.
2. Best Practices: Suggest improvements following Rust best practices and conventions.
3. Error Handling: Help with proper error handling using Result and Option types.
4. Memory Safety: Ensure code is memory-safe and follows ownership rules.
5. Performance: Identify potential performance bottlenecks and optimization opportunities.
6. Documentation: Help write clear documentation and doc comments.

When reviewing code:
- Always explain your reasoning
- Provide concrete examples when suggesting changes
- Consider backward compatibility
- Note any potential breaking changes

Remember to be concise but thorough in your responses.
"#;

    // Create agent with system caching enabled
    let mut agent = Agent::builder()
        .provider("bedrock")
        .model_str("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .system_prompt(system_prompt)
        .with_prompt_caching(CacheStrategy::SystemOnly)
        .build()
        .await?;

    // First request - cache WRITE (slightly more expensive)
    println!("Request 1 (cache write):");
    let start = Instant::now();
    let result = agent.execute("What's the difference between &str and String?").await?;
    println!("  Response time: {:?}", start.elapsed());
    println!("  Response: {}...", &result.response.chars().take(100).collect::<String>());

    // Second request - cache READ (much cheaper and faster)
    println!("\nRequest 2 (cache read):");
    let start = Instant::now();
    let result = agent.execute("How do I handle errors in Rust?").await?;
    println!("  Response time: {:?}", start.elapsed());
    println!("  Response: {}...", &result.response.chars().take(100).collect::<String>());

    Ok(())
}

/// Example 2: Full caching with tools
///
/// When using many tools, caching tool definitions can significantly
/// reduce costs since tool schemas are included in every request.
async fn full_caching_with_tools() -> Result<(), Box<dyn std::error::Error>> {
    let system_prompt = "You are a helpful assistant with access to various tools.";

    // Create agent with full caching (system + tools)
    let mut agent = Agent::builder()
        .provider("bedrock")
        .model_str("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
        .system_prompt(system_prompt)
        .tool(Box::new(CalculatorTool::new()))
        .with_prompt_caching(CacheStrategy::SystemAndTools)
        .build()
        .await?;

    // First request with tools - cache write
    println!("Request 1 (cache write for system + tools):");
    let start = Instant::now();
    let result = agent.execute("Calculate 15% of 89.99").await?;
    println!("  Response time: {:?}", start.elapsed());
    println!("  Response: {}", result.response);

    // Second request - cache read
    println!("\nRequest 2 (cache read):");
    let start = Instant::now();
    let result = agent.execute("What's 20% tip on $45.50?").await?;
    println!("  Response time: {:?}", start.elapsed());
    println!("  Response: {}", result.response);

    Ok(())
}

/// Example 3: Cache performance comparison
///
/// This demonstrates the latency improvement from caching
/// by making multiple requests with the same system prompt.
async fn cache_performance_demo() -> Result<(), Box<dyn std::error::Error>> {
    let system_prompt = r#"
You are a concise assistant. Answer questions in one or two sentences maximum.
Focus on accuracy and brevity. Do not include unnecessary explanations.
"#;

    let mut agent = Agent::builder()
        .provider("bedrock")
        .model_str("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .system_prompt(system_prompt)
        .with_system_caching() // Convenience method
        .build()
        .await?;

    let questions = vec![
        "What is the capital of France?",
        "What is 2 + 2?",
        "What color is the sky?",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light?",
    ];

    println!("Making {} requests with cached system prompt...\n", questions.len());

    let mut total_time = std::time::Duration::ZERO;

    for (i, question) in questions.iter().enumerate() {
        let start = Instant::now();
        let result = agent.execute(*question).await?;
        let elapsed = start.elapsed();
        total_time += elapsed;

        println!(
            "  Request {}: {:?} - Q: {} | A: {}",
            i + 1,
            elapsed,
            question,
            result.response.chars().take(50).collect::<String>()
        );
    }

    println!("\nTotal time for {} requests: {:?}", questions.len(), total_time);
    println!(
        "Average time per request: {:?}",
        total_time / questions.len() as u32
    );

    Ok(())
}

/// Example 4: Check model cache support
///
/// Use the BedrockProvider helper methods to check if a model
/// supports prompt caching before enabling it.
fn check_model_support() {
    let models = vec![
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "us.amazon.nova-lite-v1:0",
        "us.amazon.nova-pro-v1:0",
        "mistral.mistral-large-2407-v1:0",
    ];

    println!("Model cache support:\n");
    println!("{:<50} {:>15} {:>15}", "Model", "Prompt Cache", "Tool Cache");
    println!("{}", "-".repeat(80));

    for model in models {
        let prompt_cache = if BedrockProvider::supports_prompt_caching(model) {
            "Yes"
        } else {
            "No"
        };
        let tool_cache = if BedrockProvider::supports_tool_caching(model) {
            "Yes"
        } else {
            "No"
        };

        // Shorten model name for display
        let short_name = model.split(':').next().unwrap_or(model);
        println!("{:<50} {:>15} {:>15}", short_name, prompt_cache, tool_cache);
    }

    println!("\nNote: Nova models support prompt caching but NOT tool caching.");
    println!("Mistral models do not support prompt caching on Bedrock.");
}
