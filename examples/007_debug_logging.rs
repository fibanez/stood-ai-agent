#!/usr/bin/env cargo

//! Example 007: Debug Logging - Demonstrates comprehensive debug logging for conversation analysis
//!
//! This example shows how to enable trace-level logging to see the full conversation thread,
//! buffered requests, responses, and tool execution details for debugging purposes.
//!
//! Usage:
//! ```bash
//! # Full debug logging (very verbose, includes AWS SDK logs)
//! RUST_LOG=trace cargo run --example 007_debug_logging
//!
//! # Filtered to just our application traces (recommended)
//! RUST_LOG=stood=trace cargo run --example 007_debug_logging 2>&1 | grep "📋 TRACE"
//! ```
//!
//! This will show:
//! - Full conversation state before each request
//! - Available tools and their schemas
//! - Each iteration state in the event loop
//! - Tool execution results in detail
//! - Final streaming responses
//!
//! Perfect for debugging and understanding the conversation flow!

use stood::agent::Agent;
use stood::tools::builtin::{CalculatorTool, CurrentTimeTool};
use tokio;
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing with TRACE level to see all conversation details
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .with_target(false)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .init();

    println!("🔍 Starting Debug Logging Example");
    println!("📋 This example will show detailed debug logs of the conversation thread");
    println!("📋 Look for TRACE messages with detailed conversation state!");
    println!();

    // Create an agent with streaming enabled and tools
    let mut agent = Agent::builder()
        .provider("bedrock")
        .model_str("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .system_prompt("You are a helpful assistant with access to tools. Use the calculator tool for math problems and the time tool when asked about time.")
        .tool(Box::new(CalculatorTool))
        .tool(Box::new(CurrentTimeTool))
        .with_streaming(true)  // Enable streaming mode
        .build()
        .await?;

    println!("✅ Agent created with streaming enabled and 2 tools");
    println!("🔧 Tools: Calculator, Current Time");
    println!();

    // Execute a request that will use tools - this will trigger comprehensive trace logging
    println!("🚀 Executing: 'What time is it now, and what's 17 * 29?'");
    println!("📋 Watch the debug logs to see the full conversation flow!");
    println!();

    let response = agent
        .execute("What time is it now, and what's 17 * 29? Please use the appropriate tools.")
        .await?;

    println!("✅ Final Response:");
    println!("📝 Content: {}", response.response);
    println!("🔧 Tools Used: {}", response.used_tools);
    println!("📊 Tools Called: {:?}", response.tools_called);
    println!();

    println!("🎉 Debug Logging Example completed!");
    println!("📋 Review the debug logs above to see:");
    println!("   - Full conversation state before each request");
    println!("   - Available tools and schemas");
    println!("   - Each iteration state in the event loop");
    println!("   - Tool execution results");
    println!("   - Detailed internal execution flow");

    Ok(())
}
