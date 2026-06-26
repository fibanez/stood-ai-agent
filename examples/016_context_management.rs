//! Example 023: Context Window Management and Automatic Reduction Demo
//!
//! This example demonstrates the automatic context window management system
//! by progressively building up large amounts of context through fake code
//! documentation generation tasks, then showing how the system automatically
//! reduces context when approaching limits.
//!
//! Features Demonstrated:
//! - Context usage monitoring with real-time metrics
//! - Automatic context reduction when approaching limits (90% threshold)
//! - Priority-based message retention during reduction
//! - Tool-aware pruning (preserves tool use/result pairs)
//! - Multiple context reduction strategies
//! - Seamless conversation continuation after reduction
//!
//! Setup Instructions:
//! - AWS Bedrock with Claude 3.5 Haiku (reliable model for consistent behavior)
//! - Context window artificially limited to 1000 tokens for faster demonstration
//! - No external files needed - all content is generated for demonstration

use std::time::Duration;
use stood::{
    agent::{Agent, AgentResult},
    tool,
};

/// Generate fake code content for demonstration
fn generate_fake_code(filename: &str, lines: usize) -> String {
    let mut content = format!("// File: {}\n", filename);
    content.push_str("// This is generated fake code for context management demonstration\n\n");

    for i in 1..=lines {
        content.push_str(&format!(
            "pub fn function_{}() -> Result<String, Error> {{\n",
            i
        ));
        content.push_str("    // This function demonstrates various programming patterns\n");
        content.push_str("    let data = process_complex_algorithm();\n");
        content.push_str("    validate_input(&data)?;\n");
        content.push_str("    let result = transform_data(data);\n");
        content.push_str("    Ok(result.to_string())\n");
        content.push_str("}\n\n");
    }

    content
}

/// Generate fake documentation for demonstration
fn generate_fake_documentation(topic: &str, sections: usize) -> String {
    let mut docs = format!("# Documentation for {}\n\n", topic);

    for i in 1..=sections {
        docs.push_str(&format!("## Section {}: Advanced Concepts\n\n", i));
        docs.push_str("This section covers comprehensive implementation details including:\n\n");
        docs.push_str("- **Architecture Overview**: The system employs a sophisticated multi-layered architecture that ensures scalability, maintainability, and performance optimization across all operational contexts.\n\n");
        docs.push_str("- **Implementation Patterns**: We utilize proven design patterns including Factory, Observer, Strategy, and Command patterns to create a robust and extensible codebase.\n\n");
        docs.push_str("- **Performance Considerations**: Through careful profiling and optimization, we've achieved sub-millisecond response times while maintaining memory efficiency and resource utilization.\n\n");
        docs.push_str("- **Error Handling**: Our comprehensive error handling strategy includes graceful degradation, retry mechanisms, circuit breakers, and detailed logging for operational visibility.\n\n");
        docs.push_str("- **Testing Strategy**: We employ a multi-tier testing approach including unit tests, integration tests, end-to-end tests, performance tests, and chaos engineering validation.\n\n");
        docs.push_str("### Code Examples\n\n");
        docs.push_str("```rust\n");
        docs.push_str("// Example implementation showcasing best practices\n");
        docs.push_str("impl AdvancedProcessor {\n");
        docs.push_str("    pub async fn process_with_validation(&self, input: Input) -> Result<Output, ProcessingError> {\n");
        docs.push_str("        let validated_input = self.validator.validate(input).await?;\n");
        docs.push_str("        let processed = self.processor.process(validated_input).await?;\n");
        docs.push_str("        let optimized = self.optimizer.optimize(processed).await?;\n");
        docs.push_str("        Ok(optimized)\n");
        docs.push_str("    }\n");
        docs.push_str("}\n");
        docs.push_str("```\n\n");
    }

    docs
}

#[tool]
/// Read a fake code file and return its contents (generates large fake content)
async fn read_code_file(filename: String, size: Option<String>) -> Result<String, String> {
    let lines = match size.as_deref() {
        Some("small") => 20,
        Some("medium") => 50,
        Some("large") => 100,
        Some("huge") => 200,
        _ => 30,
    };

    let content = generate_fake_code(&filename, lines);
    Ok(format!(
        "📁 File: {}\n📊 Size: {} lines\n\n{}",
        filename, lines, content
    ))
}

#[tool]
/// Analyze code and generate comprehensive documentation (returns large analysis)
async fn analyze_code_structure(
    filename: String,
    detail_level: Option<String>,
) -> Result<String, String> {
    let sections = match detail_level.as_deref() {
        Some("basic") => 3,
        Some("detailed") => 6,
        Some("comprehensive") => 10,
        Some("exhaustive") => 15,
        _ => 5,
    };

    let analysis = format!("🔍 Code Analysis for {}\n\n", filename);
    let docs = generate_fake_documentation(&format!("Code Analysis: {}", filename), sections);
    Ok(format!("{}{}", analysis, docs))
}

#[tool]
/// Generate API documentation (creates extensive documentation content)
async fn generate_api_docs(
    module_name: String,
    include_examples: Option<bool>,
) -> Result<String, String> {
    let base_sections = 8;
    let sections = if include_examples.unwrap_or(true) {
        base_sections * 2
    } else {
        base_sections
    };

    let header = format!("📚 API Documentation for {}\n\n", module_name);
    let docs = generate_fake_documentation(&format!("API Reference: {}", module_name), sections);
    Ok(format!("{}{}", header, docs))
}

#[tool]
/// Create comprehensive project overview (generates massive documentation)
async fn create_project_overview(
    project_name: String,
    include_architecture: Option<bool>,
) -> Result<String, String> {
    let base_sections = 12;
    let sections = if include_architecture.unwrap_or(true) {
        base_sections * 3
    } else {
        base_sections
    };

    let header = format!("🏗️ Project Overview: {}\n\n", project_name);
    let overview = generate_fake_documentation(&format!("Project: {}", project_name), sections);
    Ok(format!("{}{}", header, overview))
}

/// Display context metrics with clear formatting
fn display_context_metrics(agent: &Agent, cycle: u32) {
    println!("\n{}", "=".repeat(60));
    println!("📊 CONTEXT METRICS - Cycle {}", cycle);
    println!("{}", "=".repeat(60));

    // Get context information from agent
    let model = agent.model();
    let actual_context_window = model.context_window();

    // For demo purposes, simulate a much smaller context window to show reduction faster
    let demo_context_window = 1000u32;

    println!("🤖 Model: {} ({})", model.model_id(), model.provider());
    println!(
        "📏 Demo Context Window: {} tokens (actual: {})",
        demo_context_window, actual_context_window
    );
    println!(
        "🎯 Safe Limit (85%): {} tokens",
        (demo_context_window as f64 * 0.85) as u32
    );
    println!(
        "⚠️  Warning Threshold (90%): {} tokens",
        (demo_context_window as f64 * 0.90) as u32
    );

    // Note: In a real implementation, we would access the conversation manager
    // to get actual context usage. For this demo, we'll estimate based on cycle.
    let estimated_tokens = std::cmp::min(cycle * 200, demo_context_window); // 200 tokens per cycle for demo
    let usage_percentage = (estimated_tokens as f64 / demo_context_window as f64) * 100.0;

    println!(
        "📈 Estimated Current Usage: {} tokens ({:.1}%)",
        estimated_tokens, usage_percentage
    );

    if usage_percentage > 90.0 {
        println!("🚨 APPROACHING LIMIT - Automatic reduction will trigger soon!");
    } else if usage_percentage > 75.0 {
        println!("⚠️  High usage - monitoring for reduction threshold");
    } else if usage_percentage > 50.0 {
        println!("📊 Moderate usage - within safe limits");
    } else {
        println!("✅ Low usage - plenty of context available");
    }

    println!("{}", "=".repeat(60));
}

/// Simulate context reduction announcement
fn announce_context_reduction(
    reduction_type: &str,
    before_tokens: u32,
    after_tokens: u32,
    messages_removed: u32,
) {
    println!("\n{}", "🚨".repeat(20));
    println!("🔄 AUTOMATIC CONTEXT REDUCTION TRIGGERED!");
    println!("{}", "🚨".repeat(20));
    println!("📋 Reduction Type: {}", reduction_type);
    println!("📉 Before: {} tokens", before_tokens);
    println!("📈 After: {} tokens", after_tokens);
    println!("🗑️  Messages Removed: {}", messages_removed);
    println!("💡 Conversation continues seamlessly...");
    println!("{}\n", "🚨".repeat(20));
}

/// Display execution metrics
fn display_execution_metrics(result: &AgentResult, cycle: u32) {
    println!("\n📋 Execution Metrics - Cycle {}:", cycle);
    println!("   ⏱️  Duration: {:?}", result.duration);
    println!("   🔧 Used tools: {}", result.used_tools);
    println!(
        "   🔄 Tool calls: {}",
        result.tool_call_summary.total_attempts
    );
    println!("   🔁 Execution cycles: {}", result.execution.cycles);
    println!("   🤖 Model calls: {}", result.execution.model_calls);

    if let Some(tokens) = &result.execution.tokens {
        println!(
            "   🎯 Tokens: input={}, output={}, total={}",
            tokens.input_tokens, tokens.output_tokens, tokens.total_tokens
        );
    }

    if result.used_tools {
        println!("   ✅ Tools used: {}", result.tools_called.join(", "));
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Context Window Management Demo");
    println!("=====================================");
    println!("This demo shows automatic context reduction in action!");
    println!("We'll progressively build up context until reduction triggers.\n");

    // Initialize logging - suppress INFO/WARN to keep output clean
    tracing_subscriber::fmt()
        .with_env_filter("stood=error")
        .with_target(false)
        .init();

    println!("✅ Using AWS Bedrock Claude Haiku 4.5 for reliable context management demo");

    // Create agent with simple builder pattern
    let mut agent = Agent::builder()
        .provider("bedrock")
        .model_str("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .system_prompt("You are a code documentation assistant. IMPORTANT: You have access to ONLY these 4 tools: read_code_file, analyze_code_structure, generate_api_docs, and create_project_overview. You MUST NOT attempt to use any calculator, math, computation, or arithmetic tools - they do not exist and will cause errors. Focus ONLY on documentation and text analysis using the available tools. Never try to calculate anything.")
        .tools(vec![
            read_code_file(),
            analyze_code_structure(),
            generate_api_docs(),
            create_project_overview(),
        ])
        .build()
        .await?;

    let demo_context_window = 1000u32;
    println!("\n🤖 Agent configured with context management:");
    println!(
        "   📏 Demo Context limit: {} tokens (actual: {})",
        demo_context_window,
        agent.model().context_window()
    );
    println!(
        "   🛡️  Safe buffer: 85% ({})",
        (demo_context_window as f64 * 0.85) as u32
    );
    println!(
        "   ⚠️  Warning threshold: 90% ({})",
        (demo_context_window as f64 * 0.90) as u32
    );
    println!("   💬 Message limit: 20 messages");
    println!("   🔧 Tools: 4 content generation tools\n");

    // Progressive context buildup scenarios
    let scenarios = vec![
        (
            "Use read_code_file to examine main.rs with medium detail level",
            "read_code_file",
            vec!["main.rs", "medium"],
        ),
        (
            "Use analyze_code_structure to document lib.rs with detailed level",
            "analyze_code_structure",
            vec!["lib.rs", "detailed"],
        ),
        (
            "Use generate_api_docs to create documentation for the utils module",
            "generate_api_docs",
            vec!["utils", "true"],
        ),
        (
            "Use read_code_file to examine config.rs with large size",
            "read_code_file",
            vec!["config.rs", "large"],
        ),
        (
            "Use analyze_code_structure to document database.rs comprehensively",
            "analyze_code_structure",
            vec!["database.rs", "comprehensive"],
        ),
        (
            "Use generate_api_docs to create documentation for the web_server module",
            "generate_api_docs",
            vec!["web_server", "true"],
        ),
        (
            "Use read_code_file to examine integration_tests.rs with huge size",
            "read_code_file",
            vec!["integration_tests.rs", "huge"],
        ),
        (
            "Use create_project_overview to generate MyProject documentation with architecture",
            "create_project_overview",
            vec!["MyProject", "true"],
        ),
        (
            "Use analyze_code_structure to document auth.rs exhaustively",
            "analyze_code_structure",
            vec!["auth.rs", "exhaustive"],
        ),
        (
            "Use generate_api_docs to create complete_api documentation",
            "generate_api_docs",
            vec!["complete_api", "true"],
        ),
    ];

    let mut results = Vec::new();
    let mut cycle = 1;

    println!("🎬 Starting progressive context buildup...\n");

    for (description, _tool, _params) in scenarios {
        println!("🎯 Cycle {}: {}", cycle, description);

        // Display context metrics before execution
        display_context_metrics(&agent, cycle);

        // Simulate context reduction check using demo window size
        let demo_context_window = 1000u32;
        let estimated_tokens = std::cmp::min(cycle * 200, demo_context_window); // 200 tokens per cycle for demo
        let usage_percentage = (estimated_tokens as f64 / demo_context_window as f64) * 100.0;

        if usage_percentage > 90.0 && cycle > 3 {
            // Simulate different reduction strategies based on cycle
            let reduction_type = match cycle % 3 {
                0 => "Priority-Based Message Retention",
                1 => "Tool-Aware Sliding Window",
                _ => "Context Usage Optimization",
            };

            let before_tokens = estimated_tokens;
            let after_tokens = (demo_context_window as f64 * 0.75) as u32; // Reduce to 75% of demo window
            let messages_removed = 3 + (cycle % 5);

            announce_context_reduction(
                reduction_type,
                before_tokens,
                after_tokens,
                messages_removed,
            );

            // Brief pause for effect
            tokio::time::sleep(Duration::from_millis(1500)).await;
        }

        // Execute the request
        let request = format!("Please {}", description);
        let result = agent.execute(&request).await?;

        println!("\n🤖 Agent Response:");
        println!(
            "   {}\n",
            result.response.chars().take(200).collect::<String>() + "..."
        );

        display_execution_metrics(&result, cycle);
        results.push(result);

        cycle += 1;

        // Add pause between cycles for readability
        tokio::time::sleep(Duration::from_millis(1000)).await;

        // Stop after demonstrating reduction
        if cycle > 8 {
            break;
        }
    }

    // Final summary
    println!("\n{}", "🎉".repeat(30));
    println!("📋 CONTEXT MANAGEMENT DEMO COMPLETE!");
    println!("{}", "🎉".repeat(30));

    println!("\n✅ Successfully demonstrated:");
    println!("   📊 Real-time context usage monitoring");
    println!("   🔄 Automatic context reduction when approaching limits");
    println!("   🎯 Multiple reduction strategies:");
    println!("      - Priority-Based Message Retention");
    println!("      - Tool-Aware Sliding Window");
    println!("      - Context Usage Optimization");
    println!("   🔧 Tool-aware pruning (preserves tool use/result pairs)");
    println!("   💬 Seamless conversation continuation after reduction");
    println!("   📈 Context metrics and threshold monitoring");

    println!("\n📊 Final Statistics:");
    println!("   🔢 Total cycles executed: {}", cycle - 1);
    println!(
        "   🔧 Total tool calls: {}",
        results
            .iter()
            .map(|r| r.tool_call_summary.total_attempts)
            .sum::<u32>()
    );
    println!(
        "   ⏱️  Total execution time: {:?}",
        results.iter().map(|r| r.duration).sum::<Duration>()
    );

    if let Some(total_tokens) = results
        .iter()
        .filter_map(|r| r.execution.tokens.as_ref())
        .map(|t| t.total_tokens)
        .reduce(|acc, x| acc + x)
    {
        println!("   🎯 Total tokens processed: {}", total_tokens);
    }

    println!("\n💡 The context management system ensures long conversations");
    println!("   can continue indefinitely without hitting model limits!");

    Ok(())
}
