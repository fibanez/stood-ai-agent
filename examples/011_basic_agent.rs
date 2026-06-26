//! Example 011: Basic Agent with Multiple Provider Support
//!
//! This example demonstrates the LLM provider system with four options:
//! 1. LM Studio - Gemma 3 12B (local development)
//! 2. AWS Bedrock - Claude 3.5 Haiku (cloud)
//! 3. AWS Bedrock - Nova Lite (cloud)
//! 4. AWS Bedrock - Nova Micro (cloud)
//!
//! Setup Instructions:
//! - For LM Studio: Download and install LM Studio, load Gemma 3 12B model
//! - For AWS Bedrock: Configure AWS credentials with Bedrock access
//!
//! This demonstrates the unified Agent API with built-in tools,
//! showing how to create an agent and use the single execute() method
//! for both simple chat and complex multi-tool tasks with different providers.

use std::io::{self, Write};
//use stood::tools::builtin::CalculatorTool;
use std::time::Duration;
use stood::{
    agent::{Agent, AgentResult, LogLevel},
    tool,
};

/// Interactive prompt for log level selection
fn select_log_level() -> LogLevel {
    println!("🔧 Select debug log level:");
    println!("  1. Off (no debug output)");
    println!("  2. Info (basic execution flow)");
    println!("  3. Debug (detailed step-by-step)");
    println!("  4. Trace (verbose with full details)");
    print!("Enter your choice (1-4): ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();

    match input.trim() {
        "1" => LogLevel::Off,
        "2" => LogLevel::Info,
        "3" => LogLevel::Debug,
        "4" => LogLevel::Trace,
        _ => {
            println!("Invalid choice, defaulting to Off");
            LogLevel::Off
        }
    }
}

/// Interactive prompt for streaming selection
fn select_streaming() -> bool {
    println!("\n🌊 Select streaming mode:");
    println!("  1. Streaming enabled (real-time response)");
    println!("  2. Streaming disabled (batch response)");
    print!("Enter your choice (1-2): ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();

    match input.trim() {
        "1" => true,
        "2" => false,
        _ => {
            println!("Invalid choice, defaulting to streaming enabled");
            true
        }
    }
}

/// Model configuration options
#[derive(Debug, Clone)]
enum ModelConfig {
    LmStudioGemma12B,
    LmStudioGemma27B,
    BedrockClaudeHaiku,
    BedrockNovaLite,
    BedrockNovaMicro,
}

/// Interactive prompt for model selection
fn select_model() -> ModelConfig {
    println!("\n🤖 Select model and provider:");
    println!("  1. LM Studio - Gemma 3 12B (local)");
    println!("  2. LM Studio - Gemma 3 27B (local)");
    println!("  3. AWS Bedrock - Claude 3.5 Haiku (cloud)");
    println!("  4. AWS Bedrock - Nova Lite (cloud)");
    println!("  5. AWS Bedrock - Nova Micro (cloud)");
    print!("Enter your choice (1-5): ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();

    match input.trim() {
        "1" => ModelConfig::LmStudioGemma12B,
        "2" => ModelConfig::LmStudioGemma27B,
        "3" => ModelConfig::BedrockClaudeHaiku,
        "4" => ModelConfig::BedrockNovaLite,
        "5" => ModelConfig::BedrockNovaMicro,
        _ => {
            println!("Invalid choice, defaulting to LM Studio - Gemma 3 12B");
            ModelConfig::LmStudioGemma12B
        }
    }
}

#[tool]
/// Get weather information for a given location
async fn get_weather(location: String) -> Result<String, String> {
    // Mock weather data - in real usage this would call a weather API
    let weather_info = format!(
        "The weather in {} is sunny, 72°F with light winds.",
        location
    );
    Ok(weather_info)
}

#[tool]
/// Calculate a percentage of a value
async fn calculate_percentage(value: f64, percentage: f64) -> Result<f64, String> {
    if percentage < 0.0 || percentage > 100.0 {
        return Err("Percentage must be between 0 and 100".to_string());
    }
    Ok(value * percentage / 100.0)
}

/// Aggregated metrics across multiple executions
#[derive(Debug, Clone)]
struct AggregatedMetrics {
    total_executions: u32,
    total_tool_calls: u32,
    total_successful_tools: u32,
    total_failed_tools: u32,
    total_cycles: u32,
    total_model_calls: u32,
    total_duration: Duration,
    total_input_tokens: u32,
    total_output_tokens: u32,
    total_tokens: u32,
    used_streaming: bool,
}

/// Display execution metrics for a single result
fn display_execution_metrics(result: &AgentResult, log_level: LogLevel) {
    if log_level == LogLevel::Off {
        return;
    }

    println!("Duration: {:?}", result.duration);
    println!("Used tools: {}", result.used_tools);
    println!("Tool calls: {}", result.tool_call_summary.total_attempts);
    println!("Execution cycles: {}", result.execution.cycles);
    println!("Model calls: {}", result.execution.model_calls);

    if result.used_tools {
        println!("Tools called: {}", result.tools_called.join(", "));
        println!("Tools successful: {}", result.tools_successful.join(", "));
        if !result.tools_failed.is_empty() {
            println!("Tools failed: {}", result.tools_failed.join(", "));
            println!("Tool failure details:");
            for failed_call in &result.tool_call_summary.failed_calls {
                println!(
                    "  - {}: {} (duration: {:?})",
                    failed_call.tool_name, failed_call.error_message, failed_call.duration
                );
            }
        }
        println!(
            "Tool success/failure: {}/{}",
            result.tool_call_summary.successful, result.tool_call_summary.failed
        );
    }
}

/// Display token metrics for a single result
fn display_token_metrics(result: &AgentResult, log_level: LogLevel) {
    if log_level == LogLevel::Off {
        return;
    }

    if let Some(tokens) = &result.execution.tokens {
        println!(
            "Token usage: input={}, output={}, total={}",
            tokens.input_tokens, tokens.output_tokens, tokens.total_tokens
        );
    } else {
        println!("Token usage: not available");
    }
}

/// Aggregate metrics across multiple results
fn aggregate_metrics(results: &[AgentResult]) -> AggregatedMetrics {
    let mut agg = AggregatedMetrics {
        total_executions: results.len() as u32,
        total_tool_calls: 0,
        total_successful_tools: 0,
        total_failed_tools: 0,
        total_cycles: 0,
        total_model_calls: 0,
        total_duration: Duration::new(0, 0),
        total_input_tokens: 0,
        total_output_tokens: 0,
        total_tokens: 0,
        used_streaming: false,
    };

    for result in results {
        agg.total_tool_calls += result.tool_call_summary.total_attempts;
        agg.total_successful_tools += result.tool_call_summary.successful;
        agg.total_failed_tools += result.tool_call_summary.failed;
        agg.total_cycles += result.execution.cycles;
        agg.total_model_calls += result.execution.model_calls;
        agg.total_duration += result.duration;
        agg.used_streaming |= result.execution.performance.was_streamed;

        if let Some(tokens) = &result.execution.tokens {
            agg.total_input_tokens += tokens.input_tokens;
            agg.total_output_tokens += tokens.output_tokens;
            agg.total_tokens += tokens.total_tokens;
        }
    }

    agg
}

/// Display aggregated metrics summary
fn display_aggregated_metrics(metrics: &AggregatedMetrics) {
    println!("\n📊 Aggregated Metrics Summary:");
    println!("   - Total executions: {}", metrics.total_executions);
    println!("   - Total tool calls: {}", metrics.total_tool_calls);
    println!(
        "   - Total successful tools: {}",
        metrics.total_successful_tools
    );
    println!("   - Total failed tools: {}", metrics.total_failed_tools);
    println!("   - Total execution cycles: {}", metrics.total_cycles);
    println!("   - Total model calls: {}", metrics.total_model_calls);
    println!("   - Total execution time: {:?}", metrics.total_duration);
    println!(
        "   - Total tokens: input={}, output={}, total={}",
        metrics.total_input_tokens, metrics.total_output_tokens, metrics.total_tokens
    );
    println!("   - Used streaming: {}", metrics.used_streaming);

    if metrics.total_failed_tools > 0 {
        println!(
            "   - Tool failure rate: {:.1}%",
            (metrics.total_failed_tools as f64 / metrics.total_tool_calls as f64) * 100.0
        );
    }

    if metrics.total_tokens > 0 {
        println!(
            "   - Average tokens per execution: {:.1}",
            metrics.total_tokens as f64 / metrics.total_executions as f64
        );
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🛠️  Basic Agent with Multiple Provider Demo");
    println!("===========================================");
    println!("🚀 Choose between LM Studio, AWS Bedrock Claude, Nova Lite, or Nova Micro");
    println!();

    // User selections
    println!("🔍 Select your configuration options:");

    // Interactive log level selection
    let log_level = select_log_level();

    // Interactive streaming selection
    let streaming_enabled = select_streaming();

    // Interactive model selection
    let model_config = select_model();

    // Disable telemetry when log level is Off to prevent connection attempts
    if log_level == LogLevel::Off {
        std::env::set_var("OTEL_ENABLED", "false");
    }

    // Initialize logging based on user selection
    match log_level {
        LogLevel::Off => {
            // For "Off", completely suppress all tracing/telemetry output
            tracing_subscriber::fmt()
                .with_env_filter("off") // Turn off ALL logging
                .init();
        }
        _ => {
            let filter_directive = match log_level {
                LogLevel::Info => "stood=info,stood_examples=info",
                LogLevel::Debug => "stood=debug,stood_examples=debug",
                LogLevel::Trace => "stood=trace,stood_examples=trace",
                LogLevel::Off => unreachable!(), // Already handled above
            };

            tracing_subscriber::fmt()
                .with_env_filter(filter_directive)
                .with_target(true)
                .init();
        }
    }

    println!("\n✅ Log level set to: {:?}", log_level);
    println!(
        "✅ Streaming mode: {}",
        if streaming_enabled {
            "enabled"
        } else {
            "disabled"
        }
    );
    println!("✅ Model configuration: {:?}", model_config);

    // Explain LM Studio configuration
    if matches!(
        model_config,
        ModelConfig::LmStudioGemma12B | ModelConfig::LmStudioGemma27B
    ) {
        println!("📝 Note: LM Studio/Gemma requires more agentic cycles for complex tasks");
        match model_config {
            ModelConfig::LmStudioGemma12B => println!(
                "   Increased max_cycles from 10 to 20 for better task completion (12B model)"
            ),
            ModelConfig::LmStudioGemma27B => println!(
                "   Increased max_cycles from 10 to 15 for better task completion (27B model)"
            ),
            _ => {}
        }
    }

    // Configure providers based on selection
    println!("\n🔧 Configuring providers...");
    use stood::llm::registry::{ProviderRegistry, PROVIDER_REGISTRY};
    ProviderRegistry::configure().await?;

    // Check if the selected provider is available
    let (provider_type, provider_name) = match model_config {
        ModelConfig::LmStudioGemma12B | ModelConfig::LmStudioGemma27B => {
            (stood::llm::traits::ProviderType::LmStudio, "LM Studio")
        }
        ModelConfig::BedrockClaudeHaiku
        | ModelConfig::BedrockNovaLite
        | ModelConfig::BedrockNovaMicro => {
            (stood::llm::traits::ProviderType::Bedrock, "AWS Bedrock")
        }
    };

    if !PROVIDER_REGISTRY.is_configured(provider_type).await {
        match provider_type {
            stood::llm::traits::ProviderType::LmStudio => {
                println!("❌ LM Studio not detected. Please ensure:");
                println!("   1. LM Studio is installed and running");
                println!("   2. Local server is started (usually at http://localhost:1234)");
                println!("   3. A model is loaded (we recommend Gemma 3 12B)");
                return Err("LM Studio not available".into());
            }
            stood::llm::traits::ProviderType::Bedrock => {
                println!("❌ AWS Bedrock not available. Please ensure:");
                println!("   1. AWS credentials are configured");
                println!("   2. You have access to Bedrock service");
                println!("   3. The selected model is available in your region");
                return Err("AWS Bedrock not available".into());
            }
            _ => {
                return Err(format!("{} not available", provider_name).into());
            }
        }
    }

    // Test provider connection
    let provider = PROVIDER_REGISTRY.get_provider(provider_type).await?;
    let health = provider.health_check().await;
    match health {
        Ok(status) if status.healthy => {
            println!("✅ {} connected successfully!", provider_name);
            if let Some(latency) = status.latency_ms {
                println!("   - Response time: {}ms", latency);
            }
        }
        Ok(status) => {
            println!(
                "⚠️  {} connected but unhealthy: {:?}",
                provider_name, status.error
            );
        }
        Err(e) => {
            println!("❌ {} connection failed: {}", provider_name, e);
            match provider_type {
                stood::llm::traits::ProviderType::LmStudio => {
                    println!("   Please ensure LM Studio is running with a model loaded");
                }
                stood::llm::traits::ProviderType::Bedrock => {
                    println!("   Please check your AWS credentials and Bedrock access");
                }
                _ => {}
            }
            return Err(e.into());
        }
    }

    // ✅ Hybrid approach: Mix macro tools with struct tools seamlessly
    let tools = vec![
        get_weather(), // ✅ Macro tool
        calculate_percentage(), // ✅ Macro tool
                       //Box::new(CalculatorTool::new()) as Box<dyn stood::tools::Tool>, // ✅ Struct tool
    ];

    // Create agent with the selected model and configure callbacks for streaming
    let mut agent = match model_config {
        ModelConfig::LmStudioGemma12B => {
            // LM Studio/Gemma 12B requires more cycles for complex multi-tool tasks
            let event_loop_config = stood::agent::event_loop::EventLoopConfig {
                max_cycles: 20, // Increased from default 10 to handle complex tasks
                ..Default::default()
            };

            let mut builder = Agent::builder()
                .provider("lm_studio")
                .model("google/gemma-3-12b")  // Local model via LM Studio
                .system_prompt("You are a helpful assistant. Prepare a plan to provide an answer. If you can answer confidently, answer directly, but be succint. Don't explain your logic, just provide the user's answer.")
                .with_streaming(streaming_enabled)
                .tools(tools)
                .with_log_level(log_level)
                .with_event_loop_config(event_loop_config);

            if streaming_enabled {
                builder = builder.with_printing_callbacks();
            }

            builder.build().await?
        }
        ModelConfig::LmStudioGemma27B => {
            // LM Studio/Gemma 27B is larger and should need fewer cycles than 12B
            let event_loop_config = stood::agent::event_loop::EventLoopConfig {
                max_cycles: 15, // Fewer cycles than 12B due to better performance
                ..Default::default()
            };

            let mut builder = Agent::builder()
                .provider("lm_studio")
                .model("google/gemma-3-27b")  // New 27B model via LM Studio
                .system_prompt("You are a helpful assistant. Prepare a plan to provide an answer. If you can answer confidently, answer directly, but be succint. Don't explain your logic, just provide the user's answer.")
                .with_streaming(streaming_enabled)
                .tools(tools)
                .with_log_level(log_level)
                .with_event_loop_config(event_loop_config);

            if streaming_enabled {
                builder = builder.with_printing_callbacks();
            }

            builder.build().await?
        }
        ModelConfig::BedrockClaudeHaiku => {
            let mut builder = Agent::builder()
                .provider("bedrock")
                .model("us.anthropic.claude-haiku-4-5-20251001-v1:0")  // Claude Haiku 4.5 via AWS Bedrock
                .system_prompt("You are a helpful assistant. Prepare a plan to provide an answer. If you can answer confidently, answer directly, but be succint. Don't explain your logic, just provide the user's answer.")
                .with_streaming(streaming_enabled)
                .tools(tools)
                .with_log_level(log_level);

            if streaming_enabled {
                builder = builder.with_printing_callbacks();
            }

            builder.build().await?
        }
        ModelConfig::BedrockNovaLite => {
            let mut builder = Agent::builder()
                .provider("bedrock")
                .model("us.amazon.nova-lite-v1:0")  // Nova Lite via AWS Bedrock
                .system_prompt("You are a helpful assistant. Prepare a plan to provide an answer. If you can answer confidently, answer directly, but be succint. Don't explain your logic, just provide the user's answer.")
                .with_streaming(streaming_enabled)
                .tools(tools)
                .with_log_level(log_level);

            if streaming_enabled {
                builder = builder.with_printing_callbacks();
            }

            builder.build().await?
        }
        ModelConfig::BedrockNovaMicro => {
            let mut builder = Agent::builder()
                .provider("bedrock")
                .model("us.amazon.nova-micro-v1:0")  // Nova Micro via AWS Bedrock
                .system_prompt("You are a helpful assistant. Prepare a plan to provide an answer. If you can answer confidently, answer directly, but be succint. Don't explain your logic, just provide the user's answer.")
                .with_streaming(streaming_enabled)
                .tools(tools)
                .with_log_level(log_level);

            if streaming_enabled {
                builder = builder.with_printing_callbacks();
            }

            builder.build().await?
        }
    };

    let model_name = match model_config {
        ModelConfig::LmStudioGemma12B => "Gemma 3 12B via LM Studio",
        ModelConfig::LmStudioGemma27B => "Gemma 3 27B via LM Studio",
        ModelConfig::BedrockClaudeHaiku => "Claude 3.5 Haiku via AWS Bedrock",
        ModelConfig::BedrockNovaLite => "Nova Lite via AWS Bedrock",
        ModelConfig::BedrockNovaMicro => "Nova Micro via AWS Bedrock",
    };

    println!("\n🤖 Agent created with {}", model_name);
    println!(
        "   - Model: {} ({:?})",
        agent.model().model_id(),
        agent.model().provider()
    );
    println!(
        "   - Context window: {} tokens",
        agent.model().context_window()
    );
    println!(
        "   - Max output: {} tokens",
        agent.model().max_output_tokens()
    );
    println!("   - Tools available: {}", 3);

    // Store results for aggregated metrics
    let mut results = Vec::new();

    // Example 1: Simple conversation (no tools needed)
    println!("\n=== Example 1: Simple Chat ===");
    let question = "What is the capital of France?";
    println!("Question: {}", question);

    let result1 = agent.execute(question).await?;

    if !streaming_enabled {
        println!("Agent: {}", result1.response);
    }

    // Display execution and token metrics using reflection
    display_execution_metrics(&result1, log_level);
    display_token_metrics(&result1, log_level);

    results.push(result1);

    // Example 2: Complex multi-tool task (will use multiple tools)
    println!("\n=== Example 2: Multi-Tool Task ===");
    let complex_task =
        "What's the weather like in San Francisco and what's 15% of a $67 restaurant bill? ";
    println!("Task: {}", complex_task);

    let result2 = agent.execute(complex_task).await?;

    if !streaming_enabled {
        println!("Agent: {}", result2.response);
    }

    // Display execution and token metrics using reflection
    display_execution_metrics(&result2, log_level);
    display_token_metrics(&result2, log_level);

    results.push(result2);

    println!("\n✅ Successfully demonstrated LLM provider system:");
    match model_config {
        ModelConfig::LmStudioGemma12B => {
            println!("   - Provider: LM Studio (local)");
            println!("   - Model: Gemma 3 12B (open-source)");
        }
        ModelConfig::LmStudioGemma27B => {
            println!("   - Provider: LM Studio (local)");
            println!("   - Model: Gemma 3 27B (open-source)");
        }
        ModelConfig::BedrockClaudeHaiku => {
            println!("   - Provider: AWS Bedrock (cloud)");
            println!("   - Model: Claude 3.5 Haiku (Anthropic)");
        }
        ModelConfig::BedrockNovaLite => {
            println!("   - Provider: AWS Bedrock (cloud)");
            println!("   - Model: Nova Lite (Amazon)");
        }
        ModelConfig::BedrockNovaMicro => {
            println!("   - Provider: AWS Bedrock (cloud)");
            println!("   - Model: Nova Micro (Amazon)");
        }
    }
    println!(
        "   - Streaming: {}",
        if streaming_enabled {
            "enabled"
        } else {
            "disabled"
        }
    );
    println!("   - Debug level: {:?}", log_level);
    println!("   - Tools: get_weather(), calculate_percentage()");
    println!("   - Unified API: agent.execute() for all interactions");
    println!("   - Provider auto-detection and health checking");

    // Show aggregated metrics using reflection
    let aggregated = aggregate_metrics(&results);
    display_aggregated_metrics(&aggregated);

    Ok(())
}
