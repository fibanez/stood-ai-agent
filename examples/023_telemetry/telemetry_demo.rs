//! Example 004: Comprehensive Telemetry Demo
//!
//! This example demonstrates the full power of Stood's telemetry system with:
//! - OpenTelemetry integration with Prometheus metrics
//! - Distributed tracing with detailed spans
//! - GenAI semantic conventions for AI workload observability
//! - Custom metrics for agent performance monitoring
//! - Health checks and readiness probes
//! - Graceful shutdown with telemetry flushing
//!
//! Run with: `cargo run --bin telemetry_demo`
//!
//! This demo includes three phases designed to generate rich telemetry data:
//! 1. Basic Operations - Simple tool calls and basic agentic behavior
//! 2. Complex Reasoning - Multi-step business analysis requiring several tool calls
//! 3. Multi-Cycle Evaluation - Comprehensive feasibility study designed to trigger
//!    multiple evaluation cycles and extensive tool usage for deep tracing visibility
//!
//! Prerequisites:
//! - Docker running (for Prometheus/Grafana stack)
//! - Run `./setup-telemetry.sh` to start monitoring stack
//! - AWS credentials configured for Bedrock

use std::time::Duration;
use tokio::signal;
use tracing::{error, info, warn};

use stood::{
    agent::Agent,
    config::StoodConfig,
    telemetry::{
        init_logging, LoggingConfig, StoodTracer,
        EventLoopMetrics, Timer
    },
    tools::builtin::*,
    tool,
};

#[tool]
/// Get weather information for a location (mock implementation for demo)
async fn get_weather(location: String) -> Result<String, String> {
    // Simulate API latency for realistic telemetry
    tokio::time::sleep(Duration::from_millis(200)).await;

    let weather_info = format!(
        "The weather in {} is sunny, 72°F with light winds. UV index: 6/10",
        location
    );
    info!("Weather data retrieved for location: {}", location);
    Ok(weather_info)
}

#[tool]
/// Calculate compound interest with detailed parameters
async fn calculate_compound_interest(
    principal: f64,
    annual_rate: f64,
    times_compounded: u32,
    years: f64
) -> Result<f64, String> {
    if principal <= 0.0 {
        return Err("Principal must be positive".to_string());
    }
    if annual_rate < 0.0 {
        return Err("Interest rate cannot be negative".to_string());
    }
    if times_compounded == 0 {
        return Err("Compounding frequency must be at least 1".to_string());
    }

    // Simulate computation time
    tokio::time::sleep(Duration::from_millis(50)).await;

    let rate_per_period = annual_rate / (times_compounded as f64);
    let total_periods = (times_compounded as f64) * years;
    let final_amount = principal * (1.0 + rate_per_period).powf(total_periods);

    info!("Compound interest calculated: ${:.2} -> ${:.2} over {} years",
          principal, final_amount, years);

    Ok(final_amount)
}

#[tool]
/// Analyze text for various metrics (word count, character count, sentiment simulation)
async fn analyze_text(text: String) -> Result<String, String> {
    if text.is_empty() {
        return Err("Text cannot be empty".to_string());
    }

    // Simulate text processing time
    tokio::time::sleep(Duration::from_millis(100)).await;

    let word_count = text.split_whitespace().count();
    let char_count = text.len();
    let sentence_count = text.split('.').filter(|s| !s.trim().is_empty()).count();

    // Mock sentiment analysis
    let sentiment = if text.to_lowercase().contains("good") || text.to_lowercase().contains("great") {
        "Positive"
    } else if text.to_lowercase().contains("bad") || text.to_lowercase().contains("terrible") {
        "Negative"
    } else {
        "Neutral"
    };

    let analysis = format!(
        "Text Analysis Results:\n\
         - Word count: {}\n\
         - Character count: {}\n\
         - Sentence count: {}\n\
         - Estimated sentiment: {}\n\
         - Reading time: ~{} minutes",
        word_count, char_count, sentence_count, sentiment, (word_count / 200).max(1)
    );

    info!("Text analysis completed: {} words, {} characters", word_count, char_count);
    Ok(analysis)
}

/// Comprehensive telemetry demo showcasing all observability features
struct TelemetryDemo {
    agent: Agent,
    tracer: Option<StoodTracer>,
    metrics: EventLoopMetrics,
    shutdown_tx: tokio::sync::broadcast::Sender<()>,
    shutdown_rx: tokio::sync::broadcast::Receiver<()>,
}

impl TelemetryDemo {
    /// Initialize the demo with full telemetry stack
    async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Load configuration (telemetry auto-detected)
        let _config = Self::create_config()?;

        // Initialize smart OpenTelemetry tracer with auto-detection
        // For demo purposes, show only final result rather than verbose detection process
        println!("📊 Initializing smart OpenTelemetry...");

        let telemetry_config = stood::telemetry::TelemetryConfig::from_env()
            .with_simple_processing()  // Use simple processing for immediate trace visibility
            .with_service_name("stood-telemetry-demo");
        let tracer = if telemetry_config.enabled {
            println!("🎯 Telemetry endpoint detected: {:?}", telemetry_config.otlp_endpoint);
            let tracer = StoodTracer::init(telemetry_config)?;

            // Initialize OpenTelemetry tracing subscriber to send logs to telemetry collector
            if tracer.is_some() {
                // Set minimal console logging for demo (all logs still go to telemetry collector)
                // Only show critical errors on console, everything else goes to telemetry
                std::env::set_var("RUST_LOG", "stood=error,telemetry_demo=info");

                if let Err(e) = StoodTracer::init_tracing_subscriber() {
                    eprintln!("Failed to initialize OpenTelemetry tracing subscriber: {}", e);
                } else {
                    println!("✅ OpenTelemetry tracing subscriber initialized - logs will stream to telemetry collector");
                }
            }

            tracer
        } else {
            println!("⚠️ No telemetry endpoints detected - using minimal console logging");

            // Fall back to minimal console logging when no telemetry is available
            let logging_config = LoggingConfig {
                log_dir: std::env::current_dir()?.join("logs"),
                max_file_size: 50 * 1024 * 1024, // 50MB
                max_files: 3,
                file_log_level: "DEBUG".to_string(),
                console_log_level: "ERROR".to_string(), // Only show errors on console
                console_enabled: true,
                json_format: true,
                enable_performance_tracing: true,
                enable_cycle_detection: true,
            };

            let _logging_guard = init_logging(logging_config)?;
            None
        };

        // Use tracing macros for structured logging that goes to telemetry
        tracing::info!("🚀 Initializing Stood Telemetry Demo");

        // Note: Bedrock client is now handled internally by the Agent
        tracing::info!("✅ Using Agent builder pattern with internal Bedrock client");

        // Create comprehensive tool set
        let tools = vec![
            get_weather(),
            calculate_compound_interest(),
            analyze_text(),
            Box::new(CalculatorTool::new()) as Box<dyn stood::tools::Tool>,
            Box::new(FileReadTool::new()) as Box<dyn stood::tools::Tool>,
            Box::new(FileWriteTool::new()) as Box<dyn stood::tools::Tool>,
            Box::new(CurrentTimeTool::new()) as Box<dyn stood::tools::Tool>,
            Box::new(HttpRequestTool::new()) as Box<dyn stood::tools::Tool>,
        ];

        tracing::info!("🔧 Registered {} tools for telemetry demonstration", tools.len());

        // Configure agent with smart telemetry (auto-detection)
        let agent = Agent::builder()
            .provider("bedrock")
            .model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
            .temperature(0.7)
            .max_tokens(2048)
            .system_prompt(Self::create_system_prompt())
            .tools(tools)
            .build().await?;
        tracing::info!("🤖 Agent initialized with telemetry integration");

        // Set up graceful shutdown
        let (shutdown_tx, shutdown_rx) = tokio::sync::broadcast::channel(1);
        let shutdown_tx_clone = shutdown_tx.clone();

        tokio::spawn(async move {
            match signal::ctrl_c().await {
                Ok(()) => {
                    println!("\n🛑 Shutdown signal received - stopping demo...");
                    let _ = shutdown_tx_clone.send(());
                }
                Err(err) => {
                    eprintln!("Failed to listen for shutdown signal: {}", err);
                }
            }
        });

        Ok(Self {
            agent,
            tracer,
            metrics: EventLoopMetrics::new(),
            shutdown_tx,
            shutdown_rx,
        })
    }

    /// Create simple configuration for demo (telemetry now auto-detected)
    fn create_config() -> Result<StoodConfig, Box<dyn std::error::Error>> {
        let config = StoodConfig::default();
        info!("📋 Using smart telemetry auto-detection");
        Ok(config)
    }

    /// Create an enhanced system prompt for the demo
    fn create_system_prompt() -> String {
        "You are an advanced AI assistant with comprehensive telemetry and observability capabilities. \
         You have access to multiple tools for weather, calculations, text analysis, file operations, \
         HTTP requests, and time queries.

         Every operation you perform is automatically tracked with detailed metrics including:
         - Request/response latency and token usage
         - Tool selection decisions and execution times
         - Error rates and recovery attempts
         - Distributed tracing across all operations

         Use tools when helpful and provide clear, informative responses. \
         The system includes automatic error recovery, intelligent context management, \
         and real-time performance monitoring.".to_string()
    }

    /// Run the interactive telemetry demonstration
    async fn run_demo(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("🎯 Starting Interactive Telemetry Demo");
        self.print_demo_banner().await;

        // Run demonstration scenarios with telemetry tracking
        println!("🔄 Running telemetry demonstration scenarios...\n");

        // Set up Ctrl+C handler for graceful shutdown during demo
        let mut shutdown_signal = self.shutdown_rx.resubscribe();

        tokio::select! {
            result = async {
                // Run each demo phase with cancellation support
                if let Err(e) = self.run_basic_operations_demo_interruptible().await {
                    return Err(e);
                }
                if let Err(e) = self.run_complex_reasoning_demo_interruptible().await {
                    return Err(e);
                }
                if let Err(e) = self.run_multi_cycle_evaluation_demo_interruptible().await {
                    return Err(e);
                }
                if let Err(e) = self.run_error_handling_demo_interruptible().await {
                    return Err(e);
                }
                if let Err(e) = self.run_performance_stress_test_interruptible().await {
                    return Err(e);
                }

                // Show telemetry summary
                self.show_telemetry_summary().await;

                println!("\n🎯 Demo completed successfully! Telemetry data has been sent to the collector.");
                println!("📊 Check the monitoring stack for detailed insights:");
                println!("   Prometheus: http://localhost:9090");
                println!("   Grafana: http://localhost:3000");
                println!("   Jaeger: http://localhost:16686");

                Ok::<(), Box<dyn std::error::Error>>(())
            } => {
                match result {
                    Ok(_) => println!("\n✅ Telemetry demonstration completed successfully"),
                    Err(e) => {
                        if e.to_string().contains("interrupted") {
                            println!("\n🛑 Demo was interrupted");
                        } else {
                            eprintln!("\n❌ Demo failed: {}", e);
                        }
                    }
                }
            }
            _ = shutdown_signal.recv() => {
                println!("\n🛑 Demo interrupted by user - performing graceful shutdown");
            }
        }

        self.graceful_shutdown().await?;
        Ok(())
    }

    async fn print_demo_banner(&self) {
        println!("\n{}", "=".repeat(80));
        println!("🚀 STOOD TELEMETRY DEMONSTRATION");
        println!("{}", "=".repeat(80));
        println!("📊 OpenTelemetry Integration: ✅ Active");
        println!("📈 Prometheus Metrics: http://localhost:9090");
        println!("📊 Grafana Dashboard: http://localhost:3000 (admin/admin)");
        println!("🔍 Jaeger Tracing: http://localhost:16686");
        println!("📋 Service: stood-telemetry-demo");
        println!("{}", "=".repeat(80));
        println!();
    }

    /// Demonstrate basic operations with telemetry (interruptible)
    async fn run_basic_operations_demo_interruptible(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut shutdown_signal = self.shutdown_rx.resubscribe();

        tokio::select! {
            result = self.run_basic_operations_demo() => result,
            _ = shutdown_signal.recv() => {
                Err("Demo interrupted by user".into())
            }
        }
    }

    /// Demonstrate basic operations with telemetry
    async fn run_basic_operations_demo(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("🎯 Running Basic Operations Demo");
        println!("📊 Phase 1: Basic Operations with Telemetry Tracking");

        let _timer = Timer::start("basic_operations_demo");

        // Span creation example
        if let Some(ref tracer) = self.tracer {
            let mut span = tracer.start_agent_span("basic_operations_demo");
            span.set_attribute("demo.phase", "basic_operations");
            span.set_attribute("demo.expected_tools", "weather,calculator,time");

            let operations = vec![
                "What's the current date and time?",
                "What's the weather like in San Francisco?",
                "Calculate 15% tip on a $89.50 restaurant bill",
                "What's 2 to the power of 8?",
            ];

            for (i, operation) in operations.iter().enumerate() {
                println!("\n🔄 Operation {}: {}", i + 1, operation);

                let operation_start = std::time::Instant::now();
                match self.agent.execute(*operation).await {
                    Ok(result) => {
                        let operation_duration = operation_start.elapsed();
                        println!("✅ Result: {}", result.response);
                        println!("   Cycles: {}, Duration: {:?}, Tools: {}",
                                result.execution.cycles,
                                result.duration,
                                result.tools_called.len());

                        // Record custom metrics
                        span.set_attribute(&format!("operation_{}.duration_ms", i + 1),
                                          operation_duration.as_millis() as i64);
                        span.set_attribute(&format!("operation_{}.cycles", i + 1),
                                          result.execution.cycles as i64);
                        span.set_attribute(&format!("operation_{}.tools_used", i + 1),
                                          result.tools_called.len() as i64);

                        self.metrics.add_cycle(stood::telemetry::CycleMetrics {
                            cycle_id: uuid::Uuid::new_v4(),
                            duration: result.duration,
                            model_invocations: result.execution.cycles,
                            tool_calls: result.tools_called.len() as u32,
                            tokens_used: result.execution.tokens.as_ref().map(|t| stood::telemetry::TokenUsage::new(t.input_tokens, t.output_tokens)).unwrap_or_default(),
                            trace_id: Some(span.trace_info().trace_id.clone()),
                            span_id: Some(span.trace_info().span_id.clone()),
                            start_time: chrono::Utc::now(),
                            success: true,
                            error: None,
                        });
                    }
                    Err(e) => {
                        error!("❌ Operation failed: {}", e);
                        span.set_error(&e.to_string());
                    }
                }

                // Small delay for telemetry visibility
                tokio::time::sleep(Duration::from_millis(500)).await;
            }

            span.set_success();
            span.finish();
        }

        println!("✅ Basic operations demo completed");
        Ok(())
    }

    /// Demonstrate complex multi-step reasoning with detailed tracing (interruptible)
    async fn run_complex_reasoning_demo_interruptible(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut shutdown_signal = self.shutdown_rx.resubscribe();

        tokio::select! {
            result = self.run_complex_reasoning_demo() => result,
            _ = shutdown_signal.recv() => {
                Err("Demo interrupted by user".into())
            }
        }
    }

    /// Demonstrate complex multi-step reasoning with detailed tracing
    async fn run_complex_reasoning_demo(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("🧠 Running Complex Reasoning Demo");
        println!("\n📊 Phase 2: Complex Multi-Step Reasoning with Distributed Tracing");

        if let Some(ref tracer) = self.tracer {
            let mut span = tracer.start_agent_span("complex_reasoning_demo");
            span.set_attribute("demo.phase", "complex_reasoning");
            span.set_attribute("demo.complexity", "high");

            let complex_prompt = r#"I need you to help me develop a comprehensive business and investment analysis for a tech startup idea:

SCENARIO: I'm considering launching a AI-powered personal finance management app that helps users optimize their spending, investments, and savings. I need a thorough analysis to present to potential investors.

REQUIRED ANALYSIS (work through systematically):

1. MARKET TIMING: What's the current date and time for this analysis timestamp?

2. FINANCIAL PROJECTIONS: Calculate the compound interest potential if users invest their savings optimally:
   - Calculate compound interest on $5,000 at 6.2% annual rate, compounded monthly, for 10 years
   - Then calculate the same for $15,000 at 7.1% annual rate, compounded quarterly, for 5 years
   - Compare these scenarios and recommend the better strategy

3. MARKET CONDITIONS: What's the current weather in San Francisco (where we'll be headquartered)? This affects our launch timeline and investor meetings.

4. COMPETITIVE LANDSCAPE: Analyze this market research text for insights:
   "The personal finance app market has seen explosive growth with over 200 million users globally. Leading apps like Mint and YNAB have captured significant market share, but there's still room for innovation in AI-driven personalization and investment optimization. The market is expected to grow 25% annually through 2028."

5. FOLLOW-UP FINANCIAL MODELING: Based on your analysis of the market text, if our app captures just 0.5% of that 200 million user base and charges $9.99/month, calculate our potential annual revenue. Then determine what percentage that represents if the total market is worth $12 billion annually.

6. RISK ASSESSMENT: What's the weather in New York (our backup location) in case San Francisco doesn't work out?

Please work through each step methodically, showing all calculations and reasoning. This will be part of my pitch deck, so be thorough and professional."#;

            println!("🔄 Complex Query: Comprehensive business analysis requiring multiple tool calls and calculations");

            let analysis_start = std::time::Instant::now();
            match self.agent.execute(complex_prompt).await {
                Ok(result) => {
                    let analysis_duration = analysis_start.elapsed();
                    println!("✅ Complex Analysis Completed!");
                    println!("📄 Response: {}", result.response);
                    println!("📊 Execution Metrics:");
                    println!("   Total Duration: {:?}", result.duration);
                    println!("   Cycles Executed: {}", result.execution.cycles);
                    println!("   Tools Used: {}", result.tools_called.len());
                    if let Some(tokens) = &result.execution.tokens {
                        println!("   Tokens Consumed: {} input, {} output",
                                tokens.input_tokens,
                                tokens.output_tokens);
                    }

                    // Record detailed tracing attributes
                    if let Some(tokens) = &result.execution.tokens {
                        span.record_token_usage(
                            tokens.input_tokens,
                            tokens.output_tokens
                        );
                    }
                    span.set_attribute("analysis.complexity_score", 9);
                    span.set_attribute("analysis.tool_interactions", result.tools_called.len() as i64);
                    span.set_attribute("analysis.total_duration_ms", analysis_duration.as_millis() as i64);

                    // Record tool execution details
                    for (i, tool_name) in result.tools_called.iter().enumerate() {
                        span.add_event(
                            "tool.execution.completed",
                            vec![
                                stood::telemetry::KeyValue::new("tool.name", tool_name.clone()),
                                stood::telemetry::KeyValue::new("tool.sequence", i as i64),
                            ],
                        );
                    }

                    span.set_success();
                }
                Err(e) => {
                    error!("❌ Complex analysis failed: {}", e);
                    span.set_error(&e.to_string());
                }
            }

            span.finish();
        }

        println!("✅ Complex reasoning demo completed");
        Ok(())
    }

    /// Demonstrate multi-cycle evaluation with agentic reasoning (interruptible)
    async fn run_multi_cycle_evaluation_demo_interruptible(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut shutdown_signal = self.shutdown_rx.resubscribe();

        tokio::select! {
            result = self.run_multi_cycle_evaluation_demo() => result,
            _ = shutdown_signal.recv() => {
                Err("Demo interrupted by user".into())
            }
        }
    }

    /// Demonstrate multi-cycle evaluation with agentic reasoning
    async fn run_multi_cycle_evaluation_demo(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("🔄 Running Multi-Cycle Evaluation Demo");
        println!("\n📊 Phase 3: Multi-Cycle Agentic Evaluation with Deep Tracing");

        if let Some(ref tracer) = self.tracer {
            let mut span = tracer.start_agent_span("multi_cycle_evaluation_demo");
            span.set_attribute("demo.phase", "multi_cycle_evaluation");
            span.set_attribute("demo.complexity", "extreme");
            span.set_attribute("demo.expected_cycles", "5-8");

            let evaluation_prompt = r#"I need you to conduct a comprehensive feasibility study for a revolutionary new product concept. This analysis must be extremely thorough to convince a skeptical board of directors.

PRODUCT CONCEPT: An AI-powered smart home energy optimization system that learns family routines and automatically adjusts heating, cooling, lighting, and appliances to minimize energy costs while maintaining comfort.

CRITICAL ANALYSIS REQUIREMENTS (be exhaustive and methodical):

1. TIMESTAMP & DOCUMENTATION: Start by recording the current date and time for this feasibility study.

2. FINANCIAL VIABILITY ANALYSIS:
   - Calculate the long-term savings potential: If a typical family spends $2,400/year on electricity and our system reduces usage by 35%, calculate the 10-year savings with compound interest at 3.2% annually
   - Determine break-even point: If our system costs $3,500 installed, when do savings justify the investment?
   - Calculate ROI scenarios for different energy savings percentages (25%, 35%, 45%)

3. MARKET RESEARCH & TIMING:
   - Check current weather conditions in Austin, Texas (our primary market) - extreme weather drives energy costs
   - Analyze this market intelligence: "Smart home energy management systems represent a $4.2B market growing at 23% annually. Current solutions lack AI-driven behavioral learning. Consumer adoption is accelerating due to rising energy costs and environmental consciousness. Key competitors include Nest, Ecobee, and emerging startups, but none offer comprehensive whole-home optimization with behavioral AI."

4. TECHNICAL FEASIBILITY ASSESSMENT:
   - Current time check: How long has this analysis been running? Document processing time.
   - Weather impact analysis: Check conditions in Phoenix, Arizona (secondary market) - desert climates test system limits

5. REVENUE PROJECTIONS & BUSINESS MODEL:
   - Based on the market analysis, if we capture 2.3% of the $4.2B market and it grows at 23% annually, calculate our potential revenue in year 3
   - If we charge $3,500 per system plus $19.99/month for AI services, and achieve 15,000 installations in year 1 with 85% service subscription rate, what's our blended annual revenue?

6. RISK MITIGATION PLANNING:
   - Document final timestamp for analysis duration
   - Weather contingency: Check Boston conditions (cold climate market) for extreme weather resilience planning

EVALUATION CRITERIA: This analysis will be judged on completeness, mathematical accuracy, logical flow, and practical insights. Leave no stone unturned. The board expects rigorous analysis that demonstrates why this product will succeed where others have failed.

Work through each section systematically and thoroughly. Show all calculations, cite specific data points, and provide actionable recommendations."#;

            println!("🔄 Multi-Cycle Query: Comprehensive feasibility study requiring extensive evaluation and tool usage");
            println!("   Expected: 6+ tool calls across multiple evaluation cycles");
            println!("   Complexity: Extreme - designed to trigger multiple agentic reasoning loops\n");

            let evaluation_start = std::time::Instant::now();
            match self.agent.execute(evaluation_prompt).await {
                Ok(result) => {
                    let evaluation_duration = evaluation_start.elapsed();
                    println!("✅ Multi-Cycle Evaluation Completed!");
                    println!("📊 Evaluation Metrics:");
                    println!("   - Tools called: {} times", result.tools_called.len());
                    println!("   - Analysis duration: {:.2}s", evaluation_duration.as_secs_f64());
                    println!("   - Cycles detected: Multiple evaluation passes");

                    if result.tools_called.len() >= 6 {
                        println!("   🎯 SUCCESS: Achieved expected multi-cycle complexity");
                    } else {
                        println!("   ⚠️  Lower complexity than expected - may need more challenging prompt");
                    }

                    println!("\n📄 Feasibility Study Results:");
                    println!("{}", result.response);

                    // Enhanced telemetry for evaluation cycles
                    span.set_attribute("evaluation.cycles_completed", if result.tools_called.len() >= 6 { "high" } else { "medium" });
                    span.set_attribute("evaluation.tool_calls", result.tools_called.len() as i64);
                    span.set_attribute("evaluation.duration_ms", evaluation_duration.as_millis() as i64);
                    span.set_attribute("evaluation.complexity_achieved", if result.tools_called.len() >= 6 { "extreme" } else { "moderate" });

                    // Record detailed tool execution pattern
                    for (i, tool_name) in result.tools_called.iter().enumerate() {
                        span.add_event(
                            "evaluation.tool.executed",
                            vec![
                                stood::telemetry::KeyValue::new("tool.name", tool_name.clone()),
                                stood::telemetry::KeyValue::new("tool.sequence", i as i64),
                                stood::telemetry::KeyValue::new("evaluation.phase", format!("cycle_{}", (i / 2) + 1)),
                            ],
                        );
                    }

                    span.set_success();
                }
                Err(e) => {
                    error!("❌ Multi-cycle evaluation failed: {}", e);
                    span.set_error(&e.to_string());
                }
            }

            span.finish();
        }

        println!("✅ Multi-cycle evaluation demo completed");
        Ok(())
    }

    /// Demonstrate error handling and recovery with telemetry (interruptible)
    async fn run_error_handling_demo_interruptible(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut shutdown_signal = self.shutdown_rx.resubscribe();

        tokio::select! {
            result = self.run_error_handling_demo() => result,
            _ = shutdown_signal.recv() => {
                Err("Demo interrupted by user".into())
            }
        }
    }

    /// Demonstrate error handling and recovery with telemetry
    async fn run_error_handling_demo(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("🚨 Running Error Handling Demo");
        println!("\n📊 Phase 3: Error Handling and Recovery Telemetry");

        if let Some(ref tracer) = self.tracer {
            let mut span = tracer.start_agent_span("error_handling_demo");
            span.set_attribute("demo.phase", "error_handling");
            span.set_attribute("demo.error_types", "validation,tool_failure,recovery");

            let error_scenarios = vec![
                ("Invalid calculation", "Calculate the square root of -1 using complex numbers"),
                ("Empty text analysis", "Analyze this text: ''"),
                ("Invalid compound interest", "Calculate compound interest with principal of -5000"),
            ];

            for (error_type, scenario) in error_scenarios {
                println!("\n🧪 Testing Error Scenario: {}", error_type);
                println!("   Query: {}", scenario);

                let error_start = std::time::Instant::now();
                match self.agent.execute(scenario).await {
                    Ok(result) => {
                        let error_duration = error_start.elapsed();
                        println!("🔄 Handled gracefully: {}", result.response);

                        span.add_event(
                            "error.handled",
                            vec![
                                stood::telemetry::KeyValue::new("error.type", error_type),
                                stood::telemetry::KeyValue::new("error.duration_ms", error_duration.as_millis() as i64),
                                stood::telemetry::KeyValue::new("error.recovery", "successful"),
                            ],
                        );
                    }
                    Err(e) => {
                        let error_duration = error_start.elapsed();
                        warn!("⚠️ Error occurred: {}", e);

                        span.add_event(
                            "error.occurred",
                            vec![
                                stood::telemetry::KeyValue::new("error.type", error_type),
                                stood::telemetry::KeyValue::new("error.message", e.to_string()),
                                stood::telemetry::KeyValue::new("error.duration_ms", error_duration.as_millis() as i64),
                            ],
                        );
                    }
                }

                tokio::time::sleep(Duration::from_millis(200)).await;
            }

            span.finish();
        }

        println!("✅ Error handling demo completed");
        Ok(())
    }

    /// Demonstrate performance under load with metrics (interruptible)
    async fn run_performance_stress_test_interruptible(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut shutdown_signal = self.shutdown_rx.resubscribe();

        tokio::select! {
            result = self.run_performance_stress_test() => result,
            _ = shutdown_signal.recv() => {
                Err("Demo interrupted by user".into())
            }
        }
    }

    /// Demonstrate performance under load with metrics
    async fn run_performance_stress_test(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("⚡ Running Performance Stress Test");
        println!("\n📊 Phase 4: Performance Stress Test with Metrics");

        if let Some(ref tracer) = self.tracer {
            let mut span = tracer.start_agent_span("performance_stress_test");
            span.set_attribute("demo.phase", "performance_test");
            span.set_attribute("demo.load_type", "concurrent_operations");

            let stress_operations = vec![
                "Calculate 5% of 1000",
                "What time is it?",
                "Analyze this text: 'Performance testing is important'",
                "What's 42 divided by 7?",
                "Calculate simple interest on $500 at 3% for 2 years",
            ];

            println!("🔄 Running {} concurrent operations for performance measurement",
                    stress_operations.len());

            let stress_start = std::time::Instant::now();
            let mut handles = vec![];

            for (i, operation) in stress_operations.into_iter().enumerate() {
                let mut agent_clone = self.agent.clone();
                let handle = tokio::spawn(async move {
                    let op_start = std::time::Instant::now();
                    let result = agent_clone.execute(operation).await;
                    let op_duration = op_start.elapsed();
                    (i, operation, result, op_duration)
                });
                handles.push(handle);
            }

            let mut successful_ops = 0;
            let mut total_duration = Duration::ZERO;

            for handle in handles {
                match handle.await {
                    Ok((i, operation, result, duration)) => {
                        match result {
                            Ok(exec_result) => {
                                successful_ops += 1;
                                total_duration += duration;
                                println!("✅ Op {}: {} ({:?})", i + 1, operation, duration);

                                span.add_event(
                                    "stress_test.operation.completed",
                                    vec![
                                        stood::telemetry::KeyValue::new("operation.id", i as i64),
                                        stood::telemetry::KeyValue::new("operation.duration_ms", duration.as_millis() as i64),
                                        stood::telemetry::KeyValue::new("operation.cycles", exec_result.execution.cycles as i64),
                                    ],
                                );
                            }
                            Err(e) => {
                                error!("❌ Op {}: {} failed: {}", i + 1, operation, e);
                                span.add_event(
                                    "stress_test.operation.failed",
                                    vec![
                                        stood::telemetry::KeyValue::new("operation.id", i as i64),
                                        stood::telemetry::KeyValue::new("operation.error", e.to_string()),
                                    ],
                                );
                            }
                        }
                    }
                    Err(e) => {
                        error!("❌ Task execution failed: {}", e);
                    }
                }
            }

            let total_test_duration = stress_start.elapsed();
            let avg_duration = if successful_ops > 0 {
                total_duration / successful_ops
            } else {
                Duration::ZERO
            };

            println!("📊 Performance Test Results:");
            println!("   Successful Operations: {}", successful_ops);
            println!("   Total Test Duration: {:?}", total_test_duration);
            println!("   Average Operation Duration: {:?}", avg_duration);
            println!("   Operations/Second: {:.2}", successful_ops as f64 / total_test_duration.as_secs_f64());

            span.set_attribute("stress_test.successful_operations", successful_ops as i64);
            span.set_attribute("stress_test.total_duration_ms", total_test_duration.as_millis() as i64);
            span.set_attribute("stress_test.avg_duration_ms", avg_duration.as_millis() as i64);
            span.set_attribute("stress_test.ops_per_second", successful_ops as f64 / total_test_duration.as_secs_f64());

            span.finish();
        }

        println!("✅ Performance stress test completed");
        Ok(())
    }

    /// Show comprehensive telemetry summary
    async fn show_telemetry_summary(&self) {
        println!("\n{}", "=".repeat(80));
        println!("📊 TELEMETRY SUMMARY");
        println!("{}", "=".repeat(80));

        let summary = self.metrics.summary();
        println!("📈 Agent Performance Metrics:");
        println!("   Total Cycles: {}", summary.total_cycles);
        println!("   Total Duration: {:?}", summary.total_duration);
        println!("   Average Cycle Duration: {:?}", summary.average_cycle_duration);
        println!("   Successful Tool Executions: {}", summary.successful_tool_executions);
        println!("   Failed Tool Executions: {}", summary.failed_tool_executions);
        println!("   Unique Tools Used: {}", summary.unique_tools_used);
        println!("   Total Tokens: {} input, {} output, {} total",
                summary.total_tokens.input_tokens,
                summary.total_tokens.output_tokens,
                summary.total_tokens.total_tokens);

        println!("\n🔍 View Detailed Telemetry:");
        println!("   📈 Prometheus Metrics: http://localhost:9090");
        println!("   📊 Grafana Dashboard: http://localhost:3000");
        println!("   🔍 Jaeger Traces: http://localhost:16686");
        println!("   📋 Search for service: 'stood-telemetry-demo'");

        println!("\n💡 Key Metrics to Explore:");
        println!("   - Agent cycle duration and success rates");
        println!("   - Tool selection patterns and performance");
        println!("   - Token consumption and cost analysis");
        println!("   - Error rates and recovery patterns");
        println!("   - Distributed traces across tool chains");

        println!("{}", "=".repeat(80));
    }

    /// Perform graceful shutdown with telemetry flushing
    async fn graceful_shutdown(&self) -> Result<(), Box<dyn std::error::Error>> {
        info!("🔄 Performing graceful shutdown with telemetry flush");

        if let Some(ref tracer) = self.tracer {
            println!("📊 Shutting down telemetry...");
            tracer.shutdown();
            println!("✅ Telemetry shutdown completed");
        }

        info!("✅ Telemetry demo shutdown complete");
        println!("\n🎯 Telemetry Demo Complete!");
        println!("📊 Check the monitoring stack for detailed insights:");
        println!("   Prometheus: http://localhost:9090");
        println!("   Grafana: http://localhost:3000");
        println!("   Jaeger: http://localhost:16686");

        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check prerequisites
    if std::env::var("AWS_PROFILE").is_err() && std::env::var("AWS_ACCESS_KEY_ID").is_err() {
        eprintln!("⚠️  AWS credentials required. Configure with:");
        eprintln!("   export AWS_PROFILE=your-profile");
        eprintln!("   OR export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...");
        eprintln!();
    }

    println!("🚀 Starting Stood Telemetry Demo (Press Ctrl+C to exit anytime)");

    // Initialize and run the comprehensive telemetry demo
    let mut demo = TelemetryDemo::new().await?;

    match demo.run_demo().await {
        Ok(_) => {
            println!("🎯 Demo completed successfully - exiting cleanly");
            std::process::exit(0);
        }
        Err(e) => {
            if e.to_string().contains("interrupted") {
                println!("🛑 Demo was interrupted by user - exiting cleanly");
                std::process::exit(0);
            } else {
                eprintln!("❌ Demo failed: {}", e);
                std::process::exit(1);
            }
        }
    }
}