//! CloudWatch Gen AI Observability Example
//!
//! This example demonstrates how to configure and use Stood's telemetry
//! integration with AWS CloudWatch Gen AI Observability.
//!
//! # Prerequisites
//!
//! 1. **AWS Credentials**: Configure via environment, profile, or IAM role
//!    ```bash
//!    export AWS_ACCESS_KEY_ID=your_key
//!    export AWS_SECRET_ACCESS_KEY=your_secret
//!    export AWS_REGION=us-east-1
//!    ```
//!
//! 2. **CloudWatch Setup**:
//!    - Enable Transaction Search in CloudWatch Console
//!    - Set trace destination to CloudWatch Logs:
//!      ```bash
//!      aws xray update-trace-segment-destination --destination CloudWatchLogs
//!      ```
//!
//! 3. **IAM Permissions**: Attach `AWSXrayWriteOnlyPolicy` AND CloudWatch Logs permissions:
//!    ```json
//!    {
//!      "Effect": "Allow",
//!      "Action": [
//!        "logs:CreateLogGroup",
//!        "logs:CreateLogStream",
//!        "logs:DescribeLogGroups",
//!        "logs:DescribeLogStreams"
//!      ],
//!      "Resource": "arn:aws:logs:*:*:log-group:/aws/bedrock-agentcore/*"
//!    }
//!    ```
//!
//! # Running
//!
//! ```bash
//! # With telemetry disabled (default - safe to run without AWS)
//! cargo run --example 025_cloudwatch_observability
//!
//! # With telemetry enabled (requires AWS credentials)
//! STOOD_CLOUDWATCH_ENABLED=true cargo run --example 025_cloudwatch_observability
//! ```
//!
//! # What This Example Shows
//!
//! 1. How to configure `TelemetryConfig` for CloudWatch
//! 2. How to set `agent_id` for log group naming (REQUIRED for GenAI Dashboard)
//! 3. How telemetry spans are automatically created during agent execution
//! 4. How to view traces in CloudWatch Gen AI Observability dashboard

use stood::agent::Agent;
use stood::telemetry::TelemetryConfig;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing subscriber for local logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    println!("==============================================");
    println!("  CloudWatch Gen AI Observability Example");
    println!("==============================================\n");

    // Step 1: Configure telemetry
    // --------------------------
    // Use from_env() to automatically detect if CloudWatch should be enabled
    // Set STOOD_CLOUDWATCH_ENABLED=true to enable
    //
    // IMPORTANT: Two key identifiers:
    // - service_name: Your APPLICATION name (e.g., "qanda-service")
    //   Per OpenTelemetry spec: "Logical name of the service"
    //
    // - agent_id: Unique identifier for this agent (e.g., "qanda-agent-001")
    //   Used to create CloudWatch log group: /aws/bedrock-agentcore/runtimes/{agent_id}
    //   This log group MUST exist for spans to appear in the GenAI Dashboard!
    //
    // Set via environment: OTEL_SERVICE_NAME and STOOD_AGENT_ID
    // Or use builder methods: .with_service_name() and .with_agent_id()
    let telemetry_config = TelemetryConfig::from_env()
        .with_service_name("stood-demo")
        .with_agent_id("stood-demo-agent");

    println!("Telemetry Configuration:");
    println!("  Enabled: {}", telemetry_config.is_enabled());
    println!("  Service: {}", telemetry_config.service_name());
    if let Some(agent_id) = telemetry_config.agent_id() {
        println!("  Agent ID: {}", agent_id);
    }
    if let Some(log_group) = telemetry_config.log_group_name() {
        println!("  Log Group: {}", log_group);
    }

    if telemetry_config.is_enabled() {
        println!("\n  CloudWatch telemetry is ENABLED");
        println!("  Traces will be sent to AWS X-Ray OTLP endpoint");
        if let Some(endpoint) = telemetry_config.otlp_endpoint() {
            println!("  Endpoint: {}", endpoint);
        }
        println!("\n  NOTE: The log group will be created automatically if it doesn't exist.");
        println!("  This requires logs:CreateLogGroup and logs:CreateLogStream permissions.");
    } else {
        println!("\n  CloudWatch telemetry is DISABLED");
        println!("  To enable, set: STOOD_CLOUDWATCH_ENABLED=true");
    }

    // Step 2: Build agent with telemetry
    // -----------------------------------
    println!("\n----------------------------------------------");
    println!("Building agent with telemetry...");

    // IMPORTANT: Set a descriptive agent name for your use case
    // Per OpenTelemetry GenAI spec: "Human-readable name of the GenAI agent"
    // Examples from spec: "Math Tutor", "Fiction Writer"
    let agent_result = Agent::builder()
        .name("Demo Agent") // Descriptive name for this agent's purpose
        .provider("bedrock")
        .model_str("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .system_prompt("You are a helpful assistant that provides concise answers.")
        .with_telemetry(telemetry_config)
        .build()
        .await;

    let mut agent = match agent_result {
        Ok(agent) => {
            println!("  Agent built successfully");
            agent
        }
        Err(e) => {
            println!("  Failed to build agent: {}", e);
            println!("\n  This is expected if AWS credentials are not configured.");
            println!("  The example will demonstrate configuration only.\n");
            demonstrate_configuration_only();
            return Ok(());
        }
    };

    // Step 3: Execute agent (generates telemetry spans)
    // -------------------------------------------------
    println!("\n----------------------------------------------");
    println!("Executing agent (generating telemetry)...\n");

    let prompt = "What is the capital of France? Answer in one sentence.";
    println!("Prompt: {}", prompt);

    match agent.execute(prompt).await {
        Ok(result) => {
            println!("\nResponse: {}", result.response);
            println!("\nMetrics:");
            println!("  Cycles: {}", result.execution.cycles);
            println!("  Duration: {:?}", result.duration);
            if let Some(tokens) = &result.execution.tokens {
                println!(
                    "  Tokens: {} input, {} output",
                    tokens.input_tokens, tokens.output_tokens
                );
            }

            println!("\n----------------------------------------------");
            println!("Telemetry Spans Generated:");
            println!("----------------------------------------------");
            println!("  1. invoke_agent Demo Agent");
            println!("     - gen_ai.operation.name: invoke_agent");
            println!("     - gen_ai.agent.name: Demo Agent");
            if let Some(tokens) = &result.execution.tokens {
                println!("     - gen_ai.usage.input_tokens: {}", tokens.input_tokens);
                println!(
                    "     - gen_ai.usage.output_tokens: {}",
                    tokens.output_tokens
                );
            }
            println!();
            println!("  2. chat claude-4-5-haiku (per cycle)");
            println!("     - gen_ai.operation.name: chat");
            println!("     - gen_ai.request.model: claude-4-5-haiku");
            println!("     - gen_ai.provider.name: aws.bedrock");
        }
        Err(e) => {
            println!("Agent execution failed: {}", e);
            println!("\nThis is expected without valid AWS credentials.");
        }
    }

    // Step 4: Viewing traces
    // ----------------------
    println!("\n==============================================");
    println!("  Viewing Traces in CloudWatch");
    println!("==============================================");
    println!();
    println!("1. Open AWS CloudWatch Console");
    println!("2. Navigate to: Gen AI Observability > Bedrock AgentCore");
    println!("3. Or use Application Signals > Traces for raw traces");
    println!();
    println!("Filter traces by:");
    println!("  - Log Group: /aws/bedrock-agentcore/runtimes/stood-demo-agent");
    println!("  - Service: stood-demo");
    println!("  - Agent: Demo Agent");
    println!("  - Operation: invoke_agent, chat, execute_tool");
    println!();
    println!("IMPORTANT: The log group must exist for spans to appear");
    println!("in the Gen AI Observability Dashboard. Stood creates this");
    println!("automatically when you use with_telemetry().");
    println!();

    Ok(())
}

/// Demonstrates configuration options without requiring AWS
fn demonstrate_configuration_only() {
    println!("==============================================");
    println!("  TelemetryConfig Options");
    println!("==============================================\n");

    // Option 1: Disabled (default)
    let disabled = TelemetryConfig::disabled();
    println!("1. Disabled (default):");
    println!("   let config = TelemetryConfig::disabled();");
    println!("   Enabled: {}\n", disabled.is_enabled());

    // Option 2: CloudWatch with region
    let cloudwatch = TelemetryConfig::cloudwatch("us-east-1");
    println!("2. CloudWatch (region only):");
    println!("   let config = TelemetryConfig::cloudwatch(\"us-east-1\");");
    println!("   Enabled: {}", cloudwatch.is_enabled());
    println!("   Service: {}", cloudwatch.service_name());
    println!("   Agent ID: {:?}\n", cloudwatch.agent_id());

    // Option 3: CloudWatch with custom service name and agent_id
    let cloudwatch_svc = TelemetryConfig::cloudwatch_with_service("us-west-2", "my-service")
        .with_agent_id("my-agent-001");
    println!("3. CloudWatch (with service name and agent_id):");
    println!(
        "   let config = TelemetryConfig::cloudwatch_with_service(\"us-west-2\", \"my-service\")"
    );
    println!("       .with_agent_id(\"my-agent-001\");");
    println!("   Enabled: {}", cloudwatch_svc.is_enabled());
    println!("   Service: {}", cloudwatch_svc.service_name());
    println!("   Agent ID: {:?}", cloudwatch_svc.agent_id());
    println!("   Log Group: {:?}\n", cloudwatch_svc.log_group_name());

    // Option 4: From environment
    println!("4. From environment:");
    println!("   let config = TelemetryConfig::from_env();");
    println!("   Reads: STOOD_CLOUDWATCH_ENABLED, AWS_REGION, OTEL_SERVICE_NAME, STOOD_AGENT_ID\n");

    // Option 5: Builder pattern (RECOMMENDED)
    let builder_config = TelemetryConfig::cloudwatch("eu-west-1")
        .with_service_name("production-service")
        .with_agent_id("production-agent-001")
        .with_content_capture(false)
        .with_log_level(stood::telemetry::LogLevel::DEBUG);
    println!("5. Builder pattern (RECOMMENDED):");
    println!("   let config = TelemetryConfig::cloudwatch(\"eu-west-1\")");
    println!("       .with_service_name(\"production-service\")");
    println!("       .with_agent_id(\"production-agent-001\")  // CRITICAL for GenAI Dashboard");
    println!("       .with_content_capture(false)");
    println!("       .with_log_level(LogLevel::DEBUG);");
    println!("   Service: {}", builder_config.service_name());
    println!("   Agent ID: {:?}", builder_config.agent_id());
    println!("   Log Group: {:?}", builder_config.log_group_name());
    println!("   Log Level: {:?}\n", builder_config.log_level());

    println!("==============================================");
    println!("  GenAI Semantic Conventions");
    println!("==============================================\n");
    println!("Stood follows OpenTelemetry GenAI semantic conventions:");
    println!();
    println!("Span Names:");
    println!("  - invoke_agent {{agent_name}}");
    println!("  - chat {{model}}");
    println!("  - execute_tool {{tool_name}}");
    println!();
    println!("Key Attributes:");
    println!("  - gen_ai.operation.name");
    println!("  - gen_ai.provider.name");
    println!("  - gen_ai.request.model");
    println!("  - gen_ai.usage.input_tokens");
    println!("  - gen_ai.usage.output_tokens");
    println!("  - gen_ai.agent.id");
    println!("  - gen_ai.tool.name");
    println!();
}
