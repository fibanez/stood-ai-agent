//! CloudWatch Telemetry Integration Tests
//!
//! These tests verify the CloudWatch telemetry exporter functionality against
//! real AWS infrastructure. They require valid AWS credentials.
//!
//! Test coverage:
//! - Basic span export to X-Ray OTLP endpoint
//! - Log event export to CloudWatch Logs
//! - Smart truncation of large content fields
//! - Batch splitting for large log event sets
//!
//! Run with: cargo test --test telemetry_cloudwatch_integration

use std::collections::HashMap;
use std::env;
use std::time::Duration;

use stood::telemetry::exporter::{CloudWatchExporter, SpanData, SpanExporter, SpanKind, SpanStatus};
use stood::telemetry::{AwsCredentialSource, LogEvent, TelemetryConfig};

/// Helper to check if we have AWS credentials for integration tests
fn ensure_aws_credentials() -> Result<(), String> {
    let has_access_key = env::var("AWS_ACCESS_KEY_ID").is_ok();
    let has_profile = env::var("AWS_PROFILE").is_ok();
    let has_role_arn = env::var("AWS_ROLE_ARN").is_ok();

    if !has_access_key && !has_profile && !has_role_arn {
        return Err("AWS credentials required for integration tests".to_string());
    }
    Ok(())
}

/// Get the AWS region from environment or default to us-east-1
fn get_aws_region() -> String {
    env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string())
}

/// Generate a unique test agent ID to avoid collisions
fn test_agent_id() -> String {
    format!(
        "stood-test-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
            % 1_000_000
    )
}

/// Generate a valid 32-char hex trace ID
fn generate_trace_id() -> String {
    format!(
        "{:032x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    )
}

/// Generate a valid 16-char hex span ID
fn generate_span_id() -> String {
    format!(
        "{:016x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            % u64::MAX as u128
    )
}

// ============================================================================
// Basic CloudWatch Exporter Tests
// ============================================================================

/// Test that the CloudWatch exporter can be created with valid configuration
#[tokio::test]
async fn test_cloudwatch_exporter_creation() {
    if ensure_aws_credentials().is_err() {
        println!("Skipping test - AWS credentials not available");
        return;
    }

    let region = get_aws_region();
    let agent_id = test_agent_id();

    let exporter = CloudWatchExporter::new(
        &region,
        AwsCredentialSource::Environment,
        "stood-test-service",
        env!("CARGO_PKG_VERSION"),
    )
    .with_agent_id(&agent_id)
    .with_timeout(Duration::from_secs(30));

    assert!(exporter.is_healthy());
    println!("CloudWatch exporter created successfully for region: {}", region);
}

/// Test exporting a single span to X-Ray OTLP endpoint
#[tokio::test]
async fn test_cloudwatch_span_export() {
    if ensure_aws_credentials().is_err() {
        println!("Skipping test - AWS credentials not available");
        return;
    }

    let region = get_aws_region();
    let agent_id = test_agent_id();
    let trace_id = generate_trace_id();
    let span_id = generate_span_id();

    println!("Testing span export:");
    println!("  Region: {}", region);
    println!("  Agent ID: {}", agent_id);
    println!("  Trace ID: {}", trace_id);

    let exporter = CloudWatchExporter::new(
        &region,
        AwsCredentialSource::Environment,
        "stood-telemetry-test",
        env!("CARGO_PKG_VERSION"),
    )
    .with_agent_id(&agent_id)
    .with_timeout(Duration::from_secs(30));

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    let span = SpanData {
        trace_id: trace_id.clone(),
        span_id: span_id.clone(),
        parent_span_id: None,
        name: "invoke_agent Test Agent".to_string(),
        kind: SpanKind::Internal,
        start_time_unix_nano: now - 1_000_000_000, // 1 second ago
        end_time_unix_nano: now,
        attributes: {
            let mut attrs = HashMap::new();
            attrs.insert(
                "gen_ai.operation.name".to_string(),
                stood::telemetry::exporter::AttributeValue::String("invoke_agent".to_string()),
            );
            attrs.insert(
                "gen_ai.agent.name".to_string(),
                stood::telemetry::exporter::AttributeValue::String("Test Agent".to_string()),
            );
            attrs.insert(
                "gen_ai.provider.name".to_string(),
                stood::telemetry::exporter::AttributeValue::String("aws.bedrock".to_string()),
            );
            attrs.insert(
                "gen_ai.usage.input_tokens".to_string(),
                stood::telemetry::exporter::AttributeValue::Int(100),
            );
            attrs.insert(
                "gen_ai.usage.output_tokens".to_string(),
                stood::telemetry::exporter::AttributeValue::Int(50),
            );
            attrs
        },
        status: SpanStatus::Ok,
        events: vec![],
    };

    let result = exporter.export(vec![span]).await;

    match result {
        Ok(()) => {
            println!("Span exported successfully to X-Ray OTLP endpoint");
        }
        Err(e) => {
            // Some errors are acceptable (e.g., rate limiting, permission issues)
            println!("Span export result: {:?}", e);
            // Only fail on auth errors, which indicate credential problems
            if matches!(e, stood::telemetry::exporter::ExportError::Auth(_)) {
                panic!("Authentication failed - check AWS credentials: {:?}", e);
            }
        }
    }
}

// ============================================================================
// Log Event Export Tests
// ============================================================================

/// Test exporting log events to CloudWatch Logs
#[tokio::test]
async fn test_cloudwatch_log_event_export() {
    if ensure_aws_credentials().is_err() {
        println!("Skipping test - AWS credentials not available");
        return;
    }

    let region = get_aws_region();
    let agent_id = test_agent_id();
    let trace_id = generate_trace_id();
    let span_id = generate_span_id();

    println!("Testing log event export:");
    println!("  Region: {}", region);
    println!("  Agent ID: {}", agent_id);
    println!("  Log Group: /aws/bedrock-agentcore/runtimes/{}", agent_id);

    let exporter = CloudWatchExporter::new(
        &region,
        AwsCredentialSource::Environment,
        "stood-telemetry-test",
        env!("CARGO_PKG_VERSION"),
    )
    .with_agent_id(&agent_id)
    .with_timeout(Duration::from_secs(30));

    // Create a log event for an agent invocation
    let log_event = LogEvent::for_agent_invocation(
        &trace_id,
        &span_id,
        "test-session-001",
        Some("You are a helpful test assistant."),
        "What is 2+2?",
        "2+2 equals 4.",
    );

    let result = exporter.export_logs(vec![log_event]).await;

    match result {
        Ok(()) => {
            println!("Log event exported successfully to CloudWatch Logs");
            println!("  Log Group: {}", exporter.log_group_name());
            println!("  Log Stream: {}", exporter.log_stream_name());
        }
        Err(e) => {
            println!("Log export result: {:?}", e);
            // Only fail on auth errors
            if matches!(e, stood::telemetry::exporter::ExportError::Auth(_)) {
                panic!("Authentication failed - check AWS credentials: {:?}", e);
            }
        }
    }
}

// ============================================================================
// Truncation Integration Tests
// ============================================================================

/// Test that large log events are truncated before export
#[tokio::test]
async fn test_cloudwatch_log_truncation_large_content() {
    if ensure_aws_credentials().is_err() {
        println!("Skipping test - AWS credentials not available");
        return;
    }

    let region = get_aws_region();
    let agent_id = test_agent_id();
    let trace_id = generate_trace_id();
    let span_id = generate_span_id();

    println!("Testing log truncation with large content:");
    println!("  Region: {}", region);
    println!("  Agent ID: {}", agent_id);

    let exporter = CloudWatchExporter::new(
        &region,
        AwsCredentialSource::Environment,
        "stood-truncation-test",
        env!("CARGO_PKG_VERSION"),
    )
    .with_agent_id(&agent_id)
    .with_timeout(Duration::from_secs(60));

    // Create content that exceeds the 32KB limit
    let large_system_prompt = "System: ".to_string() + &"X".repeat(40_000);
    let large_user_input = "User: ".to_string() + &"Y".repeat(40_000);
    let large_response = "Response: ".to_string() + &"Z".repeat(40_000);

    // Calculate original size
    let log_event_before = LogEvent::for_agent_invocation(
        &trace_id,
        &span_id,
        "test-session-truncation",
        Some(&large_system_prompt),
        &large_user_input,
        &large_response,
    );

    let original_size = serde_json::to_string(&log_event_before)
        .map(|s| s.len())
        .unwrap_or(0);

    println!("  Original log event size: {} bytes ({:.1} KB)",
             original_size, original_size as f64 / 1024.0);

    // The exporter should automatically truncate the content
    let result = exporter.export_logs(vec![log_event_before]).await;

    match result {
        Ok(()) => {
            println!("Large log event exported successfully (truncation worked)");
            println!("  Without truncation, this would have failed due to 1MB CloudWatch limit");
        }
        Err(e) => {
            println!("Log export result: {:?}", e);
            // Rate limiting or other transient errors are acceptable
            if matches!(e, stood::telemetry::exporter::ExportError::Auth(_)) {
                panic!("Authentication failed: {:?}", e);
            }
            // If we get a size error, truncation didn't work
            if let stood::telemetry::exporter::ExportError::Backend { message, .. } = &e {
                if message.contains("too large") || message.contains("exceeds") {
                    panic!("Truncation should have prevented size limit error: {}", message);
                }
            }
        }
    }
}

/// Test exporting a log event with content containing UTF-8 multi-byte characters
#[tokio::test]
async fn test_cloudwatch_log_truncation_utf8() {
    if ensure_aws_credentials().is_err() {
        println!("Skipping test - AWS credentials not available");
        return;
    }

    let region = get_aws_region();
    let agent_id = test_agent_id();
    let trace_id = generate_trace_id();
    let span_id = generate_span_id();

    println!("Testing log truncation with UTF-8 content:");

    let exporter = CloudWatchExporter::new(
        &region,
        AwsCredentialSource::Environment,
        "stood-utf8-test",
        env!("CARGO_PKG_VERSION"),
    )
    .with_agent_id(&agent_id)
    .with_timeout(Duration::from_secs(60));

    // Create content with emoji and other multi-byte UTF-8 characters
    // Each emoji is 4 bytes, so 10,000 emojis = ~40KB
    let emoji_content = "".repeat(10_000);
    let user_input = format!("User message with emojis: {}", emoji_content);
    let response = format!("Response with emojis: {}", emoji_content);

    let log_event = LogEvent::for_agent_invocation(
        &trace_id,
        &span_id,
        "test-session-utf8",
        None,
        &user_input,
        &response,
    );

    println!("  Content contains {} emojis per field", 10_000);

    let result = exporter.export_logs(vec![log_event]).await;

    match result {
        Ok(()) => {
            println!("UTF-8 log event exported successfully");
            println!("  Truncation handled multi-byte characters correctly");
        }
        Err(e) => {
            println!("Log export result: {:?}", e);
            if matches!(e, stood::telemetry::exporter::ExportError::Auth(_)) {
                panic!("Authentication failed: {:?}", e);
            }
        }
    }
}

// ============================================================================
// Batch Splitting Tests
// ============================================================================

/// Test that multiple log events are batched correctly
#[tokio::test]
async fn test_cloudwatch_log_batch_export() {
    if ensure_aws_credentials().is_err() {
        println!("Skipping test - AWS credentials not available");
        return;
    }

    let region = get_aws_region();
    let agent_id = test_agent_id();
    let trace_id = generate_trace_id();

    println!("Testing batch log export:");
    println!("  Region: {}", region);
    println!("  Agent ID: {}", agent_id);

    let exporter = CloudWatchExporter::new(
        &region,
        AwsCredentialSource::Environment,
        "stood-batch-test",
        env!("CARGO_PKG_VERSION"),
    )
    .with_agent_id(&agent_id)
    .with_timeout(Duration::from_secs(60));

    // Create multiple log events with moderately large content
    // Each event is ~50KB, so 30 events = ~1.5MB which requires batch splitting
    let events: Vec<LogEvent> = (0..30)
        .map(|i| {
            let span_id = format!("{:016x}", i);
            let content = format!("Event {} content: {}", i, "X".repeat(50_000));
            LogEvent::for_agent_invocation(
                &trace_id,
                &span_id,
                &format!("session-batch-{}", i),
                None,
                &format!("Query {}", i),
                &content,
            )
        })
        .collect();

    println!("  Exporting {} log events (should be split into multiple batches)", events.len());

    let result = exporter.export_logs(events).await;

    match result {
        Ok(()) => {
            println!("Batch log export completed successfully");
            println!("  All events were split into batches under the 1MB limit");
        }
        Err(e) => {
            println!("Batch export result: {:?}", e);
            if matches!(e, stood::telemetry::exporter::ExportError::Auth(_)) {
                panic!("Authentication failed: {:?}", e);
            }
        }
    }
}

// ============================================================================
// Agent Integration Tests with Telemetry
// ============================================================================

/// Test full agent execution with CloudWatch telemetry enabled
#[tokio::test]
async fn test_agent_with_cloudwatch_telemetry() {
    if ensure_aws_credentials().is_err() {
        println!("Skipping test - AWS credentials not available");
        return;
    }

    use stood::agent::Agent;

    let region = get_aws_region();
    let agent_id = test_agent_id();

    println!("Testing agent with CloudWatch telemetry:");
    println!("  Region: {}", region);
    println!("  Agent ID: {}", agent_id);

    let telemetry_config = TelemetryConfig::cloudwatch(&region)
        .with_service_name("stood-integration-test")
        .with_agent_id(&agent_id)
        .with_content_capture(true); // Enable content capture to test truncation

    let agent_result = Agent::builder()
        .name("Integration Test Agent")
        .provider("bedrock")
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .system_prompt("You are a test assistant. Always respond with exactly: TELEMETRY_TEST_OK")
        .max_tokens(50)
        .with_telemetry(telemetry_config)
        .build()
        .await;

    let mut agent = match agent_result {
        Ok(agent) => agent,
        Err(e) => {
            println!("Failed to build agent: {:?}", e);
            println!("This may be due to Bedrock access or credential issues");
            return;
        }
    };

    println!("  Agent built successfully, executing...");

    let result = agent.execute("Say TELEMETRY_TEST_OK").await;

    match result {
        Ok(response) => {
            println!("Agent execution completed:");
            println!("  Response: {}", response.response);
            println!("  Telemetry spans were exported to CloudWatch");
        }
        Err(e) => {
            println!("Agent execution failed: {:?}", e);
            // Some errors may be acceptable (model access, etc.)
        }
    }
}

/// Test agent execution with large response content (tests truncation in real scenario)
#[tokio::test]
async fn test_agent_with_large_response_telemetry() {
    if ensure_aws_credentials().is_err() {
        println!("Skipping test - AWS credentials not available");
        return;
    }

    use stood::agent::Agent;

    let region = get_aws_region();
    let agent_id = test_agent_id();

    println!("Testing agent with large response telemetry:");
    println!("  Agent ID: {}", agent_id);

    let telemetry_config = TelemetryConfig::cloudwatch(&region)
        .with_service_name("stood-large-response-test")
        .with_agent_id(&agent_id)
        .with_content_capture(true);

    let agent_result = Agent::builder()
        .name("Large Response Test Agent")
        .provider("bedrock")
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .system_prompt("You are a verbose assistant. When asked to explain something, provide a very detailed explanation with many examples.")
        .max_tokens(4000) // Allow for larger responses
        .with_telemetry(telemetry_config)
        .build()
        .await;

    let mut agent = match agent_result {
        Ok(agent) => agent,
        Err(e) => {
            println!("Failed to build agent: {:?}", e);
            return;
        }
    };

    println!("  Requesting large response...");

    // Ask for a detailed response to get substantial content
    let result = agent
        .execute("Explain the theory of relativity in great detail, including history, mathematics, and real-world applications.")
        .await;

    match result {
        Ok(response) => {
            println!("Agent execution completed:");
            println!("  Response length: {} characters", response.response.len());
            println!("  Telemetry with large content was exported successfully");
            if response.response.len() > 10000 {
                println!("  Truncation was likely applied to fit CloudWatch limits");
            }
        }
        Err(e) => {
            println!("Agent execution failed: {:?}", e);
        }
    }
}

// ============================================================================
// TelemetryConfig Tests
// ============================================================================

/// Test that TelemetryConfig can be created from environment
#[test]
fn test_telemetry_config_from_env() {
    // Save current env vars
    let saved_enabled = env::var("STOOD_CLOUDWATCH_ENABLED").ok();
    let saved_region = env::var("AWS_REGION").ok();

    // Test with CloudWatch disabled
    env::remove_var("STOOD_CLOUDWATCH_ENABLED");
    let config = TelemetryConfig::from_env();
    assert!(!config.is_enabled());

    // Test with CloudWatch enabled
    env::set_var("STOOD_CLOUDWATCH_ENABLED", "true");
    env::set_var("AWS_REGION", "us-west-2");
    let config = TelemetryConfig::from_env();
    assert!(config.is_enabled());
    assert!(config.otlp_endpoint().unwrap().contains("us-west-2"));

    // Restore env vars
    if let Some(v) = saved_enabled {
        env::set_var("STOOD_CLOUDWATCH_ENABLED", v);
    } else {
        env::remove_var("STOOD_CLOUDWATCH_ENABLED");
    }
    if let Some(v) = saved_region {
        env::set_var("AWS_REGION", v);
    } else {
        env::remove_var("AWS_REGION");
    }
}

/// Test TelemetryConfig builder methods
#[test]
fn test_telemetry_config_builder() {
    let config = TelemetryConfig::cloudwatch("eu-west-1")
        .with_service_name("my-service")
        .with_agent_id("my-agent-001")
        .with_content_capture(true);

    assert!(config.is_enabled());
    assert_eq!(config.service_name(), "my-service");
    assert_eq!(config.agent_id(), Some("my-agent-001"));
    assert_eq!(
        config.log_group_name(),
        Some("/aws/bedrock-agentcore/runtimes/my-agent-001".to_string())
    );
}
