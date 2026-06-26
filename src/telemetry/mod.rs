//! Telemetry for CloudWatch Gen AI Observability
//!
//! This module provides telemetry integration with AWS CloudWatch
//! Gen AI Observability dashboards.
//!
//! # Quick Start
//!
//! ```no_run
//! use stood::agent::Agent;
//! use stood::telemetry::TelemetryConfig;
//! use stood::llm::models::Bedrock;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Telemetry disabled by default
//!     let mut agent = Agent::builder()
//!         .provider("bedrock").model_str("us.anthropic.claude-haiku-4-5-20251001-v1:0")
//!         .build().await?;
//!
//!     let result = agent.execute("Hello, world!").await?;
//!     println!("Response: {}", result.response);
//!
//!     Ok(())
//! }
//! ```

use crate::StoodError;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};
use uuid::Uuid;

// Keep file logging - this is used in production
pub mod logging;

// Span exporter traits
pub mod exporter;

// GenAI semantic conventions
pub mod genai;

// Tracer implementation
pub mod tracer;

// Session management
pub mod session;

// AWS authentication and SigV4 signing
pub mod aws_auth;

// CloudWatch Log Group management for GenAI Dashboard
pub mod log_group;

// OTEL Log Events for AgentCore Evaluations
pub mod log_event;

pub use aws_auth::{xray_otlp_endpoint, AuthError, AwsCredentialsProvider};
pub use exporter::{ExportError, NoOpExporter, SpanData, SpanExporter};
pub use genai::{attrs, GenAiOperation, GenAiProvider, GenAiToolType};
pub use log_event::{LogEvent, LogEventBody, LogResource, LogScope, Message, MessageList};
pub use log_group::{AgentLogGroup, LogGroupError, LogGroupManager};
pub use logging::*;
pub use session::{Session, SessionManager};
pub use tracer::{StoodSpan, StoodTracer, SESSION_BAGGAGE_KEY};

// Re-export for backwards compatibility during transition
pub use opentelemetry::KeyValue;

/// Log level for telemetry output control
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum LogLevel {
    /// No telemetry output at all
    OFF,
    /// Only error messages
    ERROR,
    /// Error and warning messages
    WARN,
    /// Error, warning, and info messages
    #[default]
    INFO,
    /// Error, warning, info, and debug messages
    DEBUG,
    /// All messages including trace
    TRACE,
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogLevel::OFF => write!(f, "OFF"),
            LogLevel::ERROR => write!(f, "ERROR"),
            LogLevel::WARN => write!(f, "WARN"),
            LogLevel::INFO => write!(f, "INFO"),
            LogLevel::DEBUG => write!(f, "DEBUG"),
            LogLevel::TRACE => write!(f, "TRACE"),
        }
    }
}

impl std::str::FromStr for LogLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "OFF" => Ok(LogLevel::OFF),
            "ERROR" => Ok(LogLevel::ERROR),
            "WARN" | "WARNING" => Ok(LogLevel::WARN),
            "INFO" => Ok(LogLevel::INFO),
            "DEBUG" => Ok(LogLevel::DEBUG),
            "TRACE" => Ok(LogLevel::TRACE),
            _ => Err(format!("Invalid log level: {}", s)),
        }
    }
}

/// How to obtain AWS credentials for CloudWatch export
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AwsCredentialSource {
    /// Use environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    Environment,
    /// Use a specific AWS profile
    Profile(String),
    /// Use IAM role (for EC2/ECS/Lambda)
    IamRole,
    /// Use explicit credentials
    Explicit {
        access_key_id: String,
        secret_access_key: String,
        session_token: Option<String>,
    },
}

impl Default for AwsCredentialSource {
    fn default() -> Self {
        Self::Environment
    }
}

/// Configuration for telemetry and observability
///
/// # Variants
///
/// - `Disabled` - No telemetry exported (default)
/// - `CloudWatch` - Export to AWS CloudWatch Gen AI Observability
///
/// # Example
///
/// ```rust
/// use stood::telemetry::TelemetryConfig;
///
/// // Disabled by default
/// let config = TelemetryConfig::default();
/// assert!(!config.is_enabled());
///
/// // Enable CloudWatch export
/// let config = TelemetryConfig::cloudwatch("us-east-1");
/// assert!(config.is_enabled());
/// ```
#[derive(Debug, Clone)]
pub enum TelemetryConfig {
    /// Telemetry disabled - no spans exported
    Disabled {
        /// Service name (for compatibility)
        service_name: String,
        /// Log level for console output
        log_level: LogLevel,
    },

    /// Export to AWS CloudWatch Gen AI Observability
    CloudWatch {
        /// AWS region (e.g., "us-east-1")
        region: String,
        /// How to obtain AWS credentials
        credentials: AwsCredentialSource,
        /// Service name in traces (e.g., "qanda-service")
        /// Per OpenTelemetry spec: "Logical name of the service"
        service_name: String,
        /// Service version
        service_version: String,
        /// Agent ID for log group naming (e.g., "qanda-agent-001")
        /// Used to create /aws/bedrock-agentcore/runtimes/{agent_id}
        /// If not set, defaults to service_name
        agent_id: Option<String>,
        /// Capture message content (PII risk - default false)
        content_capture: bool,
        /// Log level for console output
        log_level: LogLevel,
        /// Skip log group existence check (assume pre-created)
        /// Set to true when log groups are created at application startup
        /// to avoid the ~1 second timeout on each agent creation
        skip_log_group_check: bool,
    },
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self::Disabled {
            service_name: "stood-agent".to_string(),
            log_level: LogLevel::INFO,
        }
    }
}

impl TelemetryConfig {
    /// Create a disabled telemetry configuration
    pub fn disabled() -> Self {
        Self::default()
    }

    /// Create CloudWatch configuration for a region (uses environment credentials)
    pub fn cloudwatch(region: impl Into<String>) -> Self {
        Self::CloudWatch {
            region: region.into(),
            credentials: AwsCredentialSource::Environment,
            service_name: "stood-agent".to_string(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            agent_id: None,
            content_capture: false,
            log_level: LogLevel::INFO,
            skip_log_group_check: false,
        }
    }

    /// Create CloudWatch configuration with a custom service name
    pub fn cloudwatch_with_service(
        region: impl Into<String>,
        service_name: impl Into<String>,
    ) -> Self {
        Self::CloudWatch {
            region: region.into(),
            credentials: AwsCredentialSource::Environment,
            service_name: service_name.into(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            agent_id: None,
            content_capture: false,
            log_level: LogLevel::INFO,
            skip_log_group_check: false,
        }
    }

    /// Create CloudWatch configuration with explicit credentials
    pub fn cloudwatch_with_credentials(
        region: impl Into<String>,
        credentials: AwsCredentialSource,
    ) -> Self {
        Self::CloudWatch {
            region: region.into(),
            credentials,
            service_name: "stood-agent".to_string(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            agent_id: None,
            content_capture: false,
            log_level: LogLevel::INFO,
            skip_log_group_check: false,
        }
    }

    /// Check if telemetry is enabled
    pub fn is_enabled(&self) -> bool {
        !matches!(self, Self::Disabled { .. })
    }

    // ========================================================================
    // Backwards-compatible field accessors
    // ========================================================================

    /// Get enabled state (for backwards compatibility)
    #[deprecated(note = "Use is_enabled() instead")]
    pub fn enabled(&self) -> bool {
        self.is_enabled()
    }

    /// Get the service name
    pub fn service_name(&self) -> &str {
        match self {
            Self::Disabled { service_name, .. } => service_name,
            Self::CloudWatch { service_name, .. } => service_name,
        }
    }

    /// Get the log level
    pub fn log_level(&self) -> &LogLevel {
        match self {
            Self::Disabled { log_level, .. } => log_level,
            Self::CloudWatch { log_level, .. } => log_level,
        }
    }

    /// Get the OTLP endpoint (if applicable)
    pub fn otlp_endpoint(&self) -> Option<String> {
        match self {
            Self::Disabled { .. } => None,
            Self::CloudWatch { region, .. } => {
                Some(format!("https://xray.{}.amazonaws.com/v1/traces", region))
            }
        }
    }

    // ========================================================================
    // Builder-style methods
    // ========================================================================

    /// Set enabled state (transitions to/from Disabled)
    pub fn with_enabled(self, enabled: bool) -> Self {
        if enabled {
            match self {
                Self::Disabled {
                    service_name,
                    log_level,
                } => Self::CloudWatch {
                    region: std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string()),
                    credentials: AwsCredentialSource::Environment,
                    service_name,
                    service_version: env!("CARGO_PKG_VERSION").to_string(),
                    agent_id: None,
                    content_capture: false,
                    log_level,
                    skip_log_group_check: false,
                },
                other => other,
            }
        } else {
            match self {
                Self::CloudWatch {
                    service_name,
                    log_level,
                    ..
                } => Self::Disabled {
                    service_name,
                    log_level,
                },
                other => other,
            }
        }
    }

    /// Set service name
    pub fn with_service_name(self, name: impl Into<String>) -> Self {
        let name = name.into();
        match self {
            Self::Disabled { log_level, .. } => Self::Disabled {
                service_name: name,
                log_level,
            },
            Self::CloudWatch {
                region,
                credentials,
                service_version,
                agent_id,
                content_capture,
                log_level,
                skip_log_group_check,
                ..
            } => Self::CloudWatch {
                region,
                credentials,
                service_name: name,
                service_version,
                agent_id,
                content_capture,
                log_level,
                skip_log_group_check,
            },
        }
    }

    /// Set service version
    pub fn with_service_version(self, version: impl Into<String>) -> Self {
        match self {
            Self::Disabled { .. } => self, // Version not relevant for disabled
            Self::CloudWatch {
                region,
                credentials,
                service_name,
                agent_id,
                content_capture,
                log_level,
                skip_log_group_check,
                ..
            } => Self::CloudWatch {
                region,
                credentials,
                service_name,
                service_version: version.into(),
                agent_id,
                content_capture,
                log_level,
                skip_log_group_check,
            },
        }
    }

    /// Set log level
    pub fn with_log_level(self, level: LogLevel) -> Self {
        match self {
            Self::Disabled { service_name, .. } => Self::Disabled {
                service_name,
                log_level: level,
            },
            Self::CloudWatch {
                region,
                credentials,
                service_name,
                service_version,
                agent_id,
                content_capture,
                skip_log_group_check,
                ..
            } => Self::CloudWatch {
                region,
                credentials,
                service_name,
                service_version,
                agent_id,
                content_capture,
                log_level: level,
                skip_log_group_check,
            },
        }
    }

    /// Set log level from string
    pub fn with_log_level_str(self, level: &str) -> Result<Self, String> {
        let parsed_level = level.parse::<LogLevel>()?;
        Ok(self.with_log_level(parsed_level))
    }

    /// Set the log level (mutates in place)
    pub fn set_log_level(&mut self, level: LogLevel) {
        match self {
            Self::Disabled { log_level, .. } => *log_level = level,
            Self::CloudWatch { log_level, .. } => *log_level = level,
        }
    }

    /// Enable content capture (PII risk)
    pub fn with_content_capture(self, capture: bool) -> Self {
        match self {
            Self::Disabled { .. } => self,
            Self::CloudWatch {
                region,
                credentials,
                service_name,
                service_version,
                agent_id,
                log_level,
                skip_log_group_check,
                ..
            } => Self::CloudWatch {
                region,
                credentials,
                service_name,
                service_version,
                agent_id,
                content_capture: capture,
                log_level,
                skip_log_group_check,
            },
        }
    }

    /// Set AWS region
    pub fn with_region(self, region: impl Into<String>) -> Self {
        match self {
            Self::Disabled { .. } => self,
            Self::CloudWatch {
                credentials,
                service_name,
                service_version,
                agent_id,
                content_capture,
                log_level,
                skip_log_group_check,
                ..
            } => Self::CloudWatch {
                region: region.into(),
                credentials,
                service_name,
                service_version,
                agent_id,
                content_capture,
                log_level,
                skip_log_group_check,
            },
        }
    }

    /// Set AWS credentials source
    pub fn with_credentials(self, credentials: AwsCredentialSource) -> Self {
        match self {
            Self::Disabled { .. } => self,
            Self::CloudWatch {
                region,
                service_name,
                service_version,
                agent_id,
                content_capture,
                log_level,
                skip_log_group_check,
                ..
            } => Self::CloudWatch {
                region,
                credentials,
                service_name,
                service_version,
                agent_id,
                content_capture,
                log_level,
                skip_log_group_check,
            },
        }
    }

    /// Set agent ID for log group naming
    ///
    /// The agent ID is used to construct the CloudWatch Log Group name:
    /// `/aws/bedrock-agentcore/runtimes/{agent_id}`
    ///
    /// This log group MUST exist for spans to appear in the GenAI Dashboard.
    /// If not set, defaults to the service_name.
    pub fn with_agent_id(self, agent_id: impl Into<String>) -> Self {
        match self {
            Self::Disabled { .. } => self,
            Self::CloudWatch {
                region,
                credentials,
                service_name,
                service_version,
                content_capture,
                log_level,
                skip_log_group_check,
                ..
            } => Self::CloudWatch {
                region,
                credentials,
                service_name,
                service_version,
                agent_id: Some(agent_id.into()),
                content_capture,
                log_level,
                skip_log_group_check,
            },
        }
    }

    /// Skip log group existence check during initialization
    ///
    /// When set to true, the tracer will not check if the CloudWatch log group
    /// exists before exporting spans. This is useful when:
    /// - Log groups are pre-created at application startup
    /// - You want to avoid the ~1 second API timeout per agent
    ///
    /// Make sure the log groups exist before enabling this option.
    pub fn with_skip_log_group_check(self, skip: bool) -> Self {
        match self {
            Self::Disabled { .. } => self,
            Self::CloudWatch {
                region,
                credentials,
                service_name,
                service_version,
                agent_id,
                content_capture,
                log_level,
                ..
            } => Self::CloudWatch {
                region,
                credentials,
                service_name,
                service_version,
                agent_id,
                content_capture,
                log_level,
                skip_log_group_check: skip,
            },
        }
    }

    // ========================================================================
    // Legacy builder methods (for backwards compatibility)
    // ========================================================================

    /// Enable batch processing (no-op for CloudWatch, kept for compatibility)
    #[deprecated(note = "Batch processing is automatic in CloudWatch export")]
    pub fn with_batch_processing(self) -> Self {
        self
    }

    /// Enable simple processing (no-op, kept for compatibility)
    #[deprecated(note = "Processing mode is automatic in CloudWatch export")]
    pub fn with_simple_processing(self) -> Self {
        self
    }

    /// Set OTLP endpoint (deprecated - use cloudwatch() instead)
    #[deprecated(note = "Use cloudwatch() for CloudWatch or wait for future OTEL support")]
    pub fn with_otlp_endpoint(self, _endpoint: impl Into<String>) -> Self {
        self
    }

    /// Enable console export (no-op, kept for compatibility)
    #[deprecated(note = "Console export not supported in new implementation")]
    pub fn with_console_export(self) -> Self {
        self
    }

    // ========================================================================
    // Logging helpers
    // ========================================================================

    /// Check if the given log level should be printed
    pub fn should_log(&self, level: LogLevel) -> bool {
        self.log_level() >= &level
    }

    /// Print an info message if log level allows
    pub fn log_info(&self, message: &str) {
        if self.should_log(LogLevel::INFO) {
            eprintln!("{}", message);
        }
    }

    /// Print a debug message if log level allows
    pub fn log_debug(&self, message: &str) {
        if self.should_log(LogLevel::DEBUG) {
            eprintln!("{}", message);
        }
    }

    /// Print a warning message if log level allows
    pub fn log_warn(&self, message: &str) {
        if self.should_log(LogLevel::WARN) {
            eprintln!("{}", message);
        }
    }

    /// Print an error message if log level allows
    pub fn log_error(&self, message: &str) {
        if self.should_log(LogLevel::ERROR) {
            eprintln!("{}", message);
        }
    }

    // ========================================================================
    // Environment-based configuration
    // ========================================================================

    /// Create telemetry configuration from environment variables
    ///
    /// Environment variables:
    /// - `STOOD_CLOUDWATCH_ENABLED`: Enable CloudWatch export (default: false)
    /// - `AWS_REGION`: AWS region (default: us-east-1)
    /// - `OTEL_SERVICE_NAME`: Service name (default: stood-agent)
    /// - `STOOD_GENAI_CONTENT_CAPTURE`: Capture message content (default: false)
    ///
    /// Legacy variables (still supported):
    /// - `OTEL_ENABLED`: Enable telemetry (default: false)
    pub fn from_env() -> Self {
        // Check for new CloudWatch-specific env var first
        if let Ok(enabled) = std::env::var("STOOD_CLOUDWATCH_ENABLED") {
            if enabled.to_lowercase() == "true" || enabled == "1" {
                return Self::CloudWatch {
                    region: std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string()),
                    credentials: AwsCredentialSource::Environment,
                    service_name: std::env::var("OTEL_SERVICE_NAME")
                        .unwrap_or_else(|_| "stood-agent".to_string()),
                    service_version: std::env::var("OTEL_SERVICE_VERSION")
                        .unwrap_or_else(|_| env!("CARGO_PKG_VERSION").to_string()),
                    agent_id: std::env::var("STOOD_AGENT_ID").ok(),
                    content_capture: std::env::var("STOOD_GENAI_CONTENT_CAPTURE")
                        .map(|v| v.to_lowercase() == "true" || v == "1")
                        .unwrap_or(false),
                    log_level: LogLevel::INFO,
                    skip_log_group_check: false,
                };
            }
        }

        // Legacy: Check OTEL_ENABLED
        if let Ok(enabled) = std::env::var("OTEL_ENABLED") {
            if enabled.to_lowercase() == "true" || enabled == "1" {
                return Self::CloudWatch {
                    region: std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string()),
                    credentials: AwsCredentialSource::Environment,
                    service_name: std::env::var("OTEL_SERVICE_NAME")
                        .unwrap_or_else(|_| "stood-agent".to_string()),
                    service_version: std::env::var("OTEL_SERVICE_VERSION")
                        .unwrap_or_else(|_| env!("CARGO_PKG_VERSION").to_string()),
                    agent_id: std::env::var("STOOD_AGENT_ID").ok(),
                    content_capture: false,
                    log_level: LogLevel::INFO,
                    skip_log_group_check: false,
                };
            }
        }

        // Default: disabled
        Self::Disabled {
            service_name: std::env::var("OTEL_SERVICE_NAME")
                .unwrap_or_else(|_| "stood-agent".to_string()),
            log_level: LogLevel::INFO,
        }
    }

    /// Create a minimal configuration for testing
    pub fn for_testing() -> Self {
        Self::CloudWatch {
            region: "us-east-1".to_string(),
            credentials: AwsCredentialSource::Environment,
            service_name: "stood-agent-test".to_string(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            agent_id: Some("stood-agent-test".to_string()),
            content_capture: false,
            log_level: LogLevel::DEBUG,
            skip_log_group_check: false,
        }
    }

    /// Get the agent ID for log group naming
    ///
    /// Returns the configured agent_id, or falls back to service_name if not set.
    pub fn agent_id(&self) -> Option<&str> {
        match self {
            Self::Disabled { .. } => None,
            Self::CloudWatch {
                agent_id,
                service_name,
                ..
            } => agent_id.as_deref().or(Some(service_name.as_str())),
        }
    }

    /// Get the log group name for GenAI Dashboard
    ///
    /// Returns the full log group path: `/aws/bedrock-agentcore/runtimes/{agent_id}`
    pub fn log_group_name(&self) -> Option<String> {
        self.agent_id()
            .map(|id| format!("/aws/bedrock-agentcore/runtimes/{}", id))
    }

    /// Validate the telemetry configuration
    pub fn validate(&self) -> Result<(), StoodError> {
        match self {
            Self::Disabled { .. } => Ok(()),
            Self::CloudWatch {
                region,
                service_name,
                ..
            } => {
                if region.is_empty() {
                    return Err(StoodError::configuration_error(
                        "AWS region cannot be empty for CloudWatch telemetry",
                    ));
                }
                if service_name.is_empty() {
                    return Err(StoodError::configuration_error(
                        "Service name cannot be empty when telemetry is enabled",
                    ));
                }
                Ok(())
            }
        }
    }
}

// ============================================================================
// Metrics types - kept for agent compatibility
// ============================================================================

/// Metrics collected during event loop execution
#[derive(Debug, Clone, Default)]
pub struct EventLoopMetrics {
    /// Individual model interaction cycle metrics
    pub cycles: Vec<CycleMetrics>,
    /// Total token usage across all cycles
    pub total_tokens: TokenUsage,
    /// Total duration of the event loop
    pub total_duration: Duration,
    /// All tool executions with timing and status
    pub tool_executions: Vec<ToolExecutionMetric>,
    /// Trace information for correlation
    pub traces: Vec<TraceInfo>,
    /// Accumulated metrics for summary reporting
    pub accumulated_usage: AccumulatedMetrics,
}

impl EventLoopMetrics {
    /// Create new empty metrics
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a model interaction cycle to the metrics
    pub fn add_cycle(&mut self, cycle: CycleMetrics) {
        self.total_tokens.input_tokens += cycle.tokens_used.input_tokens;
        self.total_tokens.output_tokens += cycle.tokens_used.output_tokens;
        self.total_tokens.total_tokens += cycle.tokens_used.total_tokens;
        self.total_duration += cycle.duration;

        self.accumulated_usage.total_cycles += 1;
        self.accumulated_usage.total_model_invocations += cycle.model_invocations;
        self.accumulated_usage.total_tool_calls += cycle.tool_calls;

        self.cycles.push(cycle);
    }

    /// Add a tool execution to the metrics
    pub fn add_tool_execution(&mut self, execution: ToolExecutionMetric) {
        self.tool_executions.push(execution);
    }

    /// Add trace information
    pub fn add_trace(&mut self, trace: TraceInfo) {
        self.traces.push(trace);
    }

    /// Get total number of cycles executed
    pub fn total_cycles(&self) -> u32 {
        self.cycles.len() as u32
    }

    /// Get total number of model calls
    pub fn total_model_calls(&self) -> u32 {
        self.accumulated_usage.total_model_invocations
    }

    /// Get total number of tool calls
    pub fn total_tool_calls(&self) -> u32 {
        self.accumulated_usage.total_tool_calls
    }

    /// Get total execution time
    pub fn total_execution_time(&self) -> Duration {
        self.total_duration
    }

    /// Get total input tokens
    pub fn total_input_tokens(&self) -> u32 {
        self.total_tokens.input_tokens
    }

    /// Get total output tokens
    pub fn total_output_tokens(&self) -> u32 {
        self.total_tokens.output_tokens
    }

    /// Get total tokens
    pub fn total_tokens(&self) -> u32 {
        self.total_tokens.total_tokens
    }

    /// Get total time spent on model calls
    pub fn total_model_time(&self) -> Duration {
        self.total_duration / 2
    }

    /// Get total time spent on tool execution
    pub fn total_tool_time(&self) -> Duration {
        self.tool_executions.iter().map(|t| t.duration).sum()
    }

    /// Get list of unique tools used
    pub fn tools_used(&self) -> Vec<String> {
        let mut tools: Vec<String> = self
            .tool_executions
            .iter()
            .map(|t| t.tool_name.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        tools.sort();
        tools
    }

    /// Get list of successful tools
    pub fn tools_successful(&self) -> Vec<String> {
        let mut tools: Vec<String> = self
            .tool_executions
            .iter()
            .filter(|t| t.success)
            .map(|t| t.tool_name.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        tools.sort();
        tools
    }

    /// Get list of failed tools
    pub fn tools_failed(&self) -> Vec<String> {
        let mut tools: Vec<String> = self
            .tool_executions
            .iter()
            .filter(|t| !t.success)
            .map(|t| t.tool_name.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        tools.sort();
        tools
    }

    /// Get detailed failed tool calls
    pub fn failed_tool_calls(&self) -> Vec<crate::agent::result::FailedToolCall> {
        self.tool_executions
            .iter()
            .filter(|t| !t.success)
            .map(|t| crate::agent::result::FailedToolCall {
                tool_name: t.tool_name.clone(),
                tool_use_id: t.tool_use_id.clone().unwrap_or_else(|| {
                    format!(
                        "execution_{}",
                        t.start_time.timestamp_nanos_opt().unwrap_or(0)
                    )
                }),
                error_message: t
                    .error
                    .clone()
                    .unwrap_or_else(|| "Unknown error".to_string()),
                duration: t.duration,
            })
            .collect()
    }

    /// Get summary statistics
    pub fn summary(&self) -> MetricsSummary {
        MetricsSummary {
            total_cycles: self.cycles.len() as u32,
            total_duration: self.total_duration,
            total_tokens: self.total_tokens.clone(),
            average_cycle_duration: if self.cycles.is_empty() {
                Duration::ZERO
            } else {
                self.total_duration / self.cycles.len() as u32
            },
            successful_tool_executions: self.tool_executions.iter().filter(|t| t.success).count()
                as u32,
            failed_tool_executions: self.tool_executions.iter().filter(|t| !t.success).count()
                as u32,
            unique_tools_used: self
                .tool_executions
                .iter()
                .map(|t| &t.tool_name)
                .collect::<std::collections::HashSet<_>>()
                .len() as u32,
        }
    }
}

/// Metrics for an individual cycle
#[derive(Debug, Clone)]
pub struct CycleMetrics {
    pub cycle_id: Uuid,
    pub duration: Duration,
    pub model_invocations: u32,
    pub tool_calls: u32,
    pub tokens_used: TokenUsage,
    pub trace_id: Option<String>,
    pub span_id: Option<String>,
    pub start_time: DateTime<Utc>,
    pub success: bool,
    pub error: Option<String>,
}

impl CycleMetrics {
    pub fn new(cycle_id: Uuid) -> Self {
        Self {
            cycle_id,
            duration: Duration::ZERO,
            model_invocations: 0,
            tool_calls: 0,
            tokens_used: TokenUsage::default(),
            trace_id: None,
            span_id: None,
            start_time: Utc::now(),
            success: false,
            error: None,
        }
    }

    pub fn complete_success(mut self, duration: Duration) -> Self {
        self.duration = duration;
        self.success = true;
        self
    }

    pub fn complete_error(mut self, duration: Duration, error: String) -> Self {
        self.duration = duration;
        self.success = false;
        self.error = Some(error);
        self
    }
}

/// Token usage information
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
}

impl TokenUsage {
    pub fn new(input_tokens: u32, output_tokens: u32) -> Self {
        Self {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
        }
    }

    pub fn add(&mut self, other: &TokenUsage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.total_tokens += other.total_tokens;
    }
}

/// Metrics for tool execution
#[derive(Debug, Clone)]
pub struct ToolExecutionMetric {
    pub tool_name: String,
    pub tool_use_id: Option<String>,
    pub duration: Duration,
    pub success: bool,
    pub error: Option<String>,
    pub trace_id: Option<String>,
    pub span_id: Option<String>,
    pub start_time: DateTime<Utc>,
    pub input_size_bytes: Option<usize>,
    pub output_size_bytes: Option<usize>,
}

/// Trace information for correlation
#[derive(Debug, Clone)]
pub struct TraceInfo {
    pub trace_id: String,
    pub span_id: String,
    pub operation: String,
    pub start_time: DateTime<Utc>,
    pub duration: Duration,
    pub status: SpanStatus,
    pub attributes: HashMap<String, String>,
}

/// Status of a telemetry span
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpanStatus {
    Ok,
    Error { message: String },
    Cancelled,
}

/// Accumulated metrics for summary reporting
#[derive(Debug, Clone, Default)]
pub struct AccumulatedMetrics {
    pub total_cycles: u32,
    pub total_model_invocations: u32,
    pub total_tool_calls: u32,
    pub total_processing_time: Duration,
    pub successful_cycles: u32,
    pub failed_cycles: u32,
}

/// Summary of metrics for quick reporting
#[derive(Debug, Clone)]
pub struct MetricsSummary {
    pub total_cycles: u32,
    pub total_duration: Duration,
    pub total_tokens: TokenUsage,
    pub average_cycle_duration: Duration,
    pub successful_tool_executions: u32,
    pub failed_tool_executions: u32,
    pub unique_tools_used: u32,
}

// ============================================================================
// Semantic conventions - kept for compatibility, will be replaced by genai.rs
// ============================================================================

/// GenAI semantic conventions
pub mod semantic_conventions {
    pub const GEN_AI_SYSTEM: &str = "gen_ai.system";
    pub const GEN_AI_REQUEST_MODEL: &str = "gen_ai.request.model";
    pub const GEN_AI_REQUEST_MAX_TOKENS: &str = "gen_ai.request.max_tokens";
    pub const GEN_AI_REQUEST_TEMPERATURE: &str = "gen_ai.request.temperature";
    pub const GEN_AI_REQUEST_TOP_P: &str = "gen_ai.request.top_p";
    pub const GEN_AI_RESPONSE_ID: &str = "gen_ai.response.id";
    pub const GEN_AI_RESPONSE_MODEL: &str = "gen_ai.response.model";
    pub const GEN_AI_RESPONSE_FINISH_REASONS: &str = "gen_ai.response.finish_reasons";
    pub const GEN_AI_RESPONSE_FINISH_REASON: &str = "gen_ai.response.finish_reason";
    pub const GEN_AI_RESPONSE_CONTENT_PREVIEW: &str = "gen_ai.response.content_preview";
    pub const GEN_AI_RESPONSE_CONTENT_LENGTH: &str = "gen_ai.response.content_length";
    pub const GEN_AI_RESPONSE_TOOL_CALLS_COUNT: &str = "gen_ai.response.tool_calls_count";
    pub const GEN_AI_RESPONSE_TOOL_NAMES: &str = "gen_ai.response.tool_names";
    pub const GEN_AI_RESPONSE_TYPE: &str = "gen_ai.response.type";
    pub const GEN_AI_USAGE_INPUT_TOKENS: &str = "gen_ai.usage.input_tokens";
    pub const GEN_AI_USAGE_OUTPUT_TOKENS: &str = "gen_ai.usage.output_tokens";
    pub const GEN_AI_USAGE_TOTAL_TOKENS: &str = "gen_ai.usage.total_tokens";
    pub const GEN_AI_OPERATION_NAME: &str = "gen_ai.operation.name";
    pub const GEN_AI_TOOL_NAME: &str = "gen_ai.tool.name";
    pub const STOOD_AGENT_ID: &str = "stood.agent.id";
    pub const STOOD_CONVERSATION_ID: &str = "stood.conversation.id";
    pub const STOOD_TOOL_EXECUTION_ID: &str = "stood.tool.execution_id";
    pub const STOOD_CYCLE_ID: &str = "stood.cycle.id";
    pub const STOOD_VERSION: &str = "stood.version";
    pub const SYSTEM_ANTHROPIC_BEDROCK: &str = "anthropic.bedrock";
    pub const OPERATION_CHAT: &str = "chat";
    pub const OPERATION_TOOL_CALL: &str = "tool_call";
    pub const OPERATION_AGENT_CYCLE: &str = "agent_cycle";
}

/// Timer for measuring operation duration
#[derive(Debug)]
pub struct Timer {
    start: Instant,
}

impl Timer {
    pub fn start(_label: impl Into<String>) -> Self {
        Self {
            start: Instant::now(),
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    pub fn finish(self) -> Duration {
        self.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_config_default() {
        let config = TelemetryConfig::default();
        assert!(!config.is_enabled());
        assert_eq!(config.service_name(), "stood-agent");
        assert_eq!(*config.log_level(), LogLevel::INFO);
    }

    #[test]
    fn test_telemetry_config_cloudwatch() {
        let config = TelemetryConfig::cloudwatch("us-east-1");
        assert!(config.is_enabled());
        assert_eq!(config.service_name(), "stood-agent");
        assert!(config.otlp_endpoint().is_some());
        assert!(config.otlp_endpoint().unwrap().contains("us-east-1"));
    }

    #[test]
    fn test_telemetry_config_validation() {
        // Disabled config should validate
        let config = TelemetryConfig::default();
        assert!(config.validate().is_ok());

        // CloudWatch config should validate
        let config = TelemetryConfig::cloudwatch("us-east-1");
        assert!(config.validate().is_ok());

        // CloudWatch config with custom service name should validate
        let config = TelemetryConfig::cloudwatch("us-west-2").with_service_name("my-agent");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_telemetry_config_set_log_level() {
        let mut config = TelemetryConfig::default();
        assert_eq!(*config.log_level(), LogLevel::INFO);

        config.set_log_level(LogLevel::DEBUG);
        assert_eq!(*config.log_level(), LogLevel::DEBUG);

        let mut cw_config = TelemetryConfig::cloudwatch("us-east-1");
        cw_config.set_log_level(LogLevel::TRACE);
        assert_eq!(*cw_config.log_level(), LogLevel::TRACE);
    }

    #[test]
    fn test_event_loop_metrics() {
        let mut metrics = EventLoopMetrics::new();
        let cycle = CycleMetrics::new(Uuid::new_v4()).complete_success(Duration::from_millis(100));
        metrics.add_cycle(cycle);

        let summary = metrics.summary();
        assert_eq!(summary.total_cycles, 1);
        assert_eq!(summary.total_duration, Duration::from_millis(100));
    }

    #[test]
    fn test_token_usage() {
        let mut usage = TokenUsage::new(100, 50);
        assert_eq!(usage.total_tokens, 150);

        let other = TokenUsage::new(25, 25);
        usage.add(&other);

        assert_eq!(usage.input_tokens, 125);
        assert_eq!(usage.output_tokens, 75);
        assert_eq!(usage.total_tokens, 200);
    }

    #[test]
    fn test_timer() {
        let timer = Timer::start("test");
        std::thread::sleep(Duration::from_millis(10));
        let elapsed = timer.finish();
        assert!(elapsed >= Duration::from_millis(10));
    }
}
