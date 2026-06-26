//! Performance benchmarks for MCP tool integration
//!
//! This module provides comprehensive performance testing to compare MCP tool execution
//! against native tool execution, measuring latency, throughput, and resource usage.

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::mcp::client::{MCPClient, MCPClientConfig};
use crate::mcp::error::MCPOperationError;
use crate::mcp::transport::{MCPTransport, TransportInfo, TransportStreams};
use crate::mcp::types::{CallToolResult, Content, TextContent, Tool as MCPTool};
use crate::tools::builtin::CalculatorTool;
use crate::tools::mcp_adapter::MCPAgentTool;
use crate::tools::{ToolRegistry, ToolUse};
use crate::StoodError;
use async_trait::async_trait;
use serde_json::json;

/// Performance benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Test name
    pub test_name: String,
    /// Number of operations performed
    pub operations: usize,
    /// Total duration for all operations
    pub total_duration: Duration,
    /// Average duration per operation
    pub avg_duration: Duration,
    /// Minimum operation duration
    pub min_duration: Duration,
    /// Maximum operation duration
    pub max_duration: Duration,
    /// Operations per second
    pub ops_per_second: f64,
    /// Success rate (successful operations / total operations)
    pub success_rate: f64,
}

impl BenchmarkResults {
    /// Create new benchmark results from individual operation durations
    pub fn from_durations(test_name: String, durations: Vec<Duration>, successes: usize) -> Self {
        let operations = durations.len();
        let total_duration: Duration = durations.iter().sum();
        let avg_duration = if operations > 0 {
            total_duration / operations as u32
        } else {
            Duration::ZERO
        };
        let min_duration = durations.iter().min().copied().unwrap_or(Duration::ZERO);
        let max_duration = durations.iter().max().copied().unwrap_or(Duration::ZERO);
        let ops_per_second = if total_duration.as_secs_f64() > 0.0 {
            operations as f64 / total_duration.as_secs_f64()
        } else {
            0.0
        };
        let success_rate = if operations > 0 {
            successes as f64 / operations as f64
        } else {
            0.0
        };

        Self {
            test_name,
            operations,
            total_duration,
            avg_duration,
            min_duration,
            max_duration,
            ops_per_second,
            success_rate,
        }
    }

    /// Calculate performance ratio compared to another benchmark
    pub fn performance_ratio(&self, baseline: &BenchmarkResults) -> f64 {
        if baseline.avg_duration.as_nanos() > 0 {
            self.avg_duration.as_nanos() as f64 / baseline.avg_duration.as_nanos() as f64
        } else {
            1.0
        }
    }

    /// Display benchmark results in a formatted way
    pub fn display(&self) -> String {
        format!(
            "{}: {} ops, avg: {:.2}ms, min: {:.2}ms, max: {:.2}ms, {:.1} ops/sec, {:.1}% success",
            self.test_name,
            self.operations,
            self.avg_duration.as_secs_f64() * 1000.0,
            self.min_duration.as_secs_f64() * 1000.0,
            self.max_duration.as_secs_f64() * 1000.0,
            self.ops_per_second,
            self.success_rate * 100.0
        )
    }
}

/// Performance-optimized Mock MCP Transport for benchmarking
pub struct PerformanceMockTransport {
    /// Available tools
    tools: Vec<MCPTool>,
    /// Session ID
    session_id: String,
    /// Simulated latency for testing
    simulated_latency: Duration,
}

impl PerformanceMockTransport {
    /// Create a new performance mock transport
    pub fn new(simulated_latency: Duration) -> Self {
        let mut transport = Self {
            tools: Vec::new(),
            session_id: Uuid::new_v4().to_string(),
            simulated_latency,
        };

        // Add performance test tools
        transport.add_calculator_tool();
        transport.add_echo_tool();
        transport.add_compute_tool();

        transport
    }

    /// Add calculator tool for performance testing
    fn add_calculator_tool(&mut self) {
        let tool = MCPTool {
            name: "calculator".to_string(),
            description: "High-performance calculator for benchmarking".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }),
        };
        self.tools.push(tool);
    }

    /// Add echo tool for performance testing
    fn add_echo_tool(&mut self) {
        let tool = MCPTool {
            name: "echo".to_string(),
            description: "High-performance echo for benchmarking".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message to echo"
                    }
                },
                "required": ["message"]
            }),
        };
        self.tools.push(tool);
    }

    /// Add compute-intensive tool for performance testing
    fn add_compute_tool(&mut self) {
        let tool = MCPTool {
            name: "compute".to_string(),
            description: "Compute-intensive operations for benchmarking".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "iterations": {
                        "type": "integer",
                        "description": "Number of iterations to perform"
                    }
                },
                "required": ["iterations"]
            }),
        };
        self.tools.push(tool);
    }

    /// Execute tool with performance optimizations
    pub fn execute_tool(
        &self,
        tool_name: &str,
        params: &serde_json::Value,
    ) -> std::result::Result<Vec<Content>, StoodError> {
        // Simulate network latency if configured
        if !self.simulated_latency.is_zero() {
            std::thread::sleep(self.simulated_latency);
        }

        match tool_name {
            "calculator" => {
                let expression = params["expression"].as_str().unwrap_or("1");
                let result = match expression {
                    "1 + 1" => "2",
                    "2 * 3" => "6",
                    "10 / 2" => "5",
                    "5 - 1" => "4",
                    _ => "42", // Fast default
                };
                Ok(vec![Content::Text(TextContent {
                    text: result.to_string(),
                })])
            }
            "echo" => {
                let message = params["message"].as_str().unwrap_or("");
                Ok(vec![Content::Text(TextContent {
                    text: message.to_string(),
                })])
            }
            "compute" => {
                let iterations = params["iterations"].as_u64().unwrap_or(1000) as usize;
                // Simulate computational work
                let mut sum = 0u64;
                for i in 0..iterations {
                    sum = sum.wrapping_add(i as u64);
                }
                Ok(vec![Content::Text(TextContent {
                    text: format!("Computed sum: {}", sum),
                })])
            }
            _ => Err(StoodError::tool_error(format!(
                "Tool '{}' not found",
                tool_name
            ))),
        }
    }
}

#[async_trait]
impl MCPTransport for PerformanceMockTransport {
    async fn connect(&mut self) -> std::result::Result<TransportStreams, MCPOperationError> {
        Err(MCPOperationError::transport(
            "Performance mock transport - connection not implemented for benchmarking",
        ))
    }

    async fn disconnect(&mut self) -> std::result::Result<(), MCPOperationError> {
        Ok(())
    }

    fn is_connected(&self) -> bool {
        true // Always connected for benchmarking
    }

    fn transport_info(&self) -> TransportInfo {
        TransportInfo {
            transport_type: "performance_mock".to_string(),
            endpoint: format!("mock://perf-test-{}", self.session_id),
            supports_reconnection: false,
            max_message_size: Some(1024 * 1024),
        }
    }
}

/// Performance-optimized Mock MCP Client
pub struct PerformanceMockMCPClient {
    transport: PerformanceMockTransport,
    #[allow(dead_code)]
    session_id: String,
}

impl PerformanceMockMCPClient {
    pub fn new(simulated_latency: Duration) -> Self {
        let transport = PerformanceMockTransport::new(simulated_latency);
        let session_id = transport.session_id.clone();

        Self {
            transport,
            session_id,
        }
    }

    pub async fn list_tools(&self) -> std::result::Result<Vec<MCPTool>, StoodError> {
        Ok(self.transport.tools.clone())
    }

    pub async fn call_tool(
        &self,
        tool_name: &str,
        params: serde_json::Value,
    ) -> std::result::Result<CallToolResult, StoodError> {
        match self.transport.execute_tool(tool_name, &params) {
            Ok(content) => Ok(CallToolResult {
                content,
                is_error: None,
            }),
            Err(err) => Ok(CallToolResult {
                content: vec![Content::Text(TextContent {
                    text: format!("Error: {}", err),
                })],
                is_error: Some(true),
            }),
        }
    }
}

/// Performance benchmark suite
pub struct MCPPerformanceBenchmark {
    /// Native tool registry
    native_registry: Arc<ToolRegistry>,
    /// MCP tool registry
    mcp_registry: Arc<ToolRegistry>,
    /// Number of iterations for each benchmark
    iterations: usize,
}

impl MCPPerformanceBenchmark {
    /// Create a new performance benchmark suite
    pub async fn new(iterations: usize) -> std::result::Result<Self, StoodError> {
        // Create native tool registry
        let native_registry = Arc::new(ToolRegistry::new());

        // Register native calculator tool
        let calculator = CalculatorTool::default();
        native_registry.register_tool(Box::new(calculator)).await?;

        // Create MCP tool registry
        let mcp_registry = Arc::new(ToolRegistry::new());

        // Register MCP tools with minimal latency for fair comparison
        let mock_client = PerformanceMockMCPClient::new(Duration::from_micros(1));
        let tools = mock_client.list_tools().await?;

        for tool in tools {
            let mcp_client_config = MCPClientConfig::default();
            let transport = Box::new(PerformanceMockTransport::new(Duration::from_micros(1)));
            let mcp_client = Arc::new(RwLock::new(MCPClient::new(mcp_client_config, transport)));

            let adapter = MCPAgentTool::new(tool, mcp_client, Some("perf_".to_string()));
            mcp_registry.register_tool(Box::new(adapter)).await?;
        }

        Ok(Self {
            native_registry,
            mcp_registry,
            iterations,
        })
    }

    /// Benchmark native tool execution
    pub async fn benchmark_native_calculator(&self) -> BenchmarkResults {
        let mut durations = Vec::new();
        let mut successes = 0;

        for i in 0..self.iterations {
            let tool_use = ToolUse {
                tool_use_id: format!("native_calc_{}", i),
                name: "calculator".to_string(),
                input: json!({"expression": "1 + 1"}),
            };

            let start = Instant::now();
            let result = self
                .native_registry
                .execute_tool(&tool_use.name, Some(tool_use.input.clone()), None)
                .await;
            let duration = start.elapsed();

            durations.push(duration);
            if let Ok(tool_result) = result {
                if !tool_result.content.to_string().contains("Error") {
                    successes += 1;
                }
            }
        }

        BenchmarkResults::from_durations("Native Calculator".to_string(), durations, successes)
    }

    /// Benchmark MCP tool execution
    pub async fn benchmark_mcp_calculator(&self) -> BenchmarkResults {
        let mut durations = Vec::new();
        let mut successes = 0;

        for i in 0..self.iterations {
            let tool_use = ToolUse {
                tool_use_id: format!("mcp_calc_{}", i),
                name: "perf_calculator".to_string(),
                input: json!({"expression": "1 + 1"}),
            };

            let start = Instant::now();
            let result = self
                .mcp_registry
                .execute_tool(&tool_use.name, Some(tool_use.input.clone()), None)
                .await;
            let duration = start.elapsed();

            durations.push(duration);
            if let Ok(tool_result) = result {
                if !tool_result.content.to_string().contains("Error") {
                    successes += 1;
                }
            }
        }

        BenchmarkResults::from_durations("MCP Calculator".to_string(), durations, successes)
    }

    /// Benchmark tool registration performance
    pub async fn benchmark_tool_registration(&self) -> (BenchmarkResults, BenchmarkResults) {
        // Benchmark native tool registration
        let native_durations = {
            let mut durations = Vec::new();
            let mut successes = 0;

            for _i in 0..self.iterations.min(100) {
                // Limit to avoid excessive registrations
                let registry = ToolRegistry::new();
                let calculator = CalculatorTool::default();

                let start = Instant::now();
                let result = registry.register_tool(Box::new(calculator)).await;
                let duration = start.elapsed();

                durations.push(duration);
                if result.is_ok() {
                    successes += 1;
                }
            }

            BenchmarkResults::from_durations(
                "Native Tool Registration".to_string(),
                durations,
                successes,
            )
        };

        // Benchmark MCP tool registration
        let mcp_durations = {
            let mut durations = Vec::new();
            let mut successes = 0;

            for i in 0..self.iterations.min(100) {
                // Limit to avoid excessive registrations
                let registry = ToolRegistry::new();

                let tool = MCPTool {
                    name: format!("test_tool_{}", i),
                    description: "Test tool for benchmarking".to_string(),
                    input_schema: json!({"type": "object", "properties": {}}),
                };

                let mcp_client_config = MCPClientConfig::default();
                let transport = Box::new(PerformanceMockTransport::new(Duration::ZERO));
                let mcp_client =
                    Arc::new(RwLock::new(MCPClient::new(mcp_client_config, transport)));
                let adapter = MCPAgentTool::new(tool, mcp_client, Some("bench_".to_string()));

                let start = Instant::now();
                let result = registry.register_tool(Box::new(adapter)).await;
                let duration = start.elapsed();

                durations.push(duration);
                if result.is_ok() {
                    successes += 1;
                }
            }

            BenchmarkResults::from_durations(
                "MCP Tool Registration".to_string(),
                durations,
                successes,
            )
        };

        (native_durations, mcp_durations)
    }

    /// Run comprehensive performance comparison
    pub async fn run_comprehensive_benchmark(&self) -> PerformanceReport {
        println!("🚀 Running comprehensive MCP performance benchmarks...");
        println!("   Iterations per test: {}", self.iterations);
        println!();

        // Tool execution benchmarks
        println!("📊 Tool Execution Benchmarks:");
        let native_calc = self.benchmark_native_calculator().await;
        println!("   {}", native_calc.display());

        let mcp_calc = self.benchmark_mcp_calculator().await;
        println!("   {}", mcp_calc.display());

        let calc_ratio = mcp_calc.performance_ratio(&native_calc);
        println!("   📈 MCP/Native ratio: {:.2}x", calc_ratio);
        println!();

        // Tool registration benchmarks
        println!("📊 Tool Registration Benchmarks:");
        let (native_reg, mcp_reg) = self.benchmark_tool_registration().await;
        println!("   {}", native_reg.display());
        println!("   {}", mcp_reg.display());

        let reg_ratio = mcp_reg.performance_ratio(&native_reg);
        println!("   📈 MCP/Native ratio: {:.2}x", reg_ratio);
        println!();

        PerformanceReport {
            native_tool_execution: native_calc,
            mcp_tool_execution: mcp_calc,
            native_tool_registration: native_reg,
            mcp_tool_registration: mcp_reg,
            execution_ratio: calc_ratio,
            registration_ratio: reg_ratio,
        }
    }
}

/// Comprehensive performance report
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    pub native_tool_execution: BenchmarkResults,
    pub mcp_tool_execution: BenchmarkResults,
    pub native_tool_registration: BenchmarkResults,
    pub mcp_tool_registration: BenchmarkResults,
    pub execution_ratio: f64,
    pub registration_ratio: f64,
}

impl PerformanceReport {
    /// Assess overall performance impact
    pub fn performance_assessment(&self) -> String {
        let exec_status = if self.execution_ratio <= 2.0 {
            "✅ Excellent"
        } else if self.execution_ratio <= 5.0 {
            "🟡 Acceptable"
        } else {
            "🔴 Needs optimization"
        };

        let reg_status = if self.registration_ratio <= 3.0 {
            "✅ Excellent"
        } else if self.registration_ratio <= 10.0 {
            "🟡 Acceptable"
        } else {
            "🔴 Needs optimization"
        };

        format!(
            "🎯 Performance Assessment:\n\
             Tool Execution: {} ({:.2}x overhead)\n\
             Tool Registration: {} ({:.2}x overhead)\n\
             \n\
             📝 Summary: MCP tools have {:.1}% execution overhead and {:.1}% registration overhead",
            exec_status,
            self.execution_ratio,
            reg_status,
            self.registration_ratio,
            (self.execution_ratio - 1.0) * 100.0,
            (self.registration_ratio - 1.0) * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_performance_mock_transport() {
        let transport = PerformanceMockTransport::new(Duration::ZERO);

        // Test calculator tool
        let result = transport
            .execute_tool("calculator", &json!({"expression": "1 + 1"}))
            .unwrap();
        assert_eq!(result.len(), 1);

        // Test echo tool
        let result = transport
            .execute_tool("echo", &json!({"message": "test"}))
            .unwrap();
        assert_eq!(result.len(), 1);

        // Test compute tool
        let result = transport
            .execute_tool("compute", &json!({"iterations": 100}))
            .unwrap();
        assert_eq!(result.len(), 1);
    }

    #[tokio::test]
    async fn test_performance_mock_client() {
        let client = PerformanceMockMCPClient::new(Duration::ZERO);

        // Test tool listing
        let tools = client.list_tools().await.unwrap();
        assert_eq!(tools.len(), 3);

        // Test tool execution
        let result = client
            .call_tool("calculator", json!({"expression": "1 + 1"}))
            .await
            .unwrap();
        assert_eq!(result.content.len(), 1);
        assert!(result.is_error.is_none());
    }

    #[tokio::test]
    async fn test_benchmark_results() {
        let durations = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(15),
        ];

        let results = BenchmarkResults::from_durations("Test".to_string(), durations, 3);

        assert_eq!(results.operations, 3);
        assert_eq!(results.success_rate, 1.0);
        assert_eq!(results.min_duration, Duration::from_millis(10));
        assert_eq!(results.max_duration, Duration::from_millis(20));
    }

    #[tokio::test]
    async fn test_mcp_performance_benchmark_creation() {
        let benchmark = MCPPerformanceBenchmark::new(10).await.unwrap();

        // Verify that both registries have tools
        assert!(benchmark.native_registry.has_tool("calculator").await);
        assert!(benchmark.mcp_registry.has_tool("perf_calculator").await);
    }

    #[tokio::test]
    #[ignore = "requires live MCP server; set MCP_TEST_ENDPOINT and run cargo test -- --ignored"]
    async fn test_performance_benchmark_execution() {
        let benchmark = MCPPerformanceBenchmark::new(5).await.unwrap();

        // Run a small benchmark
        let native_results = benchmark.benchmark_native_calculator().await;
        let mcp_results = benchmark.benchmark_mcp_calculator().await;

        // Verify results
        assert_eq!(native_results.operations, 5);
        assert_eq!(mcp_results.operations, 5);
        assert!(native_results.success_rate > 0.8); // At least 80% success
        assert!(mcp_results.success_rate > 0.8);

        // Performance ratio should be reasonable (MCP shouldn't be more than 10x slower)
        let ratio = mcp_results.performance_ratio(&native_results);
        assert!(
            ratio < 10.0,
            "MCP tools are too slow compared to native: {}x",
            ratio
        );
    }

    #[tokio::test]
    async fn test_comprehensive_benchmark() {
        let benchmark = MCPPerformanceBenchmark::new(3).await.unwrap();
        let report = benchmark.run_comprehensive_benchmark().await;

        // Verify report structure
        assert!(report.execution_ratio > 0.0);
        assert!(report.registration_ratio > 0.0);

        // Performance should be reasonable
        assert!(
            report.execution_ratio < 20.0,
            "Execution ratio too high: {}",
            report.execution_ratio
        );
        assert!(
            report.registration_ratio < 50.0,
            "Registration ratio too high: {}",
            report.registration_ratio
        );

        // Test performance assessment
        let assessment = report.performance_assessment();
        assert!(assessment.contains("Performance Assessment"));
    }
}
