//! Model Context Protocol (MCP) implementation with simplified agent integration.
//!
//! This module enables you to integrate with external data sources and tools through
//! the standardized Model Context Protocol. You'll get seamless communication with
//! MCP servers for tool discovery, execution, and resource management with one-line
//! agent integration that matches Python's simplicity.
//!
//! # Quick Start
//!
//! Simplified agent integration (recommended approach):
//! ```no_run
//! use stood::agent::Agent;
//! use stood::mcp::{MCPClient, MCPClientConfig};
//! use stood::mcp::transport::{TransportFactory, StdioConfig};
//! use stood::llm::models::Bedrock;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Configure MCP client
//! let config = StdioConfig {
//!     command: "uvx".to_string(),
//!     args: vec!["awslabs.core-mcp-server@latest".to_string()],
//!     env_vars: [("FASTMCP_LOG_LEVEL".to_string(), "ERROR".to_string())].into(),
//!     ..Default::default()
//! };
//!
//! let transport = TransportFactory::stdio(config);
//! let mut mcp_client = MCPClient::new(MCPClientConfig::default(), transport);
//! mcp_client.connect().await?;
//!
//! // One-line agent integration with namespace prefixing
//! let mut agent = Agent::builder()
//!     .provider("bedrock").model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
//!     .system_prompt("You are a helpful assistant with access to MCP tools.")
//!     .with_mcp_client(mcp_client, Some("aws_".to_string())).await?
//!     .build().await?;
//!
//! // All MCP tools are now available with aws_ prefix
//! let result = agent.execute("List my AWS resources").await?;
//! println!("Agent response: {}", result.response);
//! # Ok(())
//! # }
//! ```
//!
//! Direct MCP client usage (for advanced scenarios):
//! ```no_run
//! use stood::mcp::{MCPClient, MCPClientConfig};
//! use stood::mcp::transport::{TransportFactory, StdioConfig};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let stdio_config = StdioConfig {
//!     command: "python".to_string(),
//!     args: vec!["-m".to_string(), "my_mcp_server".to_string()],
//!     ..Default::default()
//! };
//!
//! let transport = TransportFactory::stdio(stdio_config);
//! let mut client = MCPClient::new(MCPClientConfig::default(), transport);
//!
//! client.connect().await?;
//! let tools = client.list_tools().await?;
//! println!("Available tools: {}", tools.len());
//!
//! // Call a tool directly
//! let result = client.call_tool("my_tool", Some(serde_json::json!({
//!     "parameter": "value"
//! }))).await?;
//! println!("Tool result: {:?}", result);
//! # Ok(())
//! # }
//! ```
//!
//! # Architecture
//!
//! The MCP implementation consists of three main components:
//!
//! - **Transport Layer** - WebSocket and stdio communication channels
//! - **Protocol Layer** - JSON-RPC 2.0 message handling and serialization
//! - **Client/Server** - High-level MCP session management and tool execution
//!
//! See [MCP documentation](../docs/architecture.wiki#mcp) for detailed design information.
//!
//! # Key Features
//!
//! - **Simplified Integration** - One-line agent integration with `Agent::with_mcp_client()`
//! - **Namespace Support** - Tool prefixing to prevent conflicts with multiple MCP servers
//! - **Multiple Transports** - WebSocket for network servers, stdio for local processes
//! - **Session Management** - Automatic connection handling and error recovery
//! - **Type Safety** - Full Rust type system integration with MCP message formats
//! - **Tool Discovery** - Automatic schema validation and tool registration
//!
//! # Performance
//!
//! - Message parsing: <1ms for typical MCP messages
//! - WebSocket connections: Support for concurrent clients up to system limits
//! - Stdio processes: Efficient line-buffered communication
//! - Memory usage: Minimal overhead with zero-copy where possible
//!
//! # Transport Support
//!
//! This implementation supports the official MCP transport mechanisms:
//!
//! - **WebSocket** - For network-based MCP servers with reconnection support
//! - **Stdio** - For process-based MCP servers with lifecycle management

pub mod client;
pub mod error;
pub mod server;
pub mod test_utils;
pub mod transport;
pub mod types;

// Re-export main client types for easy access
pub use client::{MCPClient, MCPClientConfig};

// Re-export comprehensive error types for proper error handling
pub use error::{JsonRpcError, MCPOperationError, SessionError, TransportError};

// Re-export server implementation types
pub use server::{MCPServer, MCPServerConfig, MCPServerHandler, StoodMCPServer};

// Re-export transport implementations and configurations
pub use transport::{
    MCPTransport, StdioConfig, StdioTransport, TransportFactory, TransportInfo, TransportStreams,
    WebSocketConfig,
};

// Re-export all MCP protocol types and message structures
pub use types::*;
