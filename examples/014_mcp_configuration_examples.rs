//! Example 014: NEW Simple MCP Configuration Methods
//!
//! This file demonstrates the NEW simple way to configure MCP servers using
//! with_mcp_client() and with_mcp_clients() builder methods that match Python's approach.

use stood::agent::Agent;
use stood::mcp::transport::{StdioConfig, TransportFactory, WebSocketConfig};
use stood::mcp::{MCPClient, MCPClientConfig};

/// 1. STDIO-based MCP Server Configuration using NEW simple builder method
async fn configure_stdio_mcp_server() -> Result<(), Box<dyn std::error::Error>> {
    // Configure the MCP server transport (matches your uvx command)
    let stdio_config = StdioConfig {
        command: "uvx".to_string(),
        args: vec!["awslabs.core-mcp-server@latest".to_string()],
        env_vars: [("FASTMCP_LOG_LEVEL".to_string(), "ERROR".to_string())].into(),
        working_dir: None,
        startup_timeout_ms: 10_000,
        max_message_size: Some(16 * 1024 * 1024),
    };

    // Create transport and MCP client
    let transport = TransportFactory::stdio(stdio_config);
    let config = MCPClientConfig {
        client_name: "stood-agent".to_string(),
        client_version: "0.1.0".to_string(),
        request_timeout_ms: 30_000,
        max_concurrent_requests: 10,
        auto_reconnect: true,
        reconnect_delay_ms: 5_000,
        ..Default::default()
    };

    let mut mcp_client = MCPClient::new(config, transport);

    // Connect the client
    mcp_client.connect().await?;
    println!("✅ Connected to stdio MCP server");

    // Create agent with NEW simple MCP integration
    let mut agent = Agent::builder()
        .provider("bedrock")
        .model_str("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .system_prompt(
            "You are a helpful assistant with access to MCP tools. Use them when appropriate.",
        )
        .with_mcp_client(mcp_client, Some("aws_".to_string()))
        .await?
        .build()
        .await?;

    println!("🤖 Agent created with AWS MCP tools (aws_ namespace)");

    // Test the agent
    let result = agent
        .execute("List some AWS services using your tools")
        .await?;
    println!("📄 Response: {}", result.response);

    Ok(())
}

/// 2. WebSocket-based MCP Server Configuration using NEW simple builder method
async fn configure_websocket_mcp_server() -> Result<(), Box<dyn std::error::Error>> {
    // Try to connect to local WebSocket MCP server
    // To test this: run `python examples/test-servers/websocket_mcp_server.py` in another terminal
    let ws_config = WebSocketConfig {
        url: "ws://localhost:8765".to_string(),
        connect_timeout_ms: 5_000, // Shorter timeout for local testing
        ping_interval_ms: None,    // Disable ping for simple testing
        max_message_size: Some(16 * 1024 * 1024),
        headers: std::collections::HashMap::new(), // No auth needed for local testing
    };

    let transport = TransportFactory::websocket(ws_config);
    let mut mcp_client = MCPClient::new(MCPClientConfig::default(), transport);

    // Connect the client
    mcp_client.connect().await?;
    println!("✅ Connected to WebSocket MCP server");

    // Create agent with NEW simple MCP integration
    let mut agent = Agent::builder()
        .provider("bedrock")
        .model_str("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .system_prompt("You are a helpful assistant with access to WebSocket MCP tools.")
        .with_mcp_client(mcp_client, Some("ws_".to_string()))
        .await?
        .build()
        .await?;

    println!("🤖 Agent created with WebSocket MCP tools (ws_ namespace)");

    // Test the agent
    let result = agent
        .execute("Use the ws_websocket_search tool to search for 'Rust programming'")
        .await?;
    println!("📄 Response: {}", result.response);

    Ok(())
}

/// 3. MULTIPLE MCP Servers using NEW with_mcp_clients() method (Python-like)
async fn agent_with_multiple_mcp_servers() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔗 MULTIPLE MCP SERVERS DEMO");
    println!("Intent: Shows how to configure an agent with multiple MCP servers using with_mcp_clients()");
    println!("Expected behavior: STDIO server will work, WebSocket will fail (no server running)");
    println!("Demonstrates: Graceful handling when some servers are unavailable");
    println!();

    // Set up STDIO MCP client (this should work)
    let stdio_config = StdioConfig {
        command: "uvx".to_string(),
        args: vec!["awslabs.core-mcp-server@latest".to_string()],
        env_vars: [("FASTMCP_LOG_LEVEL".to_string(), "ERROR".to_string())].into(),
        ..Default::default()
    };
    let mut stdio_client = MCPClient::new(
        MCPClientConfig::default(),
        TransportFactory::stdio(stdio_config),
    );
    stdio_client.connect().await?;
    println!("✅ Connected to STDIO MCP server (as expected)");

    // Set up WebSocket MCP client (this will likely fail unless you're running a test server)
    let ws_config = WebSocketConfig {
        url: "ws://localhost:8765".to_string(),
        connect_timeout_ms: 3_000, // Short timeout for demo
        ping_interval_ms: None,
        max_message_size: Some(16 * 1024 * 1024),
        headers: std::collections::HashMap::new(),
    };
    let mut ws_client = MCPClient::new(
        MCPClientConfig::default(),
        TransportFactory::websocket(ws_config),
    );

    println!("🔄 Attempting to connect to WebSocket MCP server at ws://localhost:8765...");
    println!("   (This will likely fail unless you're running examples/test-servers/websocket_mcp_server.py)");

    // Try to connect to WebSocket (expected to fail in most cases)
    match ws_client.connect().await {
        Ok(_) => {
            println!("✅ Connected to WebSocket MCP server (unexpected but great!)");

            // If both servers work, demonstrate the full multi-server setup
            let mut agent = Agent::builder()
                .provider("bedrock")
                .model_str("us.anthropic.claude-haiku-4-5-20251001-v1:0")
                .system_prompt("You are an assistant with access to both AWS tools and WebSocket tools via MCP.")
                .with_mcp_clients(vec![
                    (stdio_client, Some("aws_".to_string())),  // AWS tools with aws_ prefix
                    (ws_client, Some("ws_".to_string())),      // WebSocket tools with ws_ prefix
                ]).await?
                .build().await?;

            println!("🤖 Agent created with MULTIPLE MCP servers (aws_ and ws_ namespaces)");
            let result = agent.execute("Use your AWS tools to list some services, and if available, use WebSocket tools too").await?;
            println!("📄 Multi-server response: {}", result.response);
            return Ok(());
        }
        Err(e) => {
            println!(
                "❌ Failed to connect to WebSocket MCP server (expected): {}",
                e
            );
            println!("💡 This demonstrates graceful fallback - continuing with just STDIO server");
            println!();

            // Create agent with just STDIO client (demonstrates fallback pattern)
            let mut agent = Agent::builder()
                .provider("bedrock")
                .model_str("us.anthropic.claude-haiku-4-5-20251001-v1:0")
                .system_prompt("You are an assistant with access to AWS tools via MCP.")
                .with_mcp_client(stdio_client, Some("aws_".to_string()))
                .await?
                .build()
                .await?;

            println!("🤖 Agent created with single working MCP server (graceful degradation)");
            let result = agent
                .execute("List some AWS tools you have access to")
                .await?;
            println!("📄 Response: {}", result.response);
            return Ok(());
        }
    }
}

/// 4. Configuration from Environment Variables
async fn mcp_from_environment() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌍 ENVIRONMENT VARIABLE CONFIGURATION DEMO");
    println!("Intent: Shows how to configure MCP servers using environment variables");
    println!("Use case: Production deployments where configuration varies by environment");
    println!("Benefits: Avoids hardcoded server details, enables deployment flexibility");
    println!();

    // Demonstrate reading configuration from environment variables
    let command = std::env::var("MCP_SERVER_COMMAND").unwrap_or_else(|_| "uvx".to_string());
    let args_str = std::env::var("MCP_SERVER_ARGS")
        .unwrap_or_else(|_| "awslabs.core-mcp-server@latest".to_string());
    let args: Vec<String> = args_str.split_whitespace().map(|s| s.to_string()).collect();
    let log_level = std::env::var("MCP_LOG_LEVEL").unwrap_or_else(|_| "ERROR".to_string());

    println!("📋 Reading configuration from environment:");
    println!("   MCP_SERVER_COMMAND = '{}' (fallback: 'uvx')", command);
    println!(
        "   MCP_SERVER_ARGS = '{}' (fallback: 'awslabs.core-mcp-server@latest')",
        args_str
    );
    println!("   MCP_LOG_LEVEL = '{}' (fallback: 'ERROR')", log_level);
    println!();

    println!("💡 To test with custom values, run:");
    println!("   MCP_SERVER_COMMAND=python MCP_SERVER_ARGS='-m my_server' cargo run --example 014");
    println!();

    let config = StdioConfig {
        command: command.clone(),
        args: args.clone(),
        env_vars: [("FASTMCP_LOG_LEVEL".to_string(), log_level.clone())].into(),
        ..Default::default()
    };

    let transport = TransportFactory::stdio(config);
    let mut client = MCPClient::new(MCPClientConfig::default(), transport);

    match client.connect().await {
        Ok(_) => {
            println!("✅ Successfully connected using environment configuration!");
            println!("📊 Active configuration: {} {}", command, args.join(" "));
        }
        Err(e) => {
            println!(
                "❌ Connection failed (expected if server not available): {}",
                e
            );
            println!("📊 Attempted configuration: {} {}", command, args.join(" "));
        }
    }

    Ok(())
}

/// 5. Server IP and Port Configuration (WebSocket)
async fn mcp_server_ip_port() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌐 REMOTE SERVER IP:PORT CONFIGURATION DEMO");
    println!("Intent: Demonstrates how to configure MCP clients for different network scenarios");
    println!("Use case: Distributed systems where MCP servers run on dedicated hosts");
    println!("Benefits: Enables network-distributed MCP architectures, load balancing");
    println!(
        "Note: These are EXAMPLE configurations showing different patterns - servers don't exist"
    );
    println!();

    // Example configurations for different network scenarios (for demonstration only)
    let server_configs = vec![
        ("192.168.1.100", 8080, false, "Local network MCP server"),
        (
            "mcp-server.company.com",
            443,
            true,
            "Corporate MCP server with TLS",
        ),
        (
            "10.0.0.50",
            9090,
            false,
            "Docker/Kubernetes internal service",
        ),
    ];

    println!("📋 Configuration patterns (demonstration only - connections will fail):");

    for (ip, port, use_tls, description) in server_configs {
        let protocol = if use_tls { "wss" } else { "ws" };
        let url = format!("{}://{}:{}/mcp", protocol, ip, port);

        println!();
        println!("🔗 Configuration Pattern: {}", description);
        println!("   URL: {}", url);
        println!(
            "   TLS: {}",
            if use_tls {
                "Enabled (secure)"
            } else {
                "Disabled (testing only)"
            }
        );
        println!("   ⚠️ This is a demo configuration - server doesn't exist");

        // Show the configuration without actually trying to connect (since these are fake servers)
        let config = WebSocketConfig {
            url: url.clone(),
            connect_timeout_ms: 2_000, // Very short timeout since we know they'll fail
            ..Default::default()
        };

        println!("   📋 WebSocketConfig created successfully");
        println!("      - connect_timeout_ms: {}", config.connect_timeout_ms);
        println!("      - max_message_size: {:?}", config.max_message_size);

        // We could create the transport and client, but won't try to connect to fake servers
        let _transport = TransportFactory::websocket(config);
        let _client = MCPClient::new(MCPClientConfig::default(), _transport);

        println!("   ✅ MCP client configured for this URL pattern");
    }

    println!();
    println!("💡 Network configuration considerations:");
    println!("   • Use TLS (wss://) for production and external networks");
    println!("   • Configure firewalls to allow WebSocket connections");
    println!("   • Consider load balancers for high-availability setups");
    println!("   • Use service discovery for dynamic server addresses");

    Ok(())
}

/// 6. Error Handling and Graceful Fallback
async fn mcp_with_error_handling() -> Result<(), Box<dyn std::error::Error>> {
    println!("🛡️ ERROR HANDLING AND FALLBACK DEMO");
    println!("Intent: Shows robust MCP configuration with failover mechanisms");
    println!("Use case: Production systems requiring high availability");
    println!("Benefits: Ensures system continues working even when primary MCP servers fail");
    println!();

    // Configure multiple fallback servers in priority order
    let configs = vec![
        (
            "Primary MCP Server",
            StdioConfig {
                command: "uvx".to_string(),
                args: vec!["awslabs.core-mcp-server@latest".to_string()],
                env_vars: [("FASTMCP_LOG_LEVEL".to_string(), "ERROR".to_string())].into(),
                startup_timeout_ms: 10_000,
                ..Default::default()
            },
        ),
        (
            "Backup Python Server",
            StdioConfig {
                command: "python".to_string(),
                args: vec!["-m".to_string(), "backup_mcp_server".to_string()],
                startup_timeout_ms: 5_000,
                ..Default::default()
            },
        ),
        (
            "Local Development Server",
            StdioConfig {
                command: "node".to_string(),
                args: vec!["local-mcp-server.js".to_string()],
                startup_timeout_ms: 3_000,
                ..Default::default()
            },
        ),
    ];

    println!("📋 Attempting connection with failover strategy:");

    for (i, (description, config)) in configs.into_iter().enumerate() {
        println!();
        println!("🔄 Attempt {}: {}", i + 1, description);
        println!("   Command: {} {}", config.command, config.args.join(" "));
        println!("   Timeout: {}ms", config.startup_timeout_ms);

        let transport = TransportFactory::stdio(config);
        let mut client = MCPClient::new(MCPClientConfig::default(), transport);

        match client.connect().await {
            Ok(()) => {
                println!("   ✅ Successfully connected to {}", description);
                println!("   🎯 Failover strategy worked - system operational!");

                // Demonstrate that we could now create an agent with this working client
                println!("   📊 MCP client is ready for agent integration");
                return Ok(());
            }
            Err(e) => {
                println!("   ❌ Connection failed: {}", e);
                println!("   ⏭️ Trying next fallback server...");
            }
        }
    }

    println!();
    println!("🚨 All MCP servers failed to connect");
    println!("✅ System continues operation without MCP tools (graceful degradation)");
    println!();
    println!("💡 Production recommendations:");
    println!("   • Implement health checks for MCP servers");
    println!("   • Use monitoring/alerting for MCP connectivity issues");
    println!("   • Consider retry logic with exponential backoff");
    println!("   • Log all connection attempts for debugging");
    println!("   • Provide alternative functionality when MCP tools unavailable");

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("NEW Simple MCP Configuration Examples for Stood Library");
    println!("=======================================================");
    println!("Demonstrating NEW with_mcp_client() and with_mcp_clients() methods!");

    println!("\n1. NEW Simple STDIO MCP Integration:");
    if let Err(e) = configure_stdio_mcp_server().await {
        println!(
            "   Failed: {} (This is expected if the server isn't available)",
            e
        );
    }

    println!("\n2. NEW Simple WebSocket MCP Integration:");
    if let Err(e) = configure_websocket_mcp_server().await {
        println!(
            "   Failed: {} (This is expected if the server isn't available)",
            e
        );
    }

    println!("\n3. NEW Multiple MCP Servers with Single Builder Call:");
    if let Err(e) = agent_with_multiple_mcp_servers().await {
        println!(
            "   Failed: {} (This is expected if servers aren't available)",
            e
        );
    }

    println!("\n4. Environment-based Configuration:");
    if let Err(e) = mcp_from_environment().await {
        println!(
            "   Failed: {} (This is expected if the server isn't available)",
            e
        );
    }

    println!("\n5. IP:Port Configuration:");
    if let Err(e) = mcp_server_ip_port().await {
        println!(
            "   Failed: {} (This is expected if the server isn't available)",
            e
        );
    }

    println!("\n6. Error Handling Example:");
    if let Err(e) = mcp_with_error_handling().await {
        println!("   Failed: {}", e);
    }

    println!("\n🎉 MCP Configuration Examples Complete!");
    println!("========================================");
    println!("🔄 Key improvements with NEW methods:");
    println!("   • One-line MCP integration: .with_mcp_client(client, namespace)");
    println!("   • Multiple servers: .with_mcp_clients(vec![(client1, ns1), (client2, ns2)])");
    println!("   • Automatic tool discovery and namespace prefixing");
    println!("   • Python-like simplicity in Rust");
    println!("   • No more manual tool adapter creation!");

    Ok(())
}
