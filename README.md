# <img src="stood-icon.png" width="80" height="80" style="vertical-align: middle;"> Stood Agent Library

The Stood Agent library is an agent framework that lets Large Language Models(LLMs) autonomously execute tasks while providing developers with control and monitoring capabilities.

This Rust-based AI multi-agent framework with multi-model support is an implementation inspired by AWS' Strands Agent library. The project incorporates key architectural concepts from Strands Agent while introducing its own variations — it reinterprets the core design principles without aiming for complete feature parity or serving as a direct replacement.


## ⚠️ Important Notice

This project is neither supported by nor an official AWS project. It is an independent implementation and is provided as-is. **This software is distributed in an Alpha state, and its functionality may be incomplete or subject to change.**

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
stood = { git = "https://github.com/fibanez/stood.git" }
```

## Documentation

For comprehensive documentation, examples, and guides, see the [Documentation](docs/README.md). For core API reference, see the [API Documentation](docs/api.md).

## Overview

Stood is an AI multi-agent framework that provides:

- **Multi-Model Support** - AWS Bedrock (Claude 4.5 Sonnet/Haiku/Opus, Mistral Large 2/3, Nova Lite/Pro/Micro/Premier, Nova 2 Lite/Pro), LM Studio (local models), with additional providers in development
- **Type-Safe Tools** - Compile-time validation of tool parameters with Rust's type system
- **Agentic Execution** - Agents can autonomously chain tools and make decisions to complete complex tasks
- **Enterprise Features** - Comprehensive error handling, observability, and performance optimization, with optional zero-configuration telemetry

## Key Features

### Core Components
- **Agent Module** - Core agent implementation with conversation management
- **Tools System** - Unified tool system with compile-time validation and MCP integration
- **Built-in Tools** - Ready-to-use tools including Think tool for structured problem-solving, calculator, file operations, HTTP requests, and system utilities
- **Built-in Evaluator Support** - Force evaluation at end of agent cycle with task evaluation, agent-based evaluation, and multi-perspective strategies
- **Multi-Model Client** - Native integration with AWS Bedrock and other providers
- **OpenTelemetry Integration** - Enterprise-grade observability with OTEL standards
- **MCP Support** - Model Context Protocol integration for external tools

### Why Rust for Agentic Work?

*   **Rust First**: A native Rust library that brings agentic functionality directly to Rust applications without external dependencies.
*   **Small Executables:** Rust compiles to highly optimized binaries, resulting in small executable sizes. This is particularly advantageous for constrained environments like small devices, Lambda and Lambda@Edge Functions, and containers, leading to faster startup times and reduced resource consumption – translating to improved performance and lower costs in cloud deployments.
*   **Memory Safety:**  Rust's ownership system eliminates common memory-related errors at compile time, enhancing the reliability and security of agent operations.
*   **Concurrency & Parallelism:** Rust’s robust concurrency primitives facilitate efficient tool calling and parallel processing, enabling complex workflows to execute with minimal overhead.
*   **Exponential Scalability**: Multi-agent systems consume 15x more resources than single agents [(1)](https://www.anthropic.com/engineering/built-multi-agent-research-system), but Rust deployments achieve 60-80% memory reduction and 3-50x performance improvements over Python equivalents [(2)](https://www.linkedin.com/pulse/python-productivity-rust-performance-match-made-heaven-nilay-parikh-c08zf), making Rust essential for managing hundreds of specialized agents at scale with long-running processes.
*   **Network-First Architecture**: As Model Context Protocol (MCP) evolves from local to remote servers [(3)](https://www.anthropic.com/news/integrations), Rust's measured networking performance becomes a key differentiator. With Tokio's async runtime and zero-copy capabilities, Rust efficiently handles the latency challenges of RPC-based architectures while maintaining performance necessary for real-time agent interactions. 

## Quick Start

```rust
use stood::agent::Agent;        // Core agent builder and execution
use stood::tool;                // Macro for creating custom tools

#[tool]
/// Calculate the result of a mathematical expression
async fn calculate(expression: String) -> Result<f64, String> {
    // TODO: Implement expression parsing
    // For demonstration, returning a fixed value
    Ok(42.0)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create an agent with default settings (uses Claude Haiku 4.5)
    let agent = Agent::builder()
        .tools(vec![calculate()])
        .build()
        .await?;

    // Execute agentic workflow
    let response = agent.execute("Calculate 25 * 17").await?;
    println!("{}", response);

    Ok(())
}
```

### With Custom Configuration

```rust
use stood::agent::Agent;                    // Core agent builder and execution
// No model import needed — use plain strings
use stood::mcp::{MCPClient, MCPClientConfig}; // MCP client for external tools
use stood::mcp::transport::{TransportFactory, WebSocketConfig}; // MCP transport layer
use stood::tool;                           // Macro for creating custom tools

#[tool]
/// Get weather information for a location
async fn get_weather(location: String) -> Result<String, String> {
    // TODO: Implement weather API call
    // For demonstration, returning a fixed value
    Ok(format!("Sunny, 75°F in {}", location))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure connection to remote MCP server
    let mcp_config = WebSocketConfig {
        url: "wss://mcp-tools.acme.com/api".to_string(),
        connect_timeout_ms: 10_000,
        ..Default::default()
    };
    // Configure MCP Client and connect
    let mut mcp_client = MCPClient::new(
        MCPClientConfig::default(), 
        TransportFactory::websocket(mcp_config)
    );
    mcp_client.connect().await?;

    // Create an agent with custom configuration and remote MCP tools
    let agent = Agent::builder()
        .provider("bedrock")
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .system_prompt("You are a helpful assistant with access to Acme Corp tools")
        .temperature(0.7)
        .max_tokens(1000)
        .tools(vec![get_weather()])
        .with_mcp_client(mcp_client, Some("acme_".to_string())).await?
        .build()
        .await?;

    let response = agent.execute("What's the weather like in San Francisco? and What are the latest news from Acme?").await?;
    println!("{}", response);

    Ok(())
}
```

## Autonomous Agent Evaluation

Stood includes an **optional LLM-controlled continuation system** that enables autonomous multi-cycle agents. **By default, agents execute one cycle and let the model decide naturally when it wants to continue.** You can enable explicit evaluation strategies for more complex continuation logic.

### Evaluation Strategies

| Strategy              | Mode  | Description                                                                                                                                    | Use Case                                                          |
|-----------------------|-------|------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| **Model-Driven**      | model | **Defaults** - No explicit evaluation, model decides continuation naturally (single-cycle). Developer can use Think tool to enhance this mode. | Natural interaction, simple tasks, letting the model control flow |
| **Task Evaluation**   | task  | Agent evaluates user intent satisfaction and task completion (multi-cycle)                                                                     | Most tasks, quality assurance, iterative refinement               |
| **Agent-Based**       | agent | Separate evaluator agent independently assesses work quality and completion                                                                    | Code review, complex evaluation logic, specialized domains        |
| **Multi-Perspective** | multi | Multiple evaluation viewpoints with weighted scoring for comprehensive assessment                                                              | Content creation, comprehensive analysis, balanced evaluation     |

**Note:** Self-reflection and chain-of-thought patterns can be achieved using **Task Evaluation** with custom prompts (e.g., "Let me think step by step..." or "Reflect on the quality of my work...").

### Evaluation Examples

```rust
// Model-Driven (DEFAULT) - Single-cycle execution, model decides naturally
let agent = Agent::builder()
    // Optionally use the think tool for LLM-directed evaluation of next steps
    .with_think_tool("Use the tool to think about something. It will not obtain new information, but just append the thought to the log. Use it when complex reasoning is needed.")
    .build().await?;  // Uses model-driven by default!

// Task Evaluation - Multi-cycle execution with user intent focus
let agent = Agent::builder()
    // Forces an extra evalution with the provided prompt when end of cycle is reached
    .with_task_evaluation("Have I fully satisfied the user's request? What aspects could be improved?")
    .build().await?;

// Task Evaluation with chain-of-thought style
let agent = Agent::builder()
    // Forces an additional evalution with the provided prompt when end of cycle is reached
    .with_task_evaluation("Let me think step by step: 1) What did the user ask for? 2) What have I accomplished? 3) What's missing? 4) Is the task complete?")
    .build().await?;

// Agent-based evaluation with specialized evaluator
// First create the evaluator agent
let evaluator = Agent::builder()
    .provider("bedrock")
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
    .system_prompt("You are a critical evaluator. Assess task completion quality.")
    .build().await?;

// Then attach the evaluator to the main agent
let main_agent = Agent::builder()
    .with_agent_based_evaluation(evaluator)
    .build().await?;

// Multi-perspective evaluation with weighted scoring
// First create the multi-perspective evaluation configuration
let perspectives = vec![
    PerspectiveConfig {
        name: "quality_check".to_string(),
        prompt: "As a quality analyst, is the work complete and high-quality?".to_string(),
        weight: 0.6,
    },
    PerspectiveConfig {
        name: "user_satisfaction".to_string(),
        prompt: "From a user's perspective, does this fully address their needs?".to_string(),
        weight: 0.4,
    },
];
// Then attach the perspective configuration to the main agent  
let agent = Agent::builder()
    .with_multi_perspective_evaluation(perspectives)
    .build().await?;
```

### Benefits

- **Natural by Default**: Model-driven execution lets the model decide continuation naturally
- **Optional Multi-Cycle**: Enable explicit evaluation strategies when you need multi-cycle behavior
- **User-Focused**: Evaluation strategies center on completing the user's actual request, not arbitrary metrics
- **Flexible Architecture**: Choose from model-driven, task evaluation, agent-based, or multi-perspective strategies
- **Simple Configuration**: Clear API with explicit opt-in for evaluation complexity
- **Observable**: Full telemetry and logging of evaluation decisions and reasoning


## Examples

The examples are organized by complexity:

### Basic Examples
- [001\_tool\_macro](examples/001_tool_macro.rs) - Custom tools with #[tool] macro
- [002\_tool\_decorator\_registry](examples/002_tool_decorator_registry.rs) - Tool decorator with registry
- [003\_interactive\_chat\_simple](examples/003_interactive_chat_simple.rs) - Simple interactive chat
- [004\_streaming\_simple](examples/004_streaming_simple.rs) - Simple streaming response handling
- [005\_callbacks\_basic](examples/005_callbacks_basic.rs) - Basic callback patterns
- [006\_callback\_system\_demo](examples/006_callback_system_demo.rs) - Callback system integration
- [007\_debug\_logging](examples/007_debug_logging.rs) - Debug logging configuration

### Intermediate Examples
- [008\_streaming\_custom\_callbacks](examples/008_streaming_custom_callbacks.rs) - Custom streaming callbacks
- [009\_logging\_demo](examples/009_logging_demo.rs) - Performance logging setup and configuration
- [010\_streaming\_with\_tools](examples/010_streaming_with_tools.rs) - Streaming with tool integration
- [011\_basic\_agent](examples/011_basic_agent.rs) - Basic agent with multiple provider support
- [012\_batching\_optimization\_demo](examples/012_batching_optimization_demo.rs) - Batching optimization patterns
- [013\_mcp\_integration](examples/013_mcp_integration.rs) - Simple MCP server integration
- [014\_mcp\_configuration\_examples](examples/014_mcp_configuration_examples.rs) - MCP configuration examples
- [015\_authorization\_chat\_wrapper](examples/015_authorization_chat_wrapper.rs) - Authorization patterns

### Advanced Examples
- [016\_context\_management](examples/016_context_management.rs) - Context management patterns
- [017\_parallel\_execution](examples/017_parallel_execution.rs) - Parallel execution patterns
- [018\_task\_evaluation](examples/018_task_evaluation.rs) - Task evaluation strategy (default multi-cycle behavior)
- [019\_agent\_based\_evaluation](examples/019_agent_based_evaluation.rs) - Agent-based evaluation example
- [020\_multi\_perspective](examples/020_multi_perspective.rs) - Multi-perspective evaluation example
- [021\_agentic\_chat](examples/021_agentic_chat.rs) - Full interactive chat application

### Expert Examples
- [022\_aws\_doc\_mcp](examples/022_aws_doc_mcp/) - AWS documentation MCP integration
- [023\_telemetry](examples/023_telemetry/) - Comprehensive telemetry examples
- [024\_enterprise\_prompt\_builder](examples/024_enterprise_prompt_builder.rs) - Enterprise prompt building patterns
- [025\_cloudwatch\_observability](examples/025_cloudwatch_observability.rs) - CloudWatch GenAI observability integration
- [026\_nebula\_evaluation\_test](examples/026_nebula_evaluation_test.rs) - Nebula evaluation testing
- [027\_tool\_approval\_middleware](examples/027_tool_approval_middleware.rs) - Tool approval middleware patterns
- [030\_test\_mistral](examples/030_test_mistral.rs) - Mistral model integration testing
- [031\_test\_mistral\_large3](examples/031_test_mistral_large3.rs) - Mistral Large 3 model testing
- [032\_test\_mistral\_tools](examples/032_test_mistral_tools.rs) - Mistral with tool calling

## TODO - Work in Progress

⚠️ **This project is currently in active development and many features are planned but not yet implemented.**

### Provider Support Status

**✅ Implemented Providers:**
- AWS Bedrock (Claude 4.5 Sonnet/Haiku/Opus, Mistral Large 2/3, Nova Lite/Pro/Micro/Premier, Nova 2 Lite/Pro)
- LM Studio (Gemma 3 12B/27B, Llama 3 70B, Mistral 7B, Tessa Rust 7B)

**🚧 Planned Providers (Not Yet Implemented):**
- Anthropic API (direct Claude API access)
- OpenAI (GPT-4, GPT-3.5)
- Ollama (local LLM hosting)
- OpenRouter (multi-provider proxy)
- Candle (local Rust-based inference)

### Major Features Planned

**🎯 High Priority:**
- [ ] **Dynamic Model Registration** - runtime model registration using string-based identification patterns
- [ ] **AWS AgentCore Framework Integration** - Full compatibility with AWS AgentCore for enterprise deployments
- [ ] **Complete Provider Implementation** - Finish all placeholder providers (Anthropic, OpenAI, Ollama, OpenRouter, Candle)
- [ ] **Enhanced MCP Support** - Expand Model Context Protocol integration with more server types - AUTH
- [ ] A2A Protocol Support - Implement standard for Agent to Agent communication
- [ ] **Production Hardening** - Enhanced error handling, retry logic, and resilience patterns


## License

Licensed under Apache License, Version 2.0



