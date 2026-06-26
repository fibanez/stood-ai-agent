//! Multi-provider LLM integration with unified interface and enterprise-grade reliability.
//!
//! This module provides a comprehensive abstraction layer for interacting with multiple LLM providers
//! including AWS Bedrock, LM Studio, Anthropic, OpenAI, Ollama, OpenRouter, and Candle.
//!
//! # Quick Start
//!
//! Use different providers with the same agent API via runtime strings:
//!
//! ```no_run
//! use stood::agent::Agent;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Use AWS Bedrock (cloud)
//!     let mut bedrock_agent = Agent::builder()
//!         .provider("bedrock")
//!         .model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
//!         .build().await?;
//!
//!     // Same API for all providers
//!     let result = bedrock_agent.execute("Hello!").await?;
//!     println!("{}", result.response);
//!
//!     Ok(())
//! }
//! ```
//!
//! # Key Types
//!
//! - [`LlmProvider`] - Core trait for provider implementations
//! - [`LlmModel`] - Model abstraction trait
//! - [`StringModel`] - Runtime string-based model selection
//! - [`ProviderRegistry`] - Central registry for provider configuration
//! - [`ChatConfig`] - Provider-agnostic chat configuration
//! - [`ChatResponse`] - Unified response format across providers

pub mod client;
pub mod config;
pub mod error;
pub mod models;
pub mod providers;
pub mod registry;
pub mod streaming;
pub mod string_model;
pub mod traits;

#[cfg(test)]
pub mod integration_test;

#[cfg(test)]
pub mod tests;

// Re-export core types for convenience
pub use traits::{
    ChatConfig, ChatResponse, HealthStatus, LlmError, LlmModel, LlmProvider, ModelCapabilities,
    ProviderCapabilities, ProviderType, StreamEvent, Tool, ToolCall, Usage,
};

// Re-export string_model for the runtime string-based API
pub use string_model::StringModel;

// Re-export registry for configuration
pub use registry::{ProviderConfig, ProviderRegistry, PROVIDER_REGISTRY};
