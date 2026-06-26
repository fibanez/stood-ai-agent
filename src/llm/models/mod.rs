//! Model registry — removed.
//!
//! The compile-time typed model structs (`Bedrock::ClaudeHaiku45`, etc.) have
//! been removed in favour of runtime string-based selection.
//!
//! Use the builder's `.provider()` / `.model()` methods instead:
//!
//! ```no_run
//! use stood::agent::Agent;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let agent = Agent::builder()
//!     .provider("bedrock")
//!     .model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
//!     .build()
//!     .await?;
//! # Ok(())
//! # }
//! ```
