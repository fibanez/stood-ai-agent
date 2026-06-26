//! Test model fixtures.
//!
//! All tests should import model strings from here so that a single change
//! updates every test when a newer model ships.

/// Cheapest current Claude on Bedrock (cross-region inference prefix).
///
/// Update this string when a newer Haiku ships.
pub const TEST_CLAUDE_HAIKU: &str = "us.anthropic.claude-haiku-4-5-20251001-v1:0";

/// Nova Micro — smallest/cheapest Amazon Nova model on Bedrock.
pub const TEST_NOVA_MICRO: &str = "us.amazon.nova-micro-v1:0";

/// Nova Lite — small Amazon Nova model on Bedrock.
pub const TEST_NOVA_LITE: &str = "us.amazon.nova-lite-v1:0";

/// Nova Pro — standard Amazon Nova model on Bedrock.
pub const TEST_NOVA_PRO: &str = "us.amazon.nova-pro-v1:0";

/// Nova Premier — highest-capability Amazon Nova model on Bedrock.
pub const TEST_NOVA_PREMIER: &str = "us.amazon.nova-premier-v1:0";

/// Claude Sonnet 4.5 on Bedrock.
pub const TEST_CLAUDE_SONNET: &str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0";

// ── LM Studio (local) ──────────────────────────────────────────────────────

/// Gemma 3 12B via LM Studio (local).
pub const TEST_LM_GEMMA3_12B: &str = "google/gemma-3-12b";

/// Gemma 3 27B via LM Studio (local).
pub const TEST_LM_GEMMA3_27B: &str = "google/gemma-3-27b";

/// Llama 3 70B via LM Studio (local).
pub const TEST_LM_LLAMA3_70B: &str = "llama-3-70b";

/// Mistral 7B via LM Studio (local).
pub const TEST_LM_MISTRAL_7B: &str = "mistralai/mistral-7b-instruct-v0.3";

/// Tessa Rust 7B via LM Studio (local).
pub const TEST_LM_TESSA_RUST: &str = "tessa-rust-t1-7b";
