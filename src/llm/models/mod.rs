//! Model definitions organized by provider.
//!
//! This module defines models as pure metadata structs with no logic.
//! Each provider module exports its model structs for use with the single API pattern.

use crate::llm::traits::{LlmModel, ModelCapabilities, ProviderType};

/// AWS Bedrock provider models
///
/// Note: All model IDs use the "us." prefix for cross-region inference capability.
/// This allows models to be accessed from any AWS region, not just us-east-1.
#[allow(non_snake_case)]
pub mod Bedrock {
    use super::*;

    // ============================================================================
    // Claude 4.6 / 4.8 Models (Latest - Recommended)
    // ============================================================================

    /// Claude Sonnet 4.6 via AWS Bedrock - balanced performance model
    ///
    /// The smartest Sonnet model for complex agents and coding tasks.
    #[derive(Debug, Clone, Copy)]
    pub struct ClaudeSonnet46;

    /// Claude Opus 4.8 via AWS Bedrock - premium model
    ///
    /// Combines maximum intelligence with practical performance.
    #[derive(Debug, Clone, Copy)]
    pub struct ClaudeOpus48;

    // ============================================================================
    // Claude 4.5 Models (Current)
    // ============================================================================

    /// Claude Sonnet 4.5 via AWS Bedrock - balanced performance model
    ///
    /// The smartest model for complex agents and coding tasks.
    /// Released: September 29, 2025
    #[derive(Debug, Clone, Copy)]
    pub struct ClaudeSonnet45;

    /// Claude Haiku 4.5 via AWS Bedrock - fastest model
    ///
    /// Fastest model with near-frontier intelligence.
    /// Released: October 1, 2025
    #[derive(Debug, Clone, Copy)]
    pub struct ClaudeHaiku45;

    /// Claude Opus 4.5 via AWS Bedrock - premium model
    ///
    /// Combines maximum intelligence with practical performance.
    /// Released: November 1, 2025
    #[derive(Debug, Clone, Copy)]
    pub struct ClaudeOpus45;

    // ============================================================================
    // Model Aliases - Use these for automatic upgrades to latest versions
    // ============================================================================

    /// Alias for the latest Claude Sonnet model (currently 4.6)
    ///
    /// Using this alias ensures your application automatically uses the latest
    /// Sonnet model when upgrading the Stood library.
    pub type SonnetLatest = ClaudeSonnet46;

    /// Alias for the latest Claude Haiku model (currently 4.5)
    ///
    /// Using this alias ensures your application automatically uses the latest
    /// Haiku model when upgrading the Stood library.
    pub type HaikuLatest = ClaudeHaiku45;

    /// Alias for the latest Claude Opus model (currently 4.8)
    ///
    /// Using this alias ensures your application automatically uses the latest
    /// Opus model when upgrading the Stood library.
    pub type OpusLatest = ClaudeOpus48;

    // ============================================================================
    // Legacy Claude Models (Deprecated)
    // ============================================================================

    /// Claude 3.5 Sonnet v2 via AWS Bedrock
    #[deprecated(
        since = "0.2.0",
        note = "Use `ClaudeSonnet45` instead. Claude 3.5 Sonnet will be removed in a future release."
    )]
    #[derive(Debug, Clone, Copy)]
    pub struct Claude35Sonnet;

    /// Claude 3.5 Haiku via AWS Bedrock
    #[deprecated(
        since = "0.2.0",
        note = "Use `ClaudeHaiku45` instead. Claude 3.5 Haiku will be removed in a future release."
    )]
    #[derive(Debug, Clone, Copy)]
    pub struct Claude35Haiku;

    /// Claude 3 Haiku via AWS Bedrock
    #[deprecated(
        since = "0.2.0",
        note = "Use `ClaudeHaiku45` instead. Claude 3 Haiku will be removed in a future release."
    )]
    #[derive(Debug, Clone, Copy)]
    pub struct ClaudeHaiku3;

    /// Claude 3 Opus via AWS Bedrock
    #[deprecated(
        since = "0.2.0",
        note = "Use `ClaudeOpus45` instead. Claude 3 Opus will be removed in a future release."
    )]
    #[derive(Debug, Clone, Copy)]
    pub struct ClaudeOpus3;

    // ============================================================================
    // Mistral Models
    // ============================================================================

    /// Mistral Large 2 via AWS Bedrock - flagship reasoning model
    ///
    /// High-capability model with strong reasoning, coding, and multilingual support.
    /// 128K context window with advanced function calling.
    /// Released: July 2024
    #[derive(Debug, Clone, Copy)]
    pub struct MistralLarge2;

    /// Mistral Large 3 via AWS Bedrock - latest flagship model
    ///
    /// Latest Mistral flagship with enhanced capabilities.
    /// 128K context window with advanced function calling.
    /// Released: December 2025
    #[derive(Debug, Clone, Copy)]
    pub struct MistralLarge3;

    /// Alias for the latest Mistral Large model (currently Large 3)
    ///
    /// Using this alias ensures your application automatically uses the latest
    /// Mistral Large model when upgrading the Stood library.
    pub type MistralLatest = MistralLarge3;

    // ============================================================================
    // Amazon Nova Models
    // ============================================================================

    /// Amazon Nova Lite via AWS Bedrock
    #[derive(Debug, Clone, Copy)]
    pub struct NovaLite;

    /// Amazon Nova Pro via AWS Bedrock
    #[derive(Debug, Clone, Copy)]
    pub struct NovaPro;

    /// Amazon Nova Micro via AWS Bedrock
    #[derive(Debug, Clone, Copy)]
    pub struct NovaMicro;

    /// Amazon Nova Premier via AWS Bedrock - highest capability Nova model
    ///
    /// Best for complex agentic workflows, multimodal tasks, and model distillation.
    /// 300K context window, supports vision and video.
    #[derive(Debug, Clone, Copy)]
    pub struct NovaPremier;

    /// Amazon Nova 2 Lite via AWS Bedrock - fast reasoning model
    ///
    /// Cost-effective model with extended thinking support.
    /// 1M context window.
    #[derive(Debug, Clone, Copy)]
    pub struct Nova2Lite;

    /// Amazon Nova 2 Pro via AWS Bedrock - intelligent reasoning model
    ///
    /// Most capable Nova model for complex multistep tasks.
    /// 1M context window with extended thinking.
    #[derive(Debug, Clone, Copy)]
    pub struct Nova2Pro;

    // Implement LlmModel trait for all Bedrock models

    // ============================================================================
    // Claude 4.6 / 4.8 Model Implementations
    // ============================================================================

    impl LlmModel for ClaudeSonnet46 {
        fn model_id(&self) -> &'static str {
            // Note: us. prefix required for cross-region inference in AWS Bedrock
            "us.anthropic.claude-sonnet-4-6"
        }
        fn provider(&self) -> ProviderType {
            ProviderType::Bedrock
        }
        fn context_window(&self) -> usize {
            200_000
        }
        fn max_output_tokens(&self) -> usize {
            8_192
        }
        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                max_tokens: Some(8_192),
                supports_tools: true,
                supports_streaming: true,
                supports_thinking: true,
                supports_vision: true,
                context_window: Some(200_000),
            }
        }
        fn display_name(&self) -> &'static str {
            "Claude Sonnet 4.6"
        }
        fn default_temperature(&self) -> f32 {
            0.7
        }
        fn default_max_tokens(&self) -> u32 {
            8_192
        }
    }

    impl LlmModel for ClaudeOpus48 {
        fn model_id(&self) -> &'static str {
            // Note: us. prefix required for cross-region inference in AWS Bedrock
            "us.anthropic.claude-opus-4-8"
        }
        fn provider(&self) -> ProviderType {
            ProviderType::Bedrock
        }
        fn context_window(&self) -> usize {
            200_000
        }
        fn max_output_tokens(&self) -> usize {
            8_192
        }
        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                max_tokens: Some(8_192),
                supports_tools: true,
                supports_streaming: true,
                supports_thinking: true,
                supports_vision: true,
                context_window: Some(200_000),
            }
        }
        fn display_name(&self) -> &'static str {
            "Claude Opus 4.8"
        }
        fn default_temperature(&self) -> f32 {
            0.6
        }
        fn default_max_tokens(&self) -> u32 {
            8_192
        }
    }

    // ============================================================================
    // Claude 4.5 Model Implementations
    // ============================================================================

    impl LlmModel for ClaudeSonnet45 {
        fn model_id(&self) -> &'static str {
            // Note: us. prefix required for cross-region inference in AWS Bedrock
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        }
        fn provider(&self) -> ProviderType {
            ProviderType::Bedrock
        }
        fn context_window(&self) -> usize {
            200_000
        }
        fn max_output_tokens(&self) -> usize {
            8_192
        }
        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                max_tokens: Some(8_192),
                supports_tools: true,
                supports_streaming: true,
                supports_thinking: true,
                supports_vision: true,
                context_window: Some(200_000),
            }
        }
        fn display_name(&self) -> &'static str {
            "Claude Sonnet 4.5"
        }
        fn default_temperature(&self) -> f32 {
            0.7
        }
        fn default_max_tokens(&self) -> u32 {
            8_192
        }
    }

    impl LlmModel for ClaudeHaiku45 {
        fn model_id(&self) -> &'static str {
            // Note: us. prefix required for cross-region inference in AWS Bedrock
            "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        }
        fn provider(&self) -> ProviderType {
            ProviderType::Bedrock
        }
        fn context_window(&self) -> usize {
            200_000
        }
        fn max_output_tokens(&self) -> usize {
            8_192
        }
        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                max_tokens: Some(8_192),
                supports_tools: true,
                supports_streaming: true,
                supports_thinking: false,
                supports_vision: true,
                context_window: Some(200_000),
            }
        }
        fn display_name(&self) -> &'static str {
            "Claude Haiku 4.5"
        }
        fn default_temperature(&self) -> f32 {
            0.8
        }
        fn default_max_tokens(&self) -> u32 {
            8_192
        }
    }

    impl LlmModel for ClaudeOpus45 {
        fn model_id(&self) -> &'static str {
            // Note: us. prefix required for cross-region inference in AWS Bedrock
            "us.anthropic.claude-opus-4-5-20251101-v1:0"
        }
        fn provider(&self) -> ProviderType {
            ProviderType::Bedrock
        }
        fn context_window(&self) -> usize {
            200_000
        }
        fn max_output_tokens(&self) -> usize {
            8_192
        }
        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                max_tokens: Some(8_192),
                supports_tools: true,
                supports_streaming: true,
                supports_thinking: true,
                supports_vision: true,
                context_window: Some(200_000),
            }
        }
        fn display_name(&self) -> &'static str {
            "Claude Opus 4.5"
        }
        fn default_temperature(&self) -> f32 {
            0.6
        }
        fn default_max_tokens(&self) -> u32 {
            8_192
        }
    }

    // ============================================================================
    // Legacy Claude Model Implementations (Deprecated)
    // ============================================================================

    #[allow(deprecated)]
    impl LlmModel for Claude35Sonnet {
        fn model_id(&self) -> &'static str {
            // Note: us. prefix required for cross-region inference in AWS Bedrock
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        }
        fn provider(&self) -> ProviderType {
            ProviderType::Bedrock
        }
        fn context_window(&self) -> usize {
            200_000
        }
        fn max_output_tokens(&self) -> usize {
            8_192
        }
        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                max_tokens: Some(8_192),
                supports_tools: true,
                supports_streaming: true,
                supports_thinking: true,
                supports_vision: true,
                context_window: Some(200_000),
            }
        }
        fn display_name(&self) -> &'static str {
            "Claude 3.5 Sonnet"
        }
        fn default_temperature(&self) -> f32 {
            0.7
        }
        fn default_max_tokens(&self) -> u32 {
            8_192
        }
    }

    #[allow(deprecated)]
    impl LlmModel for Claude35Haiku {
        fn model_id(&self) -> &'static str {
            // Note: us. prefix required for cross-region inference in AWS Bedrock
            "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        }
        fn provider(&self) -> ProviderType {
            ProviderType::Bedrock
        }
        fn context_window(&self) -> usize {
            200_000
        }
        fn max_output_tokens(&self) -> usize {
            8_192
        }
        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                max_tokens: Some(8_192),
                supports_tools: true,
                supports_streaming: true,
                supports_thinking: false,
                supports_vision: true,
                context_window: Some(200_000),
            }
        }
        fn display_name(&self) -> &'static str {
            "Claude 3.5 Haiku"
        }
        fn default_temperature(&self) -> f32 {
            0.8
        }
        fn default_max_tokens(&self) -> u32 {
            8_192
        }
    }

    #[allow(deprecated)]
    impl LlmModel for ClaudeHaiku3 {
        fn model_id(&self) -> &'static str {
            // Note: us. prefix required for cross-region inference in AWS Bedrock
            "us.anthropic.claude-3-haiku-20240307-v1:0"
        }
        fn provider(&self) -> ProviderType {
            ProviderType::Bedrock
        }
        fn context_window(&self) -> usize {
            200_000
        }
        fn max_output_tokens(&self) -> usize {
            4_096
        }
        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                max_tokens: Some(4_096),
                supports_tools: true,
                supports_streaming: true,
                supports_thinking: false,
                supports_vision: true,
                context_window: Some(200_000),
            }
        }
        fn display_name(&self) -> &'static str {
            "Claude 3 Haiku"
        }
        fn default_temperature(&self) -> f32 {
            0.8
        }
        fn default_max_tokens(&self) -> u32 {
            4_096
        }
    }

    #[allow(deprecated)]
    impl LlmModel for ClaudeOpus3 {
        fn model_id(&self) -> &'static str {
            // Note: us. prefix required for cross-region inference in AWS Bedrock
            "us.anthropic.claude-3-opus-20240229-v1:0"
        }
        fn provider(&self) -> ProviderType {
            ProviderType::Bedrock
        }
        fn context_window(&self) -> usize {
            200_000
        }
        fn max_output_tokens(&self) -> usize {
            4_096
        }
        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                max_tokens: Some(4_096),
                supports_tools: true,
                supports_streaming: true,
                supports_thinking: false,
                supports_vision: true,
                context_window: Some(200_000),
            }
        }
        fn display_name(&self) -> &'static str {
            "Claude 3 Opus"
        }
        fn default_temperature(&self) -> f32 {
            0.6
        }
        fn default_max_tokens(&self) -> u32 {
            4_096
        }
    }

    // ============================================================================
    // Mistral Model Implementations
    // ============================================================================

    impl LlmModel for MistralLarge2 {
        fn model_id(&self) -> &'static str {
            "mistral.mistral-large-2407-v1:0"
        }
        fn provider(&self) -> ProviderType {
            ProviderType::Bedrock
        }
        fn context_window(&self) -> usize {
            128_000
        }
        fn max_output_tokens(&self) -> usize {
            8_192
        }
        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                max_tokens: Some(8_192),
                supports_tools: true,
                supports_streaming: true,
                supports_thinking: false,
                supports_vision: false,
                context_window: Some(128_000),
            }
        }
        fn display_name(&self) -> &'static str {
            "Mistral Large 2"
        }
        fn default_temperature(&self) -> f32 {
            0.7
        }
        fn default_max_tokens(&self) -> u32 {
            8_192
        }
    }

    impl LlmModel for MistralLarge3 {
        fn model_id(&self) -> &'static str {
            "mistral.mistral-large-3-675b-instruct"
        }
        fn provider(&self) -> ProviderType {
            ProviderType::Bedrock
        }
        fn context_window(&self) -> usize {
            128_000
        }
        fn max_output_tokens(&self) -> usize {
            8_192
        }
        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                max_tokens: Some(8_192),
                supports_tools: true,
                supports_streaming: true,
                supports_thinking: false,
                supports_vision: false,
                context_window: Some(128_000),
            }
        }
        fn display_name(&self) -> &'static str {
            "Mistral Large 3"
        }
        fn default_temperature(&self) -> f32 {
            0.7
        }
        fn default_max_tokens(&self) -> u32 {
            8_192
        }
    }

    // ============================================================================
    // Nova Model Implementations
    // ============================================================================

    impl LlmModel for NovaLite {
        fn model_id(&self) -> &'static str {
            // Note: us. prefix required for cross-region inference in AWS Bedrock
            "us.amazon.nova-lite-v1:0"
        }
        fn provider(&self) -> ProviderType {
            ProviderType::Bedrock
        }
        fn context_window(&self) -> usize {
            300_000
        }
        fn max_output_tokens(&self) -> usize {
            5_000
        }
        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                max_tokens: Some(5_000),
                supports_tools: true,
                supports_streaming: true,
                supports_thinking: false,
                supports_vision: false,
                context_window: Some(300_000),
            }
        }
        fn display_name(&self) -> &'static str {
            "Amazon Nova Lite"
        }
        fn default_temperature(&self) -> f32 {
            0.7
        }
        fn default_max_tokens(&self) -> u32 {
            5_000
        }
    }

    impl LlmModel for NovaPro {
        fn model_id(&self) -> &'static str {
            // Note: us. prefix required for cross-region inference in AWS Bedrock
            "us.amazon.nova-pro-v1:0"
        }
        fn provider(&self) -> ProviderType {
            ProviderType::Bedrock
        }
        fn context_window(&self) -> usize {
            300_000
        }
        fn max_output_tokens(&self) -> usize {
            5_000
        }
        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                max_tokens: Some(5_000),
                supports_tools: true,
                supports_streaming: true,
                supports_thinking: false,
                supports_vision: true,
                context_window: Some(300_000),
            }
        }
        fn display_name(&self) -> &'static str {
            "Amazon Nova Pro"
        }
        fn default_temperature(&self) -> f32 {
            0.7
        }
        fn default_max_tokens(&self) -> u32 {
            5_000
        }
    }

    impl LlmModel for NovaMicro {
        fn model_id(&self) -> &'static str {
            // Note: us. prefix required for cross-region inference in AWS Bedrock
            "us.amazon.nova-micro-v1:0"
        }
        fn provider(&self) -> ProviderType {
            ProviderType::Bedrock
        }
        fn context_window(&self) -> usize {
            128_000
        }
        fn max_output_tokens(&self) -> usize {
            2_048
        }
        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                max_tokens: Some(2_048),
                supports_tools: true,
                supports_streaming: true,
                supports_thinking: false,
                supports_vision: false,
                context_window: Some(128_000),
            }
        }
        fn display_name(&self) -> &'static str {
            "Amazon Nova Micro"
        }
        fn default_temperature(&self) -> f32 {
            0.7
        }
        fn default_max_tokens(&self) -> u32 {
            2_048
        }
    }

    impl LlmModel for NovaPremier {
        fn model_id(&self) -> &'static str {
            // Note: us. prefix required for cross-region inference in AWS Bedrock
            "us.amazon.nova-premier-v1:0"
        }
        fn provider(&self) -> ProviderType {
            ProviderType::Bedrock
        }
        fn context_window(&self) -> usize {
            300_000
        }
        fn max_output_tokens(&self) -> usize {
            5_000
        }
        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                max_tokens: Some(5_000),
                supports_tools: true,
                supports_streaming: true,
                supports_thinking: false,
                supports_vision: true,
                context_window: Some(300_000),
            }
        }
        fn display_name(&self) -> &'static str {
            "Amazon Nova Premier"
        }
        fn default_temperature(&self) -> f32 {
            0.7
        }
        fn default_max_tokens(&self) -> u32 {
            5_000
        }
    }

    impl LlmModel for Nova2Lite {
        fn model_id(&self) -> &'static str {
            // Note: us. prefix required for cross-region inference in AWS Bedrock
            "us.amazon.nova-2-lite-v1:0"
        }
        fn provider(&self) -> ProviderType {
            ProviderType::Bedrock
        }
        fn context_window(&self) -> usize {
            1_000_000
        }
        fn max_output_tokens(&self) -> usize {
            5_000
        }
        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                max_tokens: Some(5_000),
                supports_tools: true,
                supports_streaming: true,
                supports_thinking: true,
                supports_vision: false,
                context_window: Some(1_000_000),
            }
        }
        fn display_name(&self) -> &'static str {
            "Amazon Nova 2 Lite"
        }
        fn default_temperature(&self) -> f32 {
            0.7
        }
        fn default_max_tokens(&self) -> u32 {
            5_000
        }
    }

    impl LlmModel for Nova2Pro {
        fn model_id(&self) -> &'static str {
            // Note: us. prefix required for cross-region inference in AWS Bedrock
            "us.amazon.nova-2-pro-v1:0"
        }
        fn provider(&self) -> ProviderType {
            ProviderType::Bedrock
        }
        fn context_window(&self) -> usize {
            1_000_000
        }
        fn max_output_tokens(&self) -> usize {
            5_000
        }
        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                max_tokens: Some(5_000),
                supports_tools: true,
                supports_streaming: true,
                supports_thinking: true,
                supports_vision: false,
                context_window: Some(1_000_000),
            }
        }
        fn display_name(&self) -> &'static str {
            "Amazon Nova 2 Pro"
        }
        fn default_temperature(&self) -> f32 {
            0.7
        }
        fn default_max_tokens(&self) -> u32 {
            5_000
        }
    }
}

/// LM Studio provider models
#[allow(non_snake_case)]
pub mod LMStudio {
    use super::*;

    /// Gemma 3 12B model via LM Studio
    #[derive(Debug, Clone, Copy)]
    pub struct Gemma3_12B;

    /// Llama 3 70B model via LM Studio
    #[derive(Debug, Clone, Copy)]
    pub struct Llama3_70B;

    /// Gemma 3 27B model via LM Studio
    #[derive(Debug, Clone, Copy)]
    pub struct Gemma3_27B;

    /// Mistral 7B model via LM Studio
    #[derive(Debug, Clone, Copy)]
    pub struct Mistral7B;

    /// Tessa Rust 7B model via LM Studio - specialized for Rust code analysis
    #[derive(Debug, Clone, Copy)]
    pub struct TessaRust7B;

    impl LlmModel for Gemma3_12B {
        fn model_id(&self) -> &'static str {
            "google/gemma-3-12b"
        }
        fn provider(&self) -> ProviderType {
            ProviderType::LmStudio
        }
        fn context_window(&self) -> usize {
            8_192
        }
        fn max_output_tokens(&self) -> usize {
            2_048
        }
        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                max_tokens: Some(2_048),
                supports_tools: true, // Depends on LM Studio setup
                supports_streaming: true,
                supports_thinking: false,
                supports_vision: false,
                context_window: Some(8_192),
            }
        }
        fn display_name(&self) -> &'static str {
            "Gemma 3 12B (Local)"
        }
        fn default_temperature(&self) -> f32 {
            0.7
        }
        fn default_max_tokens(&self) -> u32 {
            2_048
        }
    }

    impl LlmModel for Gemma3_27B {
        fn model_id(&self) -> &'static str {
            "google/gemma-3-27b"
        }
        fn provider(&self) -> ProviderType {
            ProviderType::LmStudio
        }
        fn context_window(&self) -> usize {
            8_192
        }
        fn max_output_tokens(&self) -> usize {
            4_096
        }
        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                max_tokens: Some(4_096),
                supports_tools: true, // Larger model should support tools better
                supports_streaming: true,
                supports_thinking: false,
                supports_vision: false,
                context_window: Some(8_192),
            }
        }
        fn display_name(&self) -> &'static str {
            "Gemma 3 27B (Local)"
        }
        fn default_temperature(&self) -> f32 {
            0.7
        }
        fn default_max_tokens(&self) -> u32 {
            4_096
        }
    }

    impl LlmModel for Llama3_70B {
        fn model_id(&self) -> &'static str {
            "llama-3-70b"
        }
        fn provider(&self) -> ProviderType {
            ProviderType::LmStudio
        }
        fn context_window(&self) -> usize {
            32_768
        }
        fn max_output_tokens(&self) -> usize {
            4_096
        }
        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                max_tokens: Some(4_096),
                supports_tools: false, // Depends on LM Studio setup
                supports_streaming: true,
                supports_thinking: false,
                supports_vision: false,
                context_window: Some(32_768),
            }
        }
        fn display_name(&self) -> &'static str {
            "Llama 3 70B (Local)"
        }
        fn default_temperature(&self) -> f32 {
            0.7
        }
        fn default_max_tokens(&self) -> u32 {
            4_096
        }
    }

    impl LlmModel for Mistral7B {
        fn model_id(&self) -> &'static str {
            "mistralai/mistral-7b-instruct-v0.3"
        }
        fn provider(&self) -> ProviderType {
            ProviderType::LmStudio
        }
        fn context_window(&self) -> usize {
            32_768
        }
        fn max_output_tokens(&self) -> usize {
            2_048
        }
        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                max_tokens: Some(2_048),
                supports_tools: false,
                supports_streaming: true,
                supports_thinking: false,
                supports_vision: false,
                context_window: Some(32_768),
            }
        }
        fn display_name(&self) -> &'static str {
            "Mistral 7B (Local)"
        }
        fn default_temperature(&self) -> f32 {
            0.7
        }
        fn default_max_tokens(&self) -> u32 {
            2_048
        }
    }

    impl LlmModel for TessaRust7B {
        fn model_id(&self) -> &'static str {
            "tessa-rust-t1-7b"
        }
        fn provider(&self) -> ProviderType {
            ProviderType::LmStudio
        }
        fn context_window(&self) -> usize {
            8_192
        }
        fn max_output_tokens(&self) -> usize {
            2_048
        }
        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                max_tokens: Some(2_048),
                supports_tools: true,
                supports_streaming: true,
                supports_thinking: false,
                supports_vision: false,
                context_window: Some(8_192),
            }
        }
        fn display_name(&self) -> &'static str {
            "Tessa Rust 7B (Local)"
        }
        fn default_temperature(&self) -> f32 {
            0.3 // Lower temperature for more precise code analysis
        }
        fn default_max_tokens(&self) -> u32 {
            2_048
        }
    }
}

/// Anthropic Direct API provider models
#[allow(non_snake_case)]
pub mod Anthropic {
    use super::*;

    // ============================================================================
    // Claude 4.6 / 4.8 Models (Latest - Recommended)
    // ============================================================================

    /// Claude Sonnet 4.6 via Anthropic Direct API - balanced performance model
    ///
    /// The smartest Sonnet model for complex agents and coding tasks.
    #[derive(Debug, Clone, Copy)]
    pub struct ClaudeSonnet46;

    /// Claude Opus 4.8 via Anthropic Direct API - premium model
    ///
    /// Combines maximum intelligence with practical performance.
    #[derive(Debug, Clone, Copy)]
    pub struct ClaudeOpus48;

    // ============================================================================
    // Claude 4.5 Models (Current)
    // ============================================================================

    /// Claude Sonnet 4.5 via Anthropic Direct API - balanced performance model
    ///
    /// The smartest model for complex agents and coding tasks.
    /// Released: September 29, 2025
    #[derive(Debug, Clone, Copy)]
    pub struct ClaudeSonnet45;

    /// Claude Haiku 4.5 via Anthropic Direct API - fastest model
    ///
    /// Fastest model with near-frontier intelligence.
    /// Released: October 1, 2025
    #[derive(Debug, Clone, Copy)]
    pub struct ClaudeHaiku45;

    /// Claude Opus 4.5 via Anthropic Direct API - premium model
    ///
    /// Combines maximum intelligence with practical performance.
    /// Released: November 1, 2025
    #[derive(Debug, Clone, Copy)]
    pub struct ClaudeOpus45;

    // ============================================================================
    // Legacy Claude Models (Deprecated)
    // ============================================================================

    /// Claude 3.5 Sonnet via Anthropic Direct API
    #[deprecated(
        since = "0.2.0",
        note = "Use `ClaudeSonnet45` instead. Claude 3.5 Sonnet will be removed in a future release."
    )]
    #[derive(Debug, Clone, Copy)]
    pub struct Claude35Sonnet;

    /// Claude 3.5 Haiku via Anthropic Direct API
    #[deprecated(
        since = "0.2.0",
        note = "Use `ClaudeHaiku45` instead. Claude 3.5 Haiku will be removed in a future release."
    )]
    #[derive(Debug, Clone, Copy)]
    pub struct Claude35Haiku;

    /// Claude 3 Opus via Anthropic Direct API
    #[deprecated(
        since = "0.2.0",
        note = "Use `ClaudeOpus45` instead. Claude 3 Opus will be removed in a future release."
    )]
    #[derive(Debug, Clone, Copy)]
    pub struct Claude3Opus;

    // ============================================================================
    // Claude 4.6 / 4.8 Model Implementations
    // ============================================================================

    impl LlmModel for ClaudeSonnet46 {
        fn model_id(&self) -> &'static str {
            "claude-sonnet-4-6"
        }
        fn provider(&self) -> ProviderType {
            ProviderType::Anthropic
        }
        fn context_window(&self) -> usize {
            200_000
        }
        fn max_output_tokens(&self) -> usize {
            8_192
        }
        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                max_tokens: Some(8_192),
                supports_tools: true,
                supports_streaming: true,
                supports_thinking: true,
                supports_vision: true,
                context_window: Some(200_000),
            }
        }
        fn display_name(&self) -> &'static str {
            "Claude Sonnet 4.6 (Direct)"
        }
        fn default_temperature(&self) -> f32 {
            0.7
        }
        fn default_max_tokens(&self) -> u32 {
            8_192
        }
    }

    impl LlmModel for ClaudeOpus48 {
        fn model_id(&self) -> &'static str {
            "claude-opus-4-8"
        }
        fn provider(&self) -> ProviderType {
            ProviderType::Anthropic
        }
        fn context_window(&self) -> usize {
            200_000
        }
        fn max_output_tokens(&self) -> usize {
            8_192
        }
        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                max_tokens: Some(8_192),
                supports_tools: true,
                supports_streaming: true,
                supports_thinking: true,
                supports_vision: true,
                context_window: Some(200_000),
            }
        }
        fn display_name(&self) -> &'static str {
            "Claude Opus 4.8 (Direct)"
        }
        fn default_temperature(&self) -> f32 {
            0.6
        }
        fn default_max_tokens(&self) -> u32 {
            8_192
        }
    }

    // ============================================================================
    // Claude 4.5 Model Implementations
    // ============================================================================

    impl LlmModel for ClaudeSonnet45 {
        fn model_id(&self) -> &'static str {
            "claude-sonnet-4-5-20250929"
        }
        fn provider(&self) -> ProviderType {
            ProviderType::Anthropic
        }
        fn context_window(&self) -> usize {
            200_000
        }
        fn max_output_tokens(&self) -> usize {
            8_192
        }
        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                max_tokens: Some(8_192),
                supports_tools: true,
                supports_streaming: true,
                supports_thinking: true,
                supports_vision: true,
                context_window: Some(200_000),
            }
        }
        fn display_name(&self) -> &'static str {
            "Claude Sonnet 4.5 (Direct)"
        }
        fn default_temperature(&self) -> f32 {
            0.7
        }
        fn default_max_tokens(&self) -> u32 {
            8_192
        }
    }

    impl LlmModel for ClaudeHaiku45 {
        fn model_id(&self) -> &'static str {
            "claude-haiku-4-5-20251001"
        }
        fn provider(&self) -> ProviderType {
            ProviderType::Anthropic
        }
        fn context_window(&self) -> usize {
            200_000
        }
        fn max_output_tokens(&self) -> usize {
            8_192
        }
        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                max_tokens: Some(8_192),
                supports_tools: true,
                supports_streaming: true,
                supports_thinking: false,
                supports_vision: true,
                context_window: Some(200_000),
            }
        }
        fn display_name(&self) -> &'static str {
            "Claude Haiku 4.5 (Direct)"
        }
        fn default_temperature(&self) -> f32 {
            0.8
        }
        fn default_max_tokens(&self) -> u32 {
            8_192
        }
    }

    impl LlmModel for ClaudeOpus45 {
        fn model_id(&self) -> &'static str {
            "claude-opus-4-5-20251101"
        }
        fn provider(&self) -> ProviderType {
            ProviderType::Anthropic
        }
        fn context_window(&self) -> usize {
            200_000
        }
        fn max_output_tokens(&self) -> usize {
            8_192
        }
        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                max_tokens: Some(8_192),
                supports_tools: true,
                supports_streaming: true,
                supports_thinking: true,
                supports_vision: true,
                context_window: Some(200_000),
            }
        }
        fn display_name(&self) -> &'static str {
            "Claude Opus 4.5 (Direct)"
        }
        fn default_temperature(&self) -> f32 {
            0.6
        }
        fn default_max_tokens(&self) -> u32 {
            8_192
        }
    }

    // ============================================================================
    // Legacy Claude Model Implementations (Deprecated)
    // ============================================================================

    #[allow(deprecated)]
    impl LlmModel for Claude35Sonnet {
        fn model_id(&self) -> &'static str {
            "claude-3-5-sonnet-20241022"
        }
        fn provider(&self) -> ProviderType {
            ProviderType::Anthropic
        }
        fn context_window(&self) -> usize {
            200_000
        }
        fn max_output_tokens(&self) -> usize {
            8_192
        }
        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                max_tokens: Some(8_192),
                supports_tools: true,
                supports_streaming: false, // TODO: Implement streaming
                supports_thinking: false,  // TODO: Add thinking mode
                supports_vision: false,    // TODO: Add vision support
                context_window: Some(200_000),
            }
        }
        fn display_name(&self) -> &'static str {
            "Claude 3.5 Sonnet (Direct)"
        }
        fn default_temperature(&self) -> f32 {
            0.7
        }
        fn default_max_tokens(&self) -> u32 {
            8_192
        }
    }

    #[allow(deprecated)]
    impl LlmModel for Claude35Haiku {
        fn model_id(&self) -> &'static str {
            "claude-3-5-haiku-20241022"
        }
        fn provider(&self) -> ProviderType {
            ProviderType::Anthropic
        }
        fn context_window(&self) -> usize {
            200_000
        }
        fn max_output_tokens(&self) -> usize {
            8_192
        }
        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                max_tokens: Some(8_192),
                supports_tools: true,
                supports_streaming: false, // TODO: Implement streaming
                supports_thinking: false,  // TODO: Add thinking mode
                supports_vision: false,    // TODO: Add vision support
                context_window: Some(200_000),
            }
        }
        fn display_name(&self) -> &'static str {
            "Claude 3.5 Haiku (Direct)"
        }
        fn default_temperature(&self) -> f32 {
            0.8
        }
        fn default_max_tokens(&self) -> u32 {
            8_192
        }
    }

    #[allow(deprecated)]
    impl LlmModel for Claude3Opus {
        fn model_id(&self) -> &'static str {
            "claude-3-opus-20240229"
        }
        fn provider(&self) -> ProviderType {
            ProviderType::Anthropic
        }
        fn context_window(&self) -> usize {
            200_000
        }
        fn max_output_tokens(&self) -> usize {
            4_096
        }
        fn capabilities(&self) -> ModelCapabilities {
            ModelCapabilities {
                max_tokens: Some(4_096),
                supports_tools: true,
                supports_streaming: false, // TODO: Implement streaming
                supports_thinking: false,  // TODO: Add thinking mode
                supports_vision: false,    // TODO: Add vision support
                context_window: Some(200_000),
            }
        }
        fn display_name(&self) -> &'static str {
            "Claude 3 Opus (Direct)"
        }
        fn default_temperature(&self) -> f32 {
            0.6
        }
        fn default_max_tokens(&self) -> u32 {
            4_096
        }
    }
}

// Provider modules are available as: use stood::llm::models::Bedrock::ClaudeHaiku45;
// or via the top-level re-export: use stood::llm::{Bedrock, LMStudio, Anthropic};
//
// Recommended models (latest):
//   - Bedrock::ClaudeHaiku45  - Fastest, cost-effective
//   - Bedrock::ClaudeSonnet46 - Balanced performance
//   - Bedrock::ClaudeOpus48   - Maximum intelligence
//
// Or use the auto-upgrading aliases:
//   - Bedrock::HaikuLatest  (currently ClaudeHaiku45)
//   - Bedrock::SonnetLatest (currently ClaudeSonnet46)
//   - Bedrock::OpusLatest   (currently ClaudeOpus48)
