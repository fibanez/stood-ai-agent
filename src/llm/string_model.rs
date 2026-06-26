//! String-based model selection.
//!
//! [`StringModel`] is a lightweight implementation of [`LlmModel`] that holds
//! a provider type and a model-id string supplied by the caller at runtime.
//! It carries no capability metadata — the library is a thin transport that
//! passes the model id directly to the provider's API.

use crate::llm::traits::{LlmModel, ModelCapabilities, ProviderType};

/// A model identified at runtime by a plain string.
///
/// Use this (or the [`crate::agent::AgentBuilder::provider`] /
/// [`crate::agent::AgentBuilder::model`] builder methods) when you want
/// to target a model that is not listed in any compile-time registry.
///
/// # Example
/// ```no_run
/// use stood::agent::Agent;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let agent = Agent::builder()
///     .provider("bedrock")
///     .model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
///     .build()
///     .await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct StringModel {
    model_id: String,
    provider_type: ProviderType,
}

impl StringModel {
    /// Create a new `StringModel` with the given model id and provider.
    pub fn new(model_id: impl Into<String>, provider_type: ProviderType) -> Self {
        Self {
            model_id: model_id.into(),
            provider_type,
        }
    }
}

impl LlmModel for StringModel {
    fn model_id(&self) -> &'static str {
        // SAFETY: We return a reference with a deliberately short lifetime by
        // leaking the String.  This is the minimal change needed to satisfy the
        // `&'static str` contract of the trait while keeping everything
        // Send + Sync.  A proper future refactor should change the trait to
        // return `String` or `Cow<'static, str>`, but that is out of scope for
        // this PR.
        Box::leak(self.model_id.clone().into_boxed_str())
    }

    fn provider(&self) -> ProviderType {
        self.provider_type
    }

    fn context_window(&self) -> usize {
        // Unknown — return a large default so callers never hard-cap.
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
            supports_vision: false,
            context_window: Some(200_000),
        }
    }

    fn display_name(&self) -> &'static str {
        // Re-use the leaked string from model_id().
        self.model_id()
    }

    fn default_temperature(&self) -> f32 {
        0.7
    }

    fn default_max_tokens(&self) -> u32 {
        8_192
    }
}
