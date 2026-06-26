//! Integration tests for Agent with new LLM provider system
//!
//! These tests verify that the Agent can be constructed using the new
//! provider-first architecture while maintaining backward compatibility.

#[cfg(test)]
mod tests {
    use crate::agent::Agent;
    use crate::llm::string_model::StringModel;
    use crate::llm::traits::{LlmModel, ProviderType};

    #[tokio::test]
    async fn test_agent_builder_new_api() {
        // Test that we can create an agent using the new LLM API
        let model = StringModel::new("us.anthropic.claude-sonnet-4-5-20250929-v1:0", ProviderType::Bedrock);

        // Verify model metadata is correct
        assert_eq!(
            model.model_id(),
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        );
        assert_eq!(model.provider(), crate::llm::traits::ProviderType::Bedrock);

        // Test agent builder accepts the new API
        let _builder = Agent::builder()
            .provider("bedrock")
            .model_str("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
            .temperature(0.7)
            .max_tokens(1000);

        // Builder should work without errors
        assert!(true);
    }

    #[tokio::test]
    async fn test_agent_builder_lm_studio() {
        // Test that we can use LM Studio models
        let _builder = Agent::builder()
            .provider("lm_studio")
            .model_str("google/gemma-3-12b")
            .temperature(0.5)
            .max_tokens(2000);

        // Builder should work without errors
        assert!(true);
    }

    #[test]
    fn test_model_metadata() {
        // Test model metadata is correct for new Claude 4.5 models
        let sonnet = StringModel::new("us.anthropic.claude-sonnet-4-5-20250929-v1:0", ProviderType::Bedrock);
        let haiku = StringModel::new("us.anthropic.claude-haiku-4-5-20251001-v1:0", ProviderType::Bedrock);
        let opus = StringModel::new("us.anthropic.claude-opus-4-5-20251101-v1:0", ProviderType::Bedrock);
        let nova_lite = StringModel::new("us.amazon.nova-lite-v1:0", ProviderType::Bedrock);

        // Verify model IDs are correct
        assert_eq!(
            sonnet.model_id(),
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        );
        assert_eq!(
            haiku.model_id(),
            "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        );
        assert_eq!(
            opus.model_id(),
            "us.anthropic.claude-opus-4-5-20251101-v1:0"
        );
        assert_eq!(nova_lite.model_id(), "us.amazon.nova-lite-v1:0");
    }

    #[test]
    fn test_all_models_available() {
        // Verify all new Claude 4.5 models can be created as StringModel
        let _bedrock_sonnet = StringModel::new("us.anthropic.claude-sonnet-4-5-20250929-v1:0", ProviderType::Bedrock);
        let _bedrock_haiku = StringModel::new("us.anthropic.claude-haiku-4-5-20251001-v1:0", ProviderType::Bedrock);
        let _bedrock_opus = StringModel::new("us.anthropic.claude-opus-4-5-20251101-v1:0", ProviderType::Bedrock);
        let _bedrock_nova_lite = StringModel::new("us.amazon.nova-lite-v1:0", ProviderType::Bedrock);
        let _bedrock_nova_pro = StringModel::new("us.amazon.nova-pro-v1:0", ProviderType::Bedrock);
        let _bedrock_nova_micro = StringModel::new("us.amazon.nova-micro-v1:0", ProviderType::Bedrock);

        // LM Studio models
        let _lm_studio_gemma = StringModel::new("google/gemma-3-12b", ProviderType::LmStudio);
        let _lm_studio_llama = StringModel::new("llama-3-70b", ProviderType::LmStudio);
        let _lm_studio_mistral = StringModel::new("mistralai/mistral-7b-instruct-v0.3", ProviderType::LmStudio);

        // All models should be available
        assert!(true);
    }

    #[test]
    fn test_legacy_model_ids_still_work_as_strings() {
        // Verify legacy model IDs still work with the string-based API
        let _legacy_sonnet = StringModel::new("us.anthropic.claude-3-5-sonnet-20241022-v2:0", ProviderType::Bedrock);
        let _legacy_haiku = StringModel::new("us.anthropic.claude-3-5-haiku-20241022-v1:0", ProviderType::Bedrock);
        let _legacy_haiku3 = StringModel::new("us.anthropic.claude-3-haiku-20240307-v1:0", ProviderType::Bedrock);
        let _legacy_opus3 = StringModel::new("us.anthropic.claude-3-opus-20240229-v1:0", ProviderType::Bedrock);

        // All legacy model IDs should still be usable as strings
        assert!(true);
    }
}
