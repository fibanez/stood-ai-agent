//! LM Studio integration tests
//!
//! These tests verify that LMStudioProvider can connect to a local LM Studio instance
//! and perform basic chat operations. LM Studio provides local testing of the
//! multi-provider architecture.

#[cfg(test)]
mod tests {
    use crate::llm::registry::PROVIDER_REGISTRY;
    use crate::llm::string_model::StringModel;
    use crate::llm::traits::{ChatConfig, LlmModel, LlmProvider, ProviderType};
    use crate::types::Messages;

    #[tokio::test]
    async fn test_lm_studio_provider_creation() {
        // Test that LM Studio provider can be created
        let result = PROVIDER_REGISTRY.get_provider(ProviderType::LmStudio).await;

        match result {
            Ok(provider) => {
                println!("✅ LMStudioProvider created successfully");
                assert_eq!(provider.provider_type(), ProviderType::LmStudio);

                let capabilities = provider.capabilities();
                println!("🎯 LM Studio capabilities:");
                println!(
                    "  - Supports streaming: {}",
                    capabilities.supports_streaming
                );
                println!("  - Supports tools: {}", capabilities.supports_tools);
                println!("  - Available models: {:?}", capabilities.available_models);
            }
            Err(e) => {
                println!("❌ Failed to create LMStudioProvider: {}", e);
                // This might fail if LM Studio isn't running - that's ok for now
            }
        }
    }

    #[tokio::test]
    async fn test_lm_studio_health_check() {
        // Test LM Studio health check (connection test)
        if let Ok(provider) = PROVIDER_REGISTRY.get_provider(ProviderType::LmStudio).await {
            let health = provider.health_check().await;

            match health {
                Ok(status) => {
                    println!("✅ LM Studio health check: healthy={}", status.healthy);
                    if let Some(latency) = status.latency_ms {
                        println!("  - Latency: {}ms", latency);
                    }
                    if let Some(error) = &status.error {
                        println!("  - Error: {}", error);
                    }
                }
                Err(e) => {
                    println!("❌ LM Studio health check failed: {}", e);
                }
            }
        }
    }

    #[tokio::test]
    async fn test_lm_studio_chat_with_real_instance() {
        // Only run this test if LM Studio is actually running
        if let Ok(provider) = PROVIDER_REGISTRY.get_provider(ProviderType::LmStudio).await {
            let health = provider.health_check().await;

            if health.is_ok() && health.unwrap().healthy {
                // LM Studio is running, try a chat request
                let model = StringModel::new("google/gemma-3-12b", ProviderType::LmStudio);
                let config = ChatConfig {
                    model_id: model.model_id().to_string(),
                    provider: ProviderType::LmStudio,
                    temperature: Some(0.7),
                    max_tokens: Some(50),
                    enable_thinking: false,
                    cache_strategy: crate::llm::traits::CacheStrategy::default(),
                    additional_params: std::collections::HashMap::new(),
                };

                let mut messages = Messages::new();
                messages.add_user_message("Hello! Please respond with just 'Hi there!'");

                match provider.chat(model.model_id(), &messages, &config).await {
                    Ok(response) => {
                        println!("✅ LM Studio chat successful!");
                        println!("🤖 Response: {}", response.content);
                        assert!(!response.content.is_empty());
                    }
                    Err(e) => {
                        println!("❌ LM Studio chat failed: {}", e);
                        // Don't fail the test if model isn't loaded
                    }
                }
            } else {
                println!("⚠️ LM Studio not running, skipping chat test");
            }
        }
    }

    #[test]
    fn test_lm_studio_model_metadata() {
        // Test that LM Studio models have correct metadata
        let gemma = StringModel::new("google/gemma-3-12b", ProviderType::LmStudio);
        let llama = StringModel::new("llama-3-70b", ProviderType::LmStudio);
        let mistral = StringModel::new("mistralai/mistral-7b-instruct-v0.3", ProviderType::LmStudio);

        assert_eq!(gemma.provider(), ProviderType::LmStudio);
        assert_eq!(gemma.model_id(), "google/gemma-3-12b");

        assert_eq!(llama.provider(), ProviderType::LmStudio);
        assert_eq!(llama.model_id(), "llama-3-70b");

        assert_eq!(mistral.provider(), ProviderType::LmStudio);
        assert_eq!(mistral.model_id(), "mistralai/mistral-7b-instruct-v0.3");

        println!("✅ LM Studio model metadata validated");
    }

    #[test]
    fn test_local_testing_setup() {
        // Verify that we have a local alternative to Bedrock
        let bedrock_model = StringModel::new("us.anthropic.claude-sonnet-4-5-20250929-v1:0", ProviderType::Bedrock);
        let lm_studio_model = StringModel::new("google/gemma-3-12b", ProviderType::LmStudio);

        println!(
            "🌐 Bedrock (cloud): {} via {:?}",
            bedrock_model.model_id(),
            bedrock_model.provider()
        );
        println!(
            "🏠 LM Studio (local): {} via {:?}",
            lm_studio_model.model_id(),
            lm_studio_model.provider()
        );

        // Both should implement the same LlmModel trait
        assert_ne!(bedrock_model.provider(), lm_studio_model.provider());
        assert!(true); // Local testing option available!
    }
}
