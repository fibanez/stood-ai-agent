//! Integration test for retry mechanism at the LMStudio provider level

use std::time::Duration;
use stood::llm::providers::retry::RetryConfig;

#[tokio::test]
async fn test_retry_config_integration() {
    // Test 1: Default retry configuration
    let default_config = RetryConfig::lm_studio_default();
    assert_eq!(default_config.max_attempts, 3);
    assert_eq!(default_config.initial_delay, Duration::from_millis(1000));
    assert_eq!(default_config.backoff_multiplier, 2.0);
    assert!(default_config.jitter);

    // Test 2: Conservative retry configuration
    let conservative_config = RetryConfig::lm_studio_conservative();
    assert_eq!(conservative_config.max_attempts, 2);
    assert_eq!(
        conservative_config.initial_delay,
        Duration::from_millis(2000)
    );
    assert!(!conservative_config.jitter);

    // Test 3: Aggressive retry configuration
    let aggressive_config = RetryConfig::lm_studio_aggressive();
    assert_eq!(aggressive_config.max_attempts, 5);
    assert_eq!(aggressive_config.initial_delay, Duration::from_millis(500));
    assert!(aggressive_config.jitter);

    // Test 4: Disabled retry configuration
    let disabled_config = RetryConfig::disabled();
    assert_eq!(disabled_config.max_attempts, 0);
}

#[tokio::test]
async fn test_agent_builder_retry_methods() {
    use stood::agent::Agent;

    // Test that agent builder accepts retry configuration
    let agent_builder = Agent::builder()
        .provider("lm_studio")
        .model("tessa-rust-t1-7b")
        .with_retry_config(RetryConfig::lm_studio_aggressive())
        .with_conservative_retry()
        .with_aggressive_retry()
        .without_retry();

    // Just verify the builder compiles and accepts the methods
    // We don't build the agent since LM Studio might not be running
    assert!(true, "Agent builder accepts retry configuration methods");
}

#[test]
fn test_retry_config_serialization() {
    use serde_json;

    // Test that RetryConfig can be serialized and deserialized
    let config = RetryConfig::lm_studio_default();
    let serialized = serde_json::to_string(&config).expect("Should serialize");
    let deserialized: RetryConfig = serde_json::from_str(&serialized).expect("Should deserialize");

    assert_eq!(config.max_attempts, deserialized.max_attempts);
    assert_eq!(config.initial_delay, deserialized.initial_delay);
    assert_eq!(config.backoff_multiplier, deserialized.backoff_multiplier);
    assert_eq!(config.jitter, deserialized.jitter);
}

#[test]
fn test_exponential_backoff_calculation() {
    use stood::llm::providers::retry::calculate_backoff_delay;

    let config = RetryConfig {
        max_attempts: 3,
        initial_delay: Duration::from_millis(100),
        max_delay: Duration::from_secs(10),
        backoff_multiplier: 2.0,
        jitter: false,
    };

    // Test exponential backoff
    assert_eq!(
        calculate_backoff_delay(0, &config),
        Duration::from_millis(100)
    );
    assert_eq!(
        calculate_backoff_delay(1, &config),
        Duration::from_millis(200)
    );
    assert_eq!(
        calculate_backoff_delay(2, &config),
        Duration::from_millis(400)
    );
}

#[test]
fn test_should_retry_llm_error() {
    use stood::llm::providers::retry::{should_retry_llm_error, RetryDecision};
    use stood::llm::traits::{LlmError, ProviderType};

    // Test retryable errors
    let network_error = LlmError::NetworkError {
        message: "Connection refused".to_string(),
        source: None,
    };
    assert_eq!(should_retry_llm_error(&network_error), RetryDecision::Retry);

    let provider_error_503 = LlmError::ProviderError {
        provider: ProviderType::LmStudio,
        message: "Service unavailable 503".to_string(),
        source: None,
    };
    assert_eq!(
        should_retry_llm_error(&provider_error_503),
        RetryDecision::Retry
    );

    let provider_error_502 = LlmError::ProviderError {
        provider: ProviderType::LmStudio,
        message: "Bad gateway 502".to_string(),
        source: None,
    };
    assert_eq!(
        should_retry_llm_error(&provider_error_502),
        RetryDecision::Retry
    );

    // Test non-retryable errors
    let auth_error = LlmError::AuthenticationError {
        provider: ProviderType::LmStudio,
    };
    assert_eq!(
        should_retry_llm_error(&auth_error),
        RetryDecision::FailImmediately
    );

    let config_error = LlmError::ConfigurationError {
        message: "Invalid configuration".to_string(),
    };
    assert_eq!(
        should_retry_llm_error(&config_error),
        RetryDecision::FailImmediately
    );
}
