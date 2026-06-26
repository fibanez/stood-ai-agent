//! Conversation management for agent interactions
//!
//! This module provides comprehensive conversation history management with intelligent
//! context window optimization. You'll get automatic token management, message limits,
//! and seamless Bedrock API formatting for agent conversations.
//!
//! # Key Features
//!
//! - **Context Window Management** - Automatic pruning based on token and message limits
//! - **Bedrock Integration** - Native formatting for AWS Bedrock Claude models
//! - **System Prompt Support** - Persistent system prompts across conversation turns
//! - **Token Estimation** - Built-in token counting for context planning
//! - **Message History** - Complete conversation state with role-based access
//!
//! # Quick Start
//!
//! Create and manage a conversation:
//! ```rust
//! use stood::agent::conversation::ConversationManager;
//! use stood::llm::models::Bedrock;
//!
//! let mut manager = ConversationManager::new();
//!
//! // Set up system context
//! manager.set_system_prompt(Some("You are a helpful AI assistant.".to_string()));
//!
//! // Add conversation turns
//! manager.add_user_message("What is the capital of France?");
//! manager.add_assistant_message("The capital of France is Paris.");
//!
//! // Format for Bedrock API
//! let model = stood::llm::string_model::StringModel::new("us.anthropic.claude-haiku-4-5-20251001-v1:0", stood::llm::traits::ProviderType::Bedrock);
//! let request = manager.format_for_bedrock(&model)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Context Window Management
//!
//! Control conversation limits to fit model constraints:
//! ```rust
//! use stood::agent::conversation::ConversationManager;
//!
//! // Create with custom limits
//! let mut manager = ConversationManager::with_limits(
//!     50,      // Max 50 messages
//!     100_000  // Max 100k tokens
//! );
//!
//! // Add many messages - old ones are automatically pruned
//! for i in 0..100 {
//!     manager.add_user_message(format!("Message {}", i));
//!     manager.add_assistant_message(format!("Response {}", i));
//! }
//!
//! // Only the most recent messages within limits are kept
//! assert!(manager.message_count() <= 50);
//! assert!(manager.estimate_token_count() <= 100_000);
//! ```
//!
//! # System Prompt Management
//!
//! Maintain consistent agent behavior across turns:
//! ```rust
//! use stood::agent::conversation::ConversationManager;
//!
//! let mut manager = ConversationManager::new();
//!
//! // Set initial system prompt
//! manager.set_system_prompt(Some(
//!     "You are a technical support agent. Be helpful and concise.".to_string()
//! ));
//!
//! // System prompt persists across all interactions
//! manager.add_user_message("How do I reset my password?");
//! manager.add_assistant_message("To reset your password, go to Settings > Security.");
//!
//! // Update system prompt if needed
//! manager.set_system_prompt(Some(
//!     "You are now a sales assistant. Be friendly and persuasive.".to_string()
//! ));
//! ```
//!
//! # Token Management
//!
//! Monitor and optimize token usage:
//! ```rust
//! use stood::agent::conversation::ConversationManager;
//!
//! let mut manager = ConversationManager::new();
//! manager.add_user_message("Hello");
//!
//! // Get token estimates
//! let tokens = manager.estimate_token_count();
//! println!("Current conversation uses ~{} tokens", tokens);
//!
//! // Get detailed summary
//! println!("{}", manager.summary());
//! ```
//!
//! # Message Access Patterns
//!
//! Access conversation history efficiently:
//! ```rust
//! use stood::agent::conversation::ConversationManager;
//! use stood::types::MessageRole;
//!
//! let mut manager = ConversationManager::new();
//! manager.add_user_message("Hello");
//! manager.add_assistant_message("Hi there!");
//!
//! // Get the last message
//! if let Some(last) = manager.last_message() {
//!     println!("Last message: {}", last.text().unwrap_or_default());
//! }
//!
//! // Get the last assistant response
//! if let Some(response) = manager.last_assistant_message() {
//!     println!("Assistant said: {}", response.text().unwrap_or_default());
//! }
//!
//! // Access all messages
//! for message in manager.messages().iter() {
//!     println!("{:?}: {}", message.role, message.text().unwrap_or_default());
//! }
//! ```
//!
//! # Error Recovery
//!
//! Handle conversation state during errors:
//! ```rust
//! use stood::agent::conversation::ConversationManager;
//! use stood::types::MessageRole;
//!
//! let mut manager = ConversationManager::new();
//! manager.add_user_message("Generate some code");
//! manager.add_assistant_message("Here's some incomplete code...");
//!
//! // Remove failed assistant response and retry
//! if manager.remove_last_if_role(MessageRole::Assistant) {
//!     println!("Removed failed response, retrying...");
//!     // Retry with different parameters
//! }
//! ```
//!
//! # Architecture
//!
//! The conversation manager handles three key responsibilities:
//!
//! 1. **Message Storage** - Maintains conversation history with role tracking
//! 2. **Context Management** - Automatically prunes messages to fit limits
//! 3. **API Formatting** - Converts internal format to Bedrock-compatible JSON
//!
//! Context window management uses a two-phase approach:
//! - First, enforce message count limits by removing oldest messages
//! - Then, enforce token limits by removing messages until under threshold
//!
//! See [conversation patterns](../../docs/patterns.wiki#conversation-management) for advanced usage.
//!
//! # Performance
//!
//! - Message addition: O(1) with occasional O(n) for context pruning
//! - Token estimation: O(n) linear scan of message content
//! - Bedrock formatting: O(n) with zero-copy where possible
//! - Memory usage: Scales linearly with conversation length up to limits

use crate::llm::traits::LlmModel;
use crate::types::{Message, MessageRole, Messages};
use crate::Result;
use serde_json::{json, Value};

/// Manages conversation history and context window for an agent
#[derive(Debug, Clone)]
pub struct ConversationManager {
    /// The conversation history
    messages: Messages,
    /// Maximum number of messages to keep in history
    max_messages: usize,
    /// Maximum number of tokens to allow in context window
    max_tokens: usize,
    /// System prompt to include with requests
    system_prompt: Option<String>,
}

impl ConversationManager {
    /// Create a new conversation manager
    pub fn new() -> Self {
        Self {
            messages: Messages::new(),
            max_messages: 100,   // Default limit
            max_tokens: 100_000, // Default token limit
            system_prompt: None,
        }
    }

    /// Create a conversation manager with custom limits
    pub fn with_limits(max_messages: usize, max_tokens: usize) -> Self {
        Self {
            messages: Messages::new(),
            max_messages,
            max_tokens,
            system_prompt: None,
        }
    }

    /// Set the system prompt
    pub fn set_system_prompt(&mut self, prompt: Option<String>) {
        self.system_prompt = prompt;
    }

    /// Get the system prompt
    pub fn system_prompt(&self) -> Option<&str> {
        self.system_prompt.as_deref()
    }

    /// Add a message to the conversation
    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
        self.manage_context_window();
    }

    /// Add a user message
    pub fn add_user_message<S: Into<String>>(&mut self, text: S) {
        let message = Message::user(text);
        self.add_message(message);
    }

    /// Add an assistant message
    pub fn add_assistant_message<S: Into<String>>(&mut self, text: S) {
        let message = Message::assistant(text);
        self.add_message(message);
    }

    /// Get all messages in the conversation
    pub fn messages(&self) -> &Messages {
        &self.messages
    }

    /// Get messages with system prompt included (creates a new Messages struct)
    pub fn messages_with_system_prompt(&self) -> Messages {
        let mut messages = self.messages.clone();
        messages.system_prompt = self.system_prompt.clone();
        messages
    }

    /// Get the number of messages
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// Check if the conversation is empty
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Clear all messages
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Remove the last message if it matches the given role
    pub fn remove_last_if_role(&mut self, role: MessageRole) -> bool {
        if let Some(last_msg) = self.messages.messages.last() {
            if last_msg.role == role {
                self.messages.messages.pop();
                return true;
            }
        }
        false
    }

    /// Get the last message
    pub fn last_message(&self) -> Option<&Message> {
        self.messages.last()
    }

    /// Get the last assistant message
    pub fn last_assistant_message(&self) -> Option<&Message> {
        self.messages.last_assistant_message()
    }

    /// Estimate token count for the conversation (rough estimation)
    pub fn estimate_token_count(&self) -> usize {
        let mut total = 0;

        // Add system prompt tokens if present
        if let Some(system) = &self.system_prompt {
            total += estimate_text_tokens(system);
        }

        // Add message tokens
        for message in self.messages.iter() {
            if let Some(text) = message.text() {
                total += estimate_text_tokens(&text);
                total += 10; // Overhead per message (role, formatting, etc.)
            }
        }

        total
    }

    /// Format messages for Bedrock API
    pub fn format_for_bedrock(&self, model: &dyn LlmModel) -> Result<Value> {
        let mut claude_messages = Vec::new();

        // Convert messages to Claude format
        for message in self.messages.iter() {
            let role = match message.role {
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
                MessageRole::System => continue, // System messages handled separately
            };

            let content = message.text().unwrap_or_default();
            if content.is_empty() {
                continue; // Skip empty messages
            }

            claude_messages.push(json!({
                "role": role,
                "content": content
            }));
        }

        let mut request = json!({
            "messages": claude_messages,
            "max_tokens": model.default_max_tokens(),
            "temperature": model.default_temperature(),
            "anthropic_version": "bedrock-2023-05-31"
        });

        // Add system prompt if present
        if let Some(system) = &self.system_prompt {
            request["system"] = json!(system);
        }

        Ok(request)
    }

    /// Manage context window by removing old messages if limits are exceeded
    fn manage_context_window(&mut self) {
        // First, check message count limit
        if self.messages.len() > self.max_messages {
            let messages_to_remove = self.messages.len() - self.max_messages;
            self.messages.messages.drain(0..messages_to_remove);
        }

        // Then check token limit (rough estimation)
        while self.estimate_token_count() > self.max_tokens && self.messages.len() > 1 {
            // Remove the oldest message (but try to keep pairs if possible)
            self.messages.messages.remove(0);
        }
    }

    /// Get conversation summary for debugging
    pub fn summary(&self) -> String {
        format!(
            "ConversationManager: {} messages, ~{} tokens, limits: {}/{} messages, {}/{} tokens",
            self.message_count(),
            self.estimate_token_count(),
            self.message_count(),
            self.max_messages,
            self.estimate_token_count(),
            self.max_tokens
        )
    }
}

impl Default for ConversationManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Rough token estimation for text (approximately 4 characters per token)
fn estimate_text_tokens(text: &str) -> usize {
    // Simple estimation: roughly 4 characters per token
    // This is a rough approximation for planning purposes
    text.len().div_ceil(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversation_manager_creation() {
        let manager = ConversationManager::new();
        assert_eq!(manager.message_count(), 0);
        assert!(manager.is_empty());
        assert!(manager.system_prompt().is_none());
    }

    #[test]
    fn test_conversation_manager_with_limits() {
        let manager = ConversationManager::with_limits(10, 1000);
        assert_eq!(manager.max_messages, 10);
        assert_eq!(manager.max_tokens, 1000);
    }

    #[test]
    fn test_system_prompt_management() {
        let mut manager = ConversationManager::new();

        manager.set_system_prompt(Some("You are a helpful assistant".to_string()));
        assert_eq!(manager.system_prompt(), Some("You are a helpful assistant"));

        manager.set_system_prompt(None);
        assert!(manager.system_prompt().is_none());
    }

    #[test]
    fn test_message_addition() {
        let mut manager = ConversationManager::new();

        manager.add_user_message("Hello");
        assert_eq!(manager.message_count(), 1);
        assert!(!manager.is_empty());

        manager.add_assistant_message("Hi there!");
        assert_eq!(manager.message_count(), 2);

        let last_msg = manager.last_message().unwrap();
        assert_eq!(last_msg.role, MessageRole::Assistant);
        assert_eq!(last_msg.text(), Some("Hi there!".to_string()));

        let last_assistant = manager.last_assistant_message().unwrap();
        assert_eq!(last_assistant.text(), Some("Hi there!".to_string()));
    }

    #[test]
    fn test_conversation_clear() {
        let mut manager = ConversationManager::new();

        manager.add_user_message("Hello");
        manager.add_assistant_message("Hi");
        assert_eq!(manager.message_count(), 2);

        manager.clear();
        assert_eq!(manager.message_count(), 0);
        assert!(manager.is_empty());
    }

    #[test]
    fn test_token_estimation() {
        let mut manager = ConversationManager::new();

        // Empty conversation should have minimal tokens
        assert_eq!(manager.estimate_token_count(), 0);

        // Add system prompt
        manager.set_system_prompt(Some("You are helpful".to_string())); // ~4 tokens
        let tokens_with_system = manager.estimate_token_count();
        assert!(tokens_with_system > 0);

        // Add a message
        manager.add_user_message("Hello world"); // ~2 tokens + 10 overhead = 12 tokens
        let tokens_with_message = manager.estimate_token_count();
        assert!(tokens_with_message > tokens_with_system);
    }

    #[test]
    fn test_message_limit_management() {
        let mut manager = ConversationManager::with_limits(3, 100_000);

        // Add 5 messages (exceeds limit of 3)
        manager.add_user_message("Message 1");
        manager.add_assistant_message("Response 1");
        manager.add_user_message("Message 2");
        manager.add_assistant_message("Response 2");
        manager.add_user_message("Message 3");

        // Should only keep the last 3 messages
        assert_eq!(manager.message_count(), 3);

        // First message should be "Message 2" (oldest messages were removed)
        let first_msg = &manager.messages().messages[0];
        assert_eq!(first_msg.text(), Some("Message 2".to_string()));
    }

    #[test]
    fn test_bedrock_formatting() {
        let mut manager = ConversationManager::new();
        manager.set_system_prompt(Some("You are helpful".to_string()));
        manager.add_user_message("What is 2+2?");
        manager.add_assistant_message("2+2 equals 4");

        let model = crate::llm::string_model::StringModel::new("us.anthropic.claude-haiku-4-5-20251001-v1:0", crate::llm::traits::ProviderType::Bedrock);
        let formatted = manager.format_for_bedrock(&model).unwrap();

        // Should have system prompt
        assert_eq!(formatted["system"], "You are helpful");

        // Should have messages array
        let messages = formatted["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);

        // Check first message
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "What is 2+2?");

        // Check second message
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["content"], "2+2 equals 4");

        // Should have model-specific parameters
        assert_eq!(formatted["max_tokens"], model.default_max_tokens());
        assert_eq!(formatted["temperature"], model.default_temperature());
        assert_eq!(formatted["anthropic_version"], "bedrock-2023-05-31");
    }

    #[test]
    fn test_conversation_summary() {
        let mut manager = ConversationManager::with_limits(10, 1000);
        manager.add_user_message("Hello");

        let summary = manager.summary();
        assert!(summary.contains("1 messages"));
        assert!(summary.contains("1/10 messages"));
        assert!(summary.contains("1000 tokens"));
    }

    #[test]
    fn test_empty_message_handling() {
        let mut manager = ConversationManager::new();
        manager.add_message(Message::user(""));
        manager.add_user_message("Hello");

        let model = crate::llm::string_model::StringModel::new("us.anthropic.claude-haiku-4-5-20251001-v1:0", crate::llm::traits::ProviderType::Bedrock);
        let formatted = manager.format_for_bedrock(&model).unwrap();

        // Should skip empty messages in formatting
        let messages = formatted["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["content"], "Hello");
    }
}
