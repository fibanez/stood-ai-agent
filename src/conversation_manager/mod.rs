//! Conversation management for maintaining conversation coherence and context limits.
//!
//! This module provides conversation management capabilities following the Python reference
//! implementation patterns. It handles conversation size management, tool-aware message pruning,
//! and context window overflow recovery.
//!
//! Key components:
//! - `ConversationManager` trait: Abstract interface for conversation strategies
//! - `SlidingWindowManager`: Maintains conversation within token/message limits
//! - `NullConversationManager`: No-op manager for testing scenarios
//! - Tool-aware pruning: Preserves tool use/result pairs during conversation trimming
//! - Context overflow recovery: Handles model context window limits gracefully

use async_trait::async_trait;
use std::collections::HashSet;
use tracing::{debug, info, warn};

use crate::{
    context_manager::{ContextConfig as ContextManagerConfig, ContextManager},
    message_processor::MessageProcessor,
    types::{ContentBlock, Messages},
    Result, StoodError,
};

/// Configuration for conversation management behavior
#[derive(Debug, Clone)]
pub struct ConversationConfig {
    /// Maximum number of messages to keep in conversation
    pub max_messages: usize,
    /// Whether to enable tool-aware pruning (preserves tool use/result pairs)
    pub enable_tool_aware_pruning: bool,
    /// Whether to automatically clean dangling messages
    pub auto_clean_dangling: bool,
    /// Whether to enable conversation summarization
    pub enable_summarization: bool,
    /// Whether to enable context-aware management (token-based limits)
    pub enable_context_management: bool,
    /// Context manager configuration (for token-based limits)
    pub context_config: Option<ContextManagerConfig>,
}

impl Default for ConversationConfig {
    fn default() -> Self {
        Self {
            max_messages: 40, // Python reference default
            enable_tool_aware_pruning: true,
            auto_clean_dangling: true,
            enable_summarization: false,
            enable_context_management: true,
            context_config: None, // Use default context config
        }
    }
}

/// Result of a conversation management operation
#[derive(Debug, Clone)]
pub struct ManagementResult {
    /// Whether any changes were made to the conversation
    pub changes_made: bool,
    /// Number of messages removed
    pub messages_removed: usize,
    /// Number of dangling messages cleaned
    pub dangling_cleaned: usize,
    /// Description of the management operation
    pub description: String,
}

impl ManagementResult {
    /// Create a new management result
    pub fn new(
        changes_made: bool,
        messages_removed: usize,
        dangling_cleaned: usize,
        description: String,
    ) -> Self {
        Self {
            changes_made,
            messages_removed,
            dangling_cleaned,
            description,
        }
    }

    /// Create a result indicating no changes
    pub fn no_changes() -> Self {
        Self {
            changes_made: false,
            messages_removed: 0,
            dangling_cleaned: 0,
            description: "No conversation management needed".to_string(),
        }
    }
}

/// Abstract interface for conversation management strategies.
///
/// This trait provides methods for managing conversation size and handling
/// context window overflow scenarios. Implementations should follow the
/// Python reference patterns for consistent behavior.
#[async_trait]
pub trait ConversationManager: Send + Sync {
    /// Apply conversation management after every event loop model interaction.
    ///
    /// This method is called after tool execution and model responses
    /// to ensure the conversation stays within configured limits.
    ///
    /// # Arguments
    /// * `messages` - The conversation messages to manage
    ///
    /// # Returns
    /// * `Result<ManagementResult>` - Information about management operations performed
    async fn apply_management(&self, messages: &mut Messages) -> Result<ManagementResult>;

    /// Reduce conversation context when context window is exceeded.
    ///
    /// This method is called when the model throws a context window overflow
    /// exception. It should attempt to reduce the conversation size to fit
    /// within the model's context window.
    ///
    /// # Arguments
    /// * `messages` - The conversation messages to reduce
    /// * `error_message` - Optional error message from the model provider
    ///
    /// # Returns
    /// * `Result<ManagementResult>` - Information about context reduction performed
    async fn reduce_context(
        &self,
        messages: &mut Messages,
        error_message: Option<&str>,
    ) -> Result<ManagementResult>;

    /// Get the configuration for this conversation manager
    fn config(&self) -> &ConversationConfig;

    /// Check if conversation is within limits
    fn is_within_limits(&self, messages: &Messages) -> bool {
        messages.messages.len() <= self.config().max_messages
    }
}

/// Sliding window conversation manager that maintains conversation within size limits.
///
/// This implementation follows the Python reference `SlidingWindowConversationManager`
/// with tool-aware pruning to preserve conversation coherence.
pub struct SlidingWindowManager {
    config: ConversationConfig,
    context_manager: Option<ContextManager>,
}

impl SlidingWindowManager {
    /// Create a new sliding window manager with default configuration
    pub fn new() -> Self {
        let config = ConversationConfig::default();
        let context_manager = if config.enable_context_management {
            Some(ContextManager::new())
        } else {
            None
        };

        Self {
            config,
            context_manager,
        }
    }

    /// Create a new sliding window manager with custom configuration
    pub fn with_config(config: ConversationConfig) -> Self {
        let context_manager = if config.enable_context_management {
            match &config.context_config {
                Some(ctx_config) => Some(ContextManager::with_config(ctx_config.clone())),
                None => Some(ContextManager::new()),
            }
        } else {
            None
        };

        Self {
            config,
            context_manager,
        }
    }

    /// Create a new sliding window manager with custom window size
    pub fn with_window_size(window_size: usize) -> Self {
        let config = ConversationConfig {
            max_messages: window_size,
            ..ConversationConfig::default()
        };
        Self::with_config(config)
    }

    /// Remove dangling messages from the conversation.
    ///
    /// Dangling messages are:
    /// - Tool result messages without preceding tool use
    /// - Tool use messages without following tool result
    ///
    /// This follows the Python reference `_remove_dangling_messages` implementation.
    fn remove_dangling_messages(&self, messages: &mut Messages) -> usize {
        if !self.config.auto_clean_dangling {
            return 0;
        }

        debug!("Removing dangling messages from conversation");

        let mut tool_use_ids = HashSet::new();
        let mut tool_result_ids = HashSet::new();
        let mut indices_to_remove = Vec::new();

        // First pass: collect all tool use and tool result IDs
        for message in &messages.messages {
            for content in &message.content {
                match content {
                    ContentBlock::ToolUse { id, .. } => {
                        tool_use_ids.insert(id.clone());
                    }
                    ContentBlock::ToolResult { tool_use_id, .. } => {
                        tool_result_ids.insert(tool_use_id.clone());
                    }
                    _ => {}
                }
            }
        }

        // Second pass: identify dangling messages
        for (msg_idx, message) in messages.messages.iter().enumerate() {
            let mut has_dangling_content = false;

            for content in &message.content {
                match content {
                    ContentBlock::ToolUse { id, .. } => {
                        // Tool use without corresponding result is dangling
                        if !tool_result_ids.contains(id) {
                            has_dangling_content = true;
                            debug!("Found dangling tool use: {}", id);
                        }
                    }
                    ContentBlock::ToolResult { tool_use_id, .. } => {
                        // Tool result without corresponding use is dangling
                        if !tool_use_ids.contains(tool_use_id) {
                            has_dangling_content = true;
                            debug!("Found dangling tool result: {}", tool_use_id);
                        }
                    }
                    _ => {}
                }
            }

            // If message has only dangling content, mark for removal
            if has_dangling_content {
                let non_dangling_content = message.content.iter().any(|content| {
                    !matches!(
                        content,
                        ContentBlock::ToolUse { .. } | ContentBlock::ToolResult { .. }
                    )
                });

                if !non_dangling_content {
                    indices_to_remove.push(msg_idx);
                }
            }
        }

        // Remove dangling messages in reverse order
        let removed_count = indices_to_remove.len();
        for &idx in indices_to_remove.iter().rev() {
            messages.messages.remove(idx);
            debug!("Removed dangling message at index {}", idx);
        }

        removed_count
    }

    /// Find a safe trim index that preserves tool use/result pairs.
    ///
    /// This ensures that trimming doesn't create orphaned tool results
    /// or incomplete tool use sequences.
    fn find_safe_trim_index(&self, messages: &Messages, target_size: usize) -> usize {
        if messages.messages.len() <= target_size {
            return 0;
        }

        let messages_to_remove = messages.messages.len() - target_size;
        let mut trim_index = messages_to_remove;

        // If tool-aware pruning is disabled, just return the simple trim index
        if !self.config.enable_tool_aware_pruning {
            return trim_index.min(messages.messages.len());
        }

        // Ensure we don't start with a tool result (which would be orphaned)
        while trim_index < messages.messages.len() {
            let message = &messages.messages[trim_index];

            // Check if this message has tool results
            let has_tool_results = message
                .content
                .iter()
                .any(|content| matches!(content, ContentBlock::ToolResult { .. }));

            if has_tool_results {
                // Look backwards to find the corresponding tool use
                let mut found_tool_use_after_trim = false;
                for i in trim_index..messages.messages.len() {
                    let check_message = &messages.messages[i];
                    if check_message
                        .content
                        .iter()
                        .any(|content| matches!(content, ContentBlock::ToolUse { .. }))
                    {
                        found_tool_use_after_trim = true;
                        break;
                    }
                }

                // Look backwards to find the corresponding tool use that would be trimmed
                let mut found_tool_use_before_trim = false;
                for i in (0..trim_index).rev() {
                    let prev_message = &messages.messages[i];
                    if prev_message
                        .content
                        .iter()
                        .any(|content| matches!(content, ContentBlock::ToolUse { .. }))
                    {
                        found_tool_use_before_trim = true;
                        break;
                    }
                }

                if found_tool_use_before_trim && !found_tool_use_after_trim {
                    // The tool use would be trimmed but result kept - find the tool use and keep both
                    for i in (0..trim_index).rev() {
                        let prev_message = &messages.messages[i];
                        if prev_message
                            .content
                            .iter()
                            .any(|content| matches!(content, ContentBlock::ToolUse { .. }))
                        {
                            trim_index = i; // Move trim index to preserve tool use
                            break;
                        }
                    }
                    break; // We've found our final trim index
                }
            }

            // Check if this message has tool uses that need following results
            let has_tool_uses = message
                .content
                .iter()
                .any(|content| matches!(content, ContentBlock::ToolUse { .. }));

            if has_tool_uses && trim_index + 1 < messages.messages.len() {
                let next_message = &messages.messages[trim_index + 1];
                let next_has_tool_results = next_message
                    .content
                    .iter()
                    .any(|content| matches!(content, ContentBlock::ToolResult { .. }));

                if !next_has_tool_results {
                    // This tool use doesn't have a following result, move trim index forward
                    trim_index += 1;
                    continue;
                }
            }

            // This is a safe trim point
            break;
        }

        // If we couldn't find a safe trim point, return a minimal trim
        if trim_index >= messages.messages.len() {
            trim_index = messages.messages.len().saturating_sub(1);
        }

        trim_index
    }

    /// Trim messages from the conversation while preserving tool coherence
    fn trim_messages(&self, messages: &mut Messages, target_size: usize) -> usize {
        if messages.messages.len() <= target_size {
            return 0;
        }

        let trim_index = self.find_safe_trim_index(messages, target_size);
        let removed_count = trim_index;

        if removed_count > 0 {
            messages.messages.drain(0..trim_index);
            info!("Trimmed {} messages from conversation", removed_count);
        }

        removed_count
    }
}

#[async_trait]
impl ConversationManager for SlidingWindowManager {
    async fn apply_management(&self, messages: &mut Messages) -> Result<ManagementResult> {
        debug!("Applying sliding window conversation management");

        let _initial_count = messages.messages.len();
        let mut total_changes = false;
        let mut total_removed = 0;
        let mut total_dangling = 0;
        let mut context_managed = false;

        // First, clean any dangling messages
        if self.config.auto_clean_dangling {
            let dangling_removed = self.remove_dangling_messages(messages);
            if dangling_removed > 0 {
                total_changes = true;
                total_dangling = dangling_removed;
            }
        }

        // Second, apply context management if enabled
        if let Some(context_manager) = &self.context_manager {
            if context_manager.needs_management(messages) {
                debug!("Context management needed, applying context limits");
                match context_manager.manage_context(messages) {
                    Ok(context_result) => {
                        if context_result.changes_made {
                            total_changes = true;
                            total_removed += context_result.messages_removed;
                            context_managed = true;
                            info!(
                                "Context management applied: removed {} messages, saved {} chars",
                                context_result.messages_removed, context_result.characters_saved
                            );
                        }
                    }
                    Err(e) => {
                        warn!(
                            "Context management failed: {}, falling back to message limits",
                            e
                        );
                    }
                }
            }
        }

        // Third, apply window size limit (if not already managed by context)
        if !context_managed && !self.is_within_limits(messages) {
            let removed = self.trim_messages(messages, self.config.max_messages);
            if removed > 0 {
                total_changes = true;
                total_removed += removed;
            }
        }

        let description = if total_changes {
            let context_msg = if context_managed {
                " (context-aware)"
            } else {
                ""
            };
            format!(
                "Applied sliding window management{}: removed {} messages, cleaned {} dangling messages",
                context_msg, total_removed, total_dangling
            )
        } else {
            "Conversation within limits, no management needed".to_string()
        };

        Ok(ManagementResult::new(
            total_changes,
            total_removed,
            total_dangling,
            description,
        ))
    }

    async fn reduce_context(
        &self,
        messages: &mut Messages,
        error_message: Option<&str>,
    ) -> Result<ManagementResult> {
        warn!("Reducing context due to overflow: {:?}", error_message);

        let initial_count = messages.messages.len();

        // Try more aggressive trimming - reduce to 75% of window size
        let target_size = (self.config.max_messages * 3) / 4;

        let mut total_removed = 0;
        let mut total_dangling = 0;

        // Clean dangling messages first
        if self.config.auto_clean_dangling {
            total_dangling = self.remove_dangling_messages(messages);
        }

        // Apply message processor cleaning
        let processor_result = MessageProcessor::clean_orphaned_empty_tool_uses(messages);
        if processor_result.changes_made {
            info!("Cleaned orphaned tools during context reduction");
        }

        // Trim to target size
        total_removed += self.trim_messages(messages, target_size);

        let final_count = messages.messages.len();
        let total_changes = final_count != initial_count;

        if total_changes {
            info!(
                "Context reduction: {} -> {} messages ({} removed, {} dangling cleaned)",
                initial_count, final_count, total_removed, total_dangling
            );
        } else {
            warn!("Context reduction made no changes - conversation may still be too large");
        }

        Ok(ManagementResult::new(
            total_changes,
            total_removed,
            total_dangling,
            format!(
                "Context reduction: removed {} messages, cleaned {} dangling",
                total_removed, total_dangling
            ),
        ))
    }

    fn config(&self) -> &ConversationConfig {
        &self.config
    }
}

impl Default for SlidingWindowManager {
    fn default() -> Self {
        Self::new()
    }
}

/// No-op conversation manager for testing scenarios.
///
/// This implementation follows the Python reference `NullConversationManager`
/// and does not perform any conversation management. It's useful for testing
/// scenarios where conversation management should be disabled.
pub struct NullConversationManager {
    config: ConversationConfig,
}

impl NullConversationManager {
    /// Create a new null conversation manager
    pub fn new() -> Self {
        Self {
            config: ConversationConfig {
                max_messages: usize::MAX, // No limit
                enable_tool_aware_pruning: false,
                auto_clean_dangling: false,
                enable_summarization: false,
                enable_context_management: false, // Disabled for null manager
                context_config: None,
            },
        }
    }
}

#[async_trait]
impl ConversationManager for NullConversationManager {
    async fn apply_management(&self, _messages: &mut Messages) -> Result<ManagementResult> {
        debug!("Null conversation manager: no management applied");
        Ok(ManagementResult::no_changes())
    }

    async fn reduce_context(
        &self,
        _messages: &mut Messages,
        error_message: Option<&str>,
    ) -> Result<ManagementResult> {
        // Following Python reference: NullConversationManager raises exception on reduce_context
        Err(StoodError::configuration_error(format!(
            "Context window overflow with null conversation manager: {:?}",
            error_message.unwrap_or("Unknown overflow error")
        )))
    }

    fn config(&self) -> &ConversationConfig {
        &self.config
    }
}

impl Default for NullConversationManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Conversation manager factory for creating different manager types
pub struct ConversationManagerFactory;

impl ConversationManagerFactory {
    /// Create a sliding window manager with default configuration
    pub fn sliding_window() -> Box<dyn ConversationManager> {
        Box::new(SlidingWindowManager::new())
    }

    /// Create a sliding window manager with custom window size
    pub fn sliding_window_with_size(window_size: usize) -> Box<dyn ConversationManager> {
        Box::new(SlidingWindowManager::with_window_size(window_size))
    }

    /// Create a sliding window manager with custom configuration
    pub fn sliding_window_with_config(config: ConversationConfig) -> Box<dyn ConversationManager> {
        Box::new(SlidingWindowManager::with_config(config))
    }

    /// Create a null conversation manager for testing
    pub fn null() -> Box<dyn ConversationManager> {
        Box::new(NullConversationManager::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ContentBlock, Message, MessageRole, Messages, ToolResultContent};
    use chrono::Utc;
    use serde_json::json;
    use std::collections::HashMap;
    use uuid::Uuid;

    fn create_test_message(role: MessageRole, text: &str) -> Message {
        Message {
            id: Uuid::new_v4(),
            role,
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
            metadata: HashMap::new(),
            timestamp: Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_sliding_window_manager_creation() {
        let manager = SlidingWindowManager::new();
        assert_eq!(manager.config().max_messages, 40);
        assert!(manager.config().enable_tool_aware_pruning);
        assert!(manager.config().auto_clean_dangling);
    }

    #[tokio::test]
    async fn test_sliding_window_manager_custom_config() {
        let config = ConversationConfig {
            max_messages: 20,
            enable_tool_aware_pruning: false,
            auto_clean_dangling: false,
            enable_summarization: true,
            enable_context_management: false,
            context_config: None,
        };

        let manager = SlidingWindowManager::with_config(config);
        assert_eq!(manager.config().max_messages, 20);
        assert!(!manager.config().enable_tool_aware_pruning);
        assert!(!manager.config().auto_clean_dangling);
        assert!(manager.config().enable_summarization);
    }

    #[tokio::test]
    async fn test_apply_management_within_limits() {
        let manager = SlidingWindowManager::with_window_size(5);
        let mut messages = Messages::new();

        // Add 3 messages (within limit)
        for i in 0..3 {
            messages.push(Message::new(
                MessageRole::User,
                vec![ContentBlock::text(format!("Message {}", i))],
            ));
        }

        let result = manager.apply_management(&mut messages).await.unwrap();

        assert!(!result.changes_made);
        assert_eq!(result.messages_removed, 0);
        assert_eq!(messages.messages.len(), 3);
    }

    #[tokio::test]
    async fn test_apply_management_over_limits() {
        let manager = SlidingWindowManager::with_window_size(3);
        let mut messages = Messages::new();

        // Add 5 messages (over limit)
        for i in 0..5 {
            messages.push(Message::new(
                MessageRole::User,
                vec![ContentBlock::text(format!("Message {}", i))],
            ));
        }

        let result = manager.apply_management(&mut messages).await.unwrap();

        assert!(result.changes_made);
        assert_eq!(result.messages_removed, 2);
        assert_eq!(messages.messages.len(), 3);

        // Check that the last 3 messages remain
        assert!(messages.messages[0].text().unwrap().contains("Message 2"));
        assert!(messages.messages[1].text().unwrap().contains("Message 3"));
        assert!(messages.messages[2].text().unwrap().contains("Message 4"));
    }

    #[tokio::test]
    async fn test_context_management_integration() {
        use crate::context_manager::ContextConfig as ContextManagerConfig;

        let context_config = ContextManagerConfig {
            max_tokens: 100, // Very small limit to trigger management
            buffer_percentage: 0.5,
            chars_per_token: 1.0, // 1 char = 1 token for testing
            ..Default::default()
        };

        let config = ConversationConfig {
            max_messages: 100, // High message limit so context takes precedence
            enable_context_management: true,
            context_config: Some(context_config),
            ..Default::default()
        };

        let manager = SlidingWindowManager::with_config(config);

        // Create messages that exceed token limits but not message limits
        let mut messages = Messages { messages: vec![], system_prompt: None };
        for i in 0..5 {
            messages.messages.push(create_test_message(
                crate::types::MessageRole::User,
                &format!(
                    "This is a very long message number {} that will exceed our small token limit",
                    i
                ),
            ));
        }

        // Verify we have messages and they exceed the context limit
        assert!(messages.messages.len() < 100); // Under message limit
        assert!(manager.context_manager.is_some());

        let result = manager.apply_management(&mut messages).await.unwrap();

        // Should have removed messages due to context management
        assert!(result.changes_made);
        assert!(result.messages_removed > 0);
        assert!(result.description.contains("context-aware"));
    }

    #[tokio::test]
    async fn test_tool_aware_pruning() {
        let manager = SlidingWindowManager::with_window_size(4); // Allow room for tool pair + recent messages
        let mut messages = Messages::new();

        // Add some initial messages
        messages.push(Message::new(
            MessageRole::User,
            vec![ContentBlock::text("Initial message 1")],
        ));

        messages.push(Message::new(
            MessageRole::User,
            vec![ContentBlock::text("Initial message 2")],
        ));

        // Add tool use message
        messages.push(Message::new(
            MessageRole::Assistant,
            vec![ContentBlock::ToolUse {
                id: "tool_1".to_string(),
                name: "calculator".to_string(),
                input: json!({"expression": "2+2"}),
            }],
        ));

        // Add tool result message
        messages.push(Message::new(
            MessageRole::User,
            vec![ContentBlock::ToolResult {
                tool_use_id: "tool_1".to_string(),
                content: ToolResultContent::text("4"),
                is_error: false,
            }],
        ));

        // Add more messages to exceed limit
        for i in 0..3 {
            messages.push(Message::new(
                MessageRole::User,
                vec![ContentBlock::text(format!("Recent message {}", i))],
            ));
        }

        let result = manager.apply_management(&mut messages).await.unwrap();

        assert!(result.changes_made);
        // Should be 5 or fewer (tool-aware pruning might keep more to preserve tool pairs)
        assert!(messages.messages.len() <= 5);
        assert!(messages.messages.len() >= 4);

        // Should preserve tool use/result pair and keep recent messages
        let has_tool_use = messages.messages.iter().any(|msg| {
            msg.content
                .iter()
                .any(|content| matches!(content, ContentBlock::ToolUse { .. }))
        });
        let has_tool_result = messages.messages.iter().any(|msg| {
            msg.content
                .iter()
                .any(|content| matches!(content, ContentBlock::ToolResult { .. }))
        });

        // The first two "Initial message" messages should be trimmed
        let has_initial_messages = messages.messages.iter().any(|msg| {
            msg.text()
                .map_or(false, |text| text.contains("Initial message"))
        });

        assert!(has_tool_use, "Tool use should be preserved");
        assert!(has_tool_result, "Tool result should be preserved");
        assert!(!has_initial_messages, "Initial messages should be trimmed");
    }

    #[tokio::test]
    async fn test_remove_dangling_messages() {
        let manager = SlidingWindowManager::new();
        let mut messages = Messages::new();

        // Add orphaned tool result (no corresponding tool use)
        messages.push(Message::new(
            MessageRole::User,
            vec![ContentBlock::ToolResult {
                tool_use_id: "orphan_1".to_string(),
                content: ToolResultContent::text("orphaned result"),
                is_error: false,
            }],
        ));

        // Add valid message
        messages.push(Message::new(
            MessageRole::User,
            vec![ContentBlock::text("Valid message")],
        ));

        // Add orphaned tool use (no corresponding tool result)
        messages.push(Message::new(
            MessageRole::Assistant,
            vec![ContentBlock::ToolUse {
                id: "orphan_2".to_string(),
                name: "calculator".to_string(),
                input: json!({"expression": "2+2"}),
            }],
        ));

        let result = manager.apply_management(&mut messages).await.unwrap();

        assert!(result.changes_made);
        assert_eq!(result.dangling_cleaned, 2);
        assert_eq!(messages.messages.len(), 1);

        // Only the valid message should remain
        assert!(messages.messages[0]
            .text()
            .unwrap()
            .contains("Valid message"));
    }

    #[tokio::test]
    async fn test_reduce_context() {
        let manager = SlidingWindowManager::with_window_size(10);
        let mut messages = Messages::new();

        // Add many messages
        for i in 0..15 {
            messages.push(Message::new(
                MessageRole::User,
                vec![ContentBlock::text(format!("Message {}", i))],
            ));
        }

        let result = manager
            .reduce_context(&mut messages, Some("Context too large"))
            .await
            .unwrap();

        assert!(result.changes_made);
        assert!(result.messages_removed > 0);

        // Should reduce to 75% of window size (7-8 messages)
        assert!(messages.messages.len() <= 8);
        assert!(messages.messages.len() >= 7);
    }

    #[tokio::test]
    async fn test_null_conversation_manager() {
        let manager = NullConversationManager::new();
        let mut messages = Messages::new();

        // Add many messages
        for i in 0..100 {
            messages.push(Message::new(
                MessageRole::User,
                vec![ContentBlock::text(format!("Message {}", i))],
            ));
        }

        let result = manager.apply_management(&mut messages).await.unwrap();

        assert!(!result.changes_made);
        assert_eq!(result.messages_removed, 0);
        assert_eq!(messages.messages.len(), 100); // No changes
    }

    #[tokio::test]
    async fn test_null_conversation_manager_reduce_context_fails() {
        let manager = NullConversationManager::new();
        let mut messages = Messages::new();

        let result = manager
            .reduce_context(&mut messages, Some("Context overflow"))
            .await;

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Context window overflow"));
    }

    #[tokio::test]
    async fn test_conversation_manager_factory() {
        let sliding = ConversationManagerFactory::sliding_window();
        assert_eq!(sliding.config().max_messages, 40);

        let sliding_custom = ConversationManagerFactory::sliding_window_with_size(20);
        assert_eq!(sliding_custom.config().max_messages, 20);

        let null = ConversationManagerFactory::null();
        assert_eq!(null.config().max_messages, usize::MAX);
    }

    #[tokio::test]
    async fn test_find_safe_trim_index() {
        let manager = SlidingWindowManager::with_window_size(5);
        let mut messages = Messages::new();

        // Add messages including tool use/result pair
        messages.push(Message::new(
            MessageRole::User,
            vec![ContentBlock::text("Message 0")],
        ));

        messages.push(Message::new(
            MessageRole::Assistant,
            vec![ContentBlock::ToolUse {
                id: "tool_1".to_string(),
                name: "calculator".to_string(),
                input: json!({"expression": "2+2"}),
            }],
        ));

        messages.push(Message::new(
            MessageRole::User,
            vec![ContentBlock::ToolResult {
                tool_use_id: "tool_1".to_string(),
                content: ToolResultContent::text("4"),
                is_error: false,
            }],
        ));

        for i in 3..8 {
            messages.push(Message::new(
                MessageRole::User,
                vec![ContentBlock::text(format!("Message {}", i))],
            ));
        }

        let trim_index = manager.find_safe_trim_index(&messages, 5);

        // Should find a safe trim point that doesn't break tool use/result pair
        assert!(trim_index < messages.messages.len());
    }
}
