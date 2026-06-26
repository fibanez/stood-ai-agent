//! Context management for tracking and controlling conversation context usage.
//!
//! This module provides context window management capabilities for preventing
//! context overflow errors and optimizing conversation size. It includes:
//!
//! - Token counting and estimation utilities
//! - Context window monitoring and tracking
//! - Proactive overflow prevention
//! - Priority-based message retention strategies
//! - Integration with conversation management systems

use std::collections::HashMap;
use tracing::{debug, info};

use crate::{
    types::{ContentBlock, Message, MessageRole, Messages},
    Result,
};

/// Configuration for context management behavior
#[derive(Debug, Clone)]
pub struct ContextConfig {
    /// Maximum estimated tokens to allow in context (defaults to Claude's ~200k)
    pub max_tokens: usize,
    /// Buffer percentage to keep below max (0.0-1.0, defaults to 0.85 = 85%)
    pub buffer_percentage: f32,
    /// Character-to-token ratio estimate (defaults to 4.0 chars per token)
    pub chars_per_token: f32,
    /// Whether to enable proactive overflow prevention
    pub enable_proactive_prevention: bool,
    /// Whether to use priority-based retention for important messages
    pub enable_priority_retention: bool,
    /// Minimum number of messages to always keep
    pub min_messages: usize,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_tokens: 200_000,     // Claude 3.5 Sonnet context window
            buffer_percentage: 0.85, // Keep 85% buffer
            chars_per_token: 4.0,    // Conservative estimate
            enable_proactive_prevention: true,
            enable_priority_retention: true,
            min_messages: 2, // Always keep at least 2 messages
        }
    }
}

/// Priority levels for message retention
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MessagePriority {
    /// System messages and initial context
    Critical = 4,
    /// Recent user messages and tool interactions
    High = 3,
    /// Assistant responses with tool usage
    Medium = 2,
    /// General conversation
    Normal = 1,
    /// Old messages that can be removed first
    Low = 0,
}

/// Context usage information
#[derive(Debug, Clone)]
pub struct ContextUsage {
    /// Estimated total tokens
    pub estimated_tokens: usize,
    /// Total character count
    pub character_count: usize,
    /// Number of messages
    pub message_count: usize,
    /// Breakdown by content type
    pub content_breakdown: ContentBreakdown,
    /// Usage as percentage of maximum
    pub usage_percentage: f32,
    /// Whether context is near the limit
    pub approaching_limit: bool,
    /// Whether context exceeds safe limits
    pub exceeds_safe_limit: bool,
}

/// Breakdown of context usage by content type
#[derive(Debug, Clone, Default)]
pub struct ContentBreakdown {
    /// Characters in text content
    pub text_chars: usize,
    /// Characters in tool use content
    pub tool_use_chars: usize,
    /// Characters in tool result content
    pub tool_result_chars: usize,
    /// Characters in thinking content
    pub thinking_chars: usize,
    /// Number of tool interactions
    pub tool_interactions: usize,
}

/// Result of context management operations
#[derive(Debug, Clone)]
pub struct ContextManagementResult {
    /// Usage before management
    pub before_usage: ContextUsage,
    /// Usage after management
    pub after_usage: ContextUsage,
    /// Number of messages removed
    pub messages_removed: usize,
    /// Number of characters saved
    pub characters_saved: usize,
    /// Whether any changes were made
    pub changes_made: bool,
    /// Messages removed by priority level
    pub removed_by_priority: HashMap<MessagePriority, usize>,
}

/// Context manager for tracking and controlling conversation context
#[derive(Debug, Clone)]
pub struct ContextManager {
    config: ContextConfig,
}

impl ContextManager {
    /// Create a new context manager with default configuration
    pub fn new() -> Self {
        Self {
            config: ContextConfig::default(),
        }
    }

    /// Create a new context manager with custom configuration
    pub fn with_config(config: ContextConfig) -> Self {
        Self { config }
    }

    /// Analyze current context usage
    pub fn analyze_usage(&self, messages: &Messages) -> ContextUsage {
        let breakdown = self.calculate_content_breakdown(messages);
        let character_count = breakdown.text_chars
            + breakdown.tool_use_chars
            + breakdown.tool_result_chars
            + breakdown.thinking_chars;

        let estimated_tokens =
            (character_count as f32 / self.config.chars_per_token).ceil() as usize;
        let usage_percentage = (estimated_tokens as f32 / self.config.max_tokens as f32) * 100.0;

        let safe_limit = (self.config.max_tokens as f32 * self.config.buffer_percentage) as usize;
        let approaching_limit = estimated_tokens > (safe_limit * 90 / 100); // 90% of safe limit
        let exceeds_safe_limit = estimated_tokens > safe_limit;

        debug!(
            "Context analysis: {}/{} tokens ({:.1}%), {} messages, {} chars",
            estimated_tokens,
            self.config.max_tokens,
            usage_percentage,
            messages.messages.len(),
            character_count
        );

        ContextUsage {
            estimated_tokens,
            character_count,
            message_count: messages.messages.len(),
            content_breakdown: breakdown,
            usage_percentage,
            approaching_limit,
            exceeds_safe_limit,
        }
    }

    /// Check if proactive management is needed
    pub fn needs_management(&self, messages: &Messages) -> bool {
        if !self.config.enable_proactive_prevention {
            return false;
        }

        let usage = self.analyze_usage(messages);
        usage.exceeds_safe_limit
            || (usage.approaching_limit && messages.messages.len() > self.config.min_messages)
    }

    /// Proactively manage context to prevent overflow
    pub fn manage_context(&self, messages: &mut Messages) -> Result<ContextManagementResult> {
        let before_usage = self.analyze_usage(messages);

        if !before_usage.exceeds_safe_limit && !before_usage.approaching_limit {
            debug!("Context management: No action needed, usage within limits");
            return Ok(ContextManagementResult {
                before_usage: before_usage.clone(),
                after_usage: before_usage,
                messages_removed: 0,
                characters_saved: 0,
                changes_made: false,
                removed_by_priority: HashMap::new(),
            });
        }

        info!(
            "Context management: Starting reduction, usage at {:.1}% ({} tokens)",
            before_usage.usage_percentage, before_usage.estimated_tokens
        );

        let original_count = messages.messages.len();
        let mut removed_by_priority = HashMap::new();

        // Apply management strategy
        if self.config.enable_priority_retention {
            self.priority_based_reduction(messages, &mut removed_by_priority)?;
        } else {
            self.simple_sliding_window_reduction(messages)?;
        }

        let after_usage = self.analyze_usage(messages);
        let messages_removed = original_count - messages.messages.len();
        let characters_saved = before_usage
            .character_count
            .saturating_sub(after_usage.character_count);

        info!(
            "Context management: Completed reduction, removed {} messages, saved {} chars, usage now {:.1}%",
            messages_removed, characters_saved, after_usage.usage_percentage
        );

        Ok(ContextManagementResult {
            before_usage,
            after_usage,
            messages_removed,
            characters_saved,
            changes_made: messages_removed > 0,
            removed_by_priority,
        })
    }

    /// Get the maximum safe token count
    pub fn max_safe_tokens(&self) -> usize {
        (self.config.max_tokens as f32 * self.config.buffer_percentage) as usize
    }

    /// Estimate tokens for a single message
    pub fn estimate_message_tokens(&self, message: &Message) -> usize {
        let char_count = self.calculate_message_chars(message);
        (char_count as f32 / self.config.chars_per_token).ceil() as usize
    }

    /// Calculate content breakdown for analysis
    fn calculate_content_breakdown(&self, messages: &Messages) -> ContentBreakdown {
        let mut breakdown = ContentBreakdown::default();

        for message in &messages.messages {
            for content in &message.content {
                match content {
                    ContentBlock::Text { text } => {
                        breakdown.text_chars += text.len();
                    }
                    ContentBlock::ToolUse { input, .. } => {
                        breakdown.tool_use_chars += input.to_string().len() + 100; // Overhead
                        breakdown.tool_interactions += 1;
                    }
                    ContentBlock::ToolResult { content, .. } => {
                        breakdown.tool_result_chars += self.calculate_tool_result_chars(content);
                    }
                    ContentBlock::Thinking { content, .. } => {
                        breakdown.thinking_chars += content.len();
                    }
                    ContentBlock::ReasoningContent { reasoning } => {
                        breakdown.thinking_chars += reasoning.text().len();
                    }
                }
            }
        }

        breakdown
    }

    /// Calculate character count for a single message
    fn calculate_message_chars(&self, message: &Message) -> usize {
        message
            .content
            .iter()
            .map(|content| match content {
                ContentBlock::Text { text } => text.len(),
                ContentBlock::ToolUse { input, .. } => input.to_string().len() + 100,
                ContentBlock::ToolResult { content, .. } => {
                    self.calculate_tool_result_chars(content)
                }
                ContentBlock::Thinking { content, .. } => content.len(),
                ContentBlock::ReasoningContent { reasoning } => reasoning.text().len(),
            })
            .sum()
    }

    /// Calculate character count for tool result content
    #[allow(clippy::only_used_in_recursion)]
    fn calculate_tool_result_chars(&self, content: &crate::types::ToolResultContent) -> usize {
        match content {
            crate::types::ToolResultContent::Text { text } => text.len(),
            crate::types::ToolResultContent::Json { data } => data.to_string().len(),
            crate::types::ToolResultContent::Binary { data, .. } => data.len(),
            crate::types::ToolResultContent::Multiple { blocks } => blocks
                .iter()
                .map(|b| self.calculate_tool_result_chars(b))
                .sum(),
        }
    }

    /// Assign priority to a message based on its content and position
    fn assign_message_priority(
        &self,
        message: &Message,
        index: usize,
        total: usize,
    ) -> MessagePriority {
        // System messages are always critical
        if message.role == MessageRole::System {
            return MessagePriority::Critical;
        }

        // Recent messages (last 20%) get higher priority
        let recent_threshold = (total as f32 * 0.8) as usize;
        let is_recent = index >= recent_threshold;

        // Check if message contains tool interactions
        let has_tool_use = message
            .content
            .iter()
            .any(|c| matches!(c, ContentBlock::ToolUse { .. }));
        let has_tool_result = message
            .content
            .iter()
            .any(|c| matches!(c, ContentBlock::ToolResult { .. }));

        match (
            is_recent,
            has_tool_use || has_tool_result,
            message.role.clone(),
        ) {
            (_, _, MessageRole::System) => MessagePriority::Critical,
            (true, true, MessageRole::User) => MessagePriority::High,
            (true, true, MessageRole::Assistant) => MessagePriority::Medium,
            (true, false, _) => MessagePriority::Medium,
            (false, true, _) => MessagePriority::Medium,
            (false, false, _) => MessagePriority::Normal,
        }
    }

    /// Priority-based reduction strategy
    fn priority_based_reduction(
        &self,
        messages: &mut Messages,
        removed_by_priority: &mut HashMap<MessagePriority, usize>,
    ) -> Result<()> {
        let target_tokens = self.max_safe_tokens();

        // Calculate priorities for all messages
        let mut message_priorities: Vec<(usize, MessagePriority)> = messages
            .messages
            .iter()
            .enumerate()
            .map(|(i, msg)| {
                (
                    i,
                    self.assign_message_priority(msg, i, messages.messages.len()),
                )
            })
            .collect();

        // Sort by priority (lowest first) and index (oldest first within same priority)
        message_priorities.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

        // Remove messages starting from lowest priority
        let mut indices_to_remove = Vec::new();
        for (index, priority) in message_priorities {
            if self.analyze_usage(messages).estimated_tokens <= target_tokens {
                break;
            }

            // Don't remove if it would go below minimum
            if messages.messages.len() - indices_to_remove.len() <= self.config.min_messages {
                break;
            }

            indices_to_remove.push(index);
            *removed_by_priority.entry(priority).or_insert(0) += 1;
        }

        // Remove messages in reverse order to maintain indices
        indices_to_remove.sort_by(|a, b| b.cmp(a));
        for index in indices_to_remove {
            messages.messages.remove(index);
        }

        Ok(())
    }

    /// Simple sliding window reduction (removes from beginning)
    fn simple_sliding_window_reduction(&self, messages: &mut Messages) -> Result<()> {
        let target_tokens = self.max_safe_tokens();

        while self.analyze_usage(messages).estimated_tokens > target_tokens {
            if messages.messages.len() <= self.config.min_messages {
                break;
            }
            messages.messages.remove(0);
        }

        Ok(())
    }
}

impl Default for ContextManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MessageRole, ToolResultContent};
    use chrono::Utc;
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

    fn create_test_messages() -> Messages {
        Messages {
            messages: vec![
                create_test_message(MessageRole::System, "You are a helpful assistant."),
                create_test_message(MessageRole::User, "Hello, how are you?"),
                create_test_message(MessageRole::Assistant, "I'm doing well, thank you!"),
                create_test_message(MessageRole::User, "What's the weather like?"),
                create_test_message(
                    MessageRole::Assistant,
                    "I don't have access to real-time weather data.",
                ),
            ],
            system_prompt: None,
        }
    }

    #[test]
    fn test_context_manager_creation() {
        let manager = ContextManager::new();
        assert_eq!(manager.config.max_tokens, 200_000);
        assert_eq!(manager.config.buffer_percentage, 0.85);
    }

    #[test]
    fn test_context_manager_custom_config() {
        let config = ContextConfig {
            max_tokens: 100_000,
            buffer_percentage: 0.9,
            ..Default::default()
        };
        let manager = ContextManager::with_config(config);
        assert_eq!(manager.config.max_tokens, 100_000);
        assert_eq!(manager.config.buffer_percentage, 0.9);
    }

    #[test]
    fn test_analyze_usage() {
        let manager = ContextManager::new();
        let messages = create_test_messages();

        let usage = manager.analyze_usage(&messages);

        assert_eq!(usage.message_count, 5);
        assert!(usage.character_count > 0);
        assert!(usage.estimated_tokens > 0);
        assert!(usage.usage_percentage < 1.0); // Should be very low for test messages
        assert!(!usage.exceeds_safe_limit);
    }

    #[test]
    fn test_needs_management_false_for_small_context() {
        let manager = ContextManager::new();
        let messages = create_test_messages();

        assert!(!manager.needs_management(&messages));
    }

    #[test]
    fn test_manage_context_no_changes_needed() {
        let manager = ContextManager::new();
        let mut messages = create_test_messages();

        let result = manager.manage_context(&mut messages).unwrap();

        assert!(!result.changes_made);
        assert_eq!(result.messages_removed, 0);
        assert_eq!(result.characters_saved, 0);
    }

    #[test]
    fn test_max_safe_tokens() {
        let manager = ContextManager::new();
        let expected = (200_000 as f32 * 0.85) as usize;
        assert_eq!(manager.max_safe_tokens(), expected);
    }

    #[test]
    fn test_estimate_message_tokens() {
        let manager = ContextManager::new();
        let message = create_test_message(MessageRole::User, "Hello world");

        let tokens = manager.estimate_message_tokens(&message);

        // "Hello world" = 11 chars, with 4 chars/token = ~3 tokens
        assert!(tokens >= 2 && tokens <= 4);
    }

    #[test]
    fn test_content_breakdown() {
        let manager = ContextManager::new();
        let mut messages = Messages { messages: vec![], system_prompt: None };

        // Add message with tool use
        messages.messages.push(Message {
            id: Uuid::new_v4(),
            role: MessageRole::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: "Let me help you".to_string(),
                },
                ContentBlock::ToolUse {
                    id: "test".to_string(),
                    name: "calculator".to_string(),
                    input: serde_json::json!({"x": 1, "y": 2}),
                },
            ],
            metadata: HashMap::new(),
            timestamp: Utc::now(),
        });

        // Add message with tool result
        messages.messages.push(Message {
            id: Uuid::new_v4(),
            role: MessageRole::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "test".to_string(),
                content: ToolResultContent::Text {
                    text: "Result: 3".to_string(),
                },
                is_error: false,
            }],
            metadata: HashMap::new(),
            timestamp: Utc::now(),
        });

        let usage = manager.analyze_usage(&messages);

        assert!(usage.content_breakdown.text_chars > 0);
        assert!(usage.content_breakdown.tool_use_chars > 0);
        assert!(usage.content_breakdown.tool_result_chars > 0);
        assert_eq!(usage.content_breakdown.tool_interactions, 1);
    }

    #[test]
    fn test_message_priority_assignment() {
        let manager = ContextManager::new();

        // System message should be critical
        let system_msg = create_test_message(MessageRole::System, "System prompt");
        let priority = manager.assign_message_priority(&system_msg, 0, 5);
        assert_eq!(priority, MessagePriority::Critical);

        // Recent user message should be high priority
        let user_msg = create_test_message(MessageRole::User, "Recent question");
        let priority = manager.assign_message_priority(&user_msg, 4, 5); // Last message
        assert_eq!(priority, MessagePriority::Medium);

        // Old message should be normal priority
        let old_msg = create_test_message(MessageRole::Assistant, "Old response");
        let priority = manager.assign_message_priority(&old_msg, 1, 5); // Early message
        assert_eq!(priority, MessagePriority::Normal);
    }

    #[test]
    fn test_context_management_with_large_context() {
        let config = ContextConfig {
            max_tokens: 100, // Very small limit to trigger management
            buffer_percentage: 0.5,
            chars_per_token: 1.0, // 1 char = 1 token for easy testing
            ..Default::default()
        };
        let manager = ContextManager::with_config(config);

        // Create messages that exceed the limit
        let mut messages = Messages { messages: vec![], system_prompt: None };
        for i in 0..10 {
            messages.messages.push(create_test_message(
                MessageRole::User,
                &format!(
                    "This is a long message number {} that contains many characters",
                    i
                ),
            ));
        }

        assert!(manager.needs_management(&messages));

        let result = manager.manage_context(&mut messages).unwrap();

        assert!(result.changes_made);
        assert!(result.messages_removed > 0);
        assert!(result.characters_saved > 0);
        assert!(messages.messages.len() >= manager.config.min_messages);
    }
}
