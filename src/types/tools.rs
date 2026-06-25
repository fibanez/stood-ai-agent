//! Tool-related type definitions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Specification for a tool that can be called by the agent
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Name of the tool
    pub name: String,
    /// Description of what the tool does
    pub description: String,
    /// JSON schema for the tool's input parameters
    pub input_schema: serde_json::Value,
    /// Optional metadata about the tool
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ToolSpec {
    /// Create a new tool specification
    pub fn new<S: Into<String>>(name: S, description: S, input_schema: serde_json::Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the tool specification
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Result of executing a tool
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolExecutionResult {
    /// ID of the tool call
    pub call_id: String,
    /// Name of the tool that was executed
    pub tool_name: String,
    /// Result content
    pub content: super::content::ToolResultContent,
    /// Whether the execution was successful
    pub success: bool,
    /// Optional error message if execution failed
    pub error: Option<String>,
    /// Execution duration in milliseconds
    pub duration_ms: u64,
    /// Optional metadata about the execution
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ToolExecutionResult {
    /// Create a successful tool execution result
    pub fn success<S: Into<String>>(
        call_id: S,
        tool_name: S,
        content: super::content::ToolResultContent,
        duration_ms: u64,
    ) -> Self {
        Self {
            call_id: call_id.into(),
            tool_name: tool_name.into(),
            content,
            success: true,
            error: None,
            duration_ms,
            metadata: HashMap::new(),
        }
    }

    /// Create a failed tool execution result
    pub fn error<S: Into<String>>(call_id: S, tool_name: S, error: S, duration_ms: u64) -> Self {
        let error_msg = error.into();
        Self {
            call_id: call_id.into(),
            tool_name: tool_name.into(),
            content: super::content::ToolResultContent::text(format!("Error: {}", error_msg)),
            success: false,
            error: Some(error_msg),
            duration_ms,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the execution result
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Configuration for tools to be provided to the LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolConfig {
    /// List of available tools
    pub tools: Vec<Tool>,
    /// How the LLM should choose tools
    pub tool_choice: ToolChoice,
}

impl ToolConfig {
    /// Create a new tool configuration with auto tool choice
    pub fn new(tools: Vec<Tool>) -> Self {
        Self {
            tools,
            tool_choice: ToolChoice::Auto,
        }
    }

    /// Create tool configuration with specific tool choice
    pub fn with_choice(tools: Vec<Tool>, tool_choice: ToolChoice) -> Self {
        Self { tools, tool_choice }
    }
}

/// A tool definition for the LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// The tool specification
    #[serde(rename = "toolSpec")]
    pub tool_spec: ToolSpec,
}

impl Tool {
    /// Create a new tool from a specification
    pub fn new(tool_spec: ToolSpec) -> Self {
        Self { tool_spec }
    }
}

/// How the LLM should choose which tools to use
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ToolChoice {
    /// LLM can choose any available tool or no tool (default)
    Auto,
    /// LLM must use at least one available tool
    Any,
    /// LLM must use a specific named tool
    Tool { name: String },
    /// Prevent the LLM from using any tools
    ///
    /// When set, tools are excluded from the request entirely so the LLM
    /// responds without calling any tools regardless of what tools are registered.
    None,
}

/// A tool use request from the LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolUse {
    /// Unique identifier for this tool call
    pub tool_use_id: String,
    /// Name of the tool to execute
    pub name: String,
    /// Input parameters for the tool
    pub input: serde_json::Value,
}

impl ToolUse {
    /// Create a new tool use request
    pub fn new<S: Into<String>>(tool_use_id: S, name: S, input: serde_json::Value) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            name: name.into(),
            input,
        }
    }
}

/// Reasons why the LLM stopped generating
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Normal completion
    EndTurn,
    /// LLM wants to use tools
    ToolUse,
    /// Maximum token limit reached
    MaxTokens,
    /// Stop sequence encountered
    StopSequence,
    /// Content filtering triggered
    ContentFiltered,
}
