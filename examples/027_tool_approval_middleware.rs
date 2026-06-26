//! Example 025: Tool Approval Middleware
//!
//! Interactive chat with tool execution approval via middleware.
//! Before any tool executes, the user is prompted to approve or deny.
//!
//! This demonstrates:
//! - ToolMiddleware for intercepting tool calls
//! - User confirmation prompts before tool execution
//! - Synthetic results for denied tools (keeps conversation valid)
//! - Cancellation with Ctrl+C during long-running tools
//! - Debug level selection at startup
//!
//! Usage:
//! ```bash
//! cargo run --example 025_tool_approval_middleware
//! ```

use async_trait::async_trait;
use colored::*;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use serde_json::{json, Value};
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use stood::agent::Agent;
use stood::tools::{
    AfterToolAction, Tool, ToolContext, ToolError, ToolMiddleware, ToolMiddlewareAction, ToolResult,
};
use stood::{Result, StoodError};

/// Shared state for cancellation handling
#[derive(Debug)]
struct CancellationState {
    /// Flag to signal cancellation to tools
    cancelled: AtomicBool,
    /// Timestamp of last Ctrl+C (for double-press detection)
    last_ctrl_c_ms: AtomicU64,
    /// Flag to signal program should exit
    should_exit: AtomicBool,
}

impl CancellationState {
    fn new() -> Self {
        Self {
            cancelled: AtomicBool::new(false),
            last_ctrl_c_ms: AtomicU64::new(0),
            should_exit: AtomicBool::new(false),
        }
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    fn reset(&self) {
        self.cancelled.store(false, Ordering::SeqCst);
    }

    fn should_exit(&self) -> bool {
        self.should_exit.load(Ordering::SeqCst)
    }

    /// Handle Ctrl+C press. Returns true if this is a double-press (exit signal).
    fn handle_ctrl_c(&self) -> bool {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let last_ms = self.last_ctrl_c_ms.swap(now_ms, Ordering::SeqCst);

        // Double-press within 2 seconds = exit
        if now_ms - last_ms < 2000 {
            self.should_exit.store(true, Ordering::SeqCst);
            return true;
        }

        // Single press = cancel current operation
        self.cancel();
        false
    }
}

/// A slow "typewriter" tool that prints text character by character
/// This gives time to test Ctrl+C cancellation
#[derive(Debug)]
struct SlowTyperTool {
    state: Arc<CancellationState>,
}

impl SlowTyperTool {
    fn new(state: Arc<CancellationState>) -> Self {
        Self { state }
    }
}

#[async_trait]
impl Tool for SlowTyperTool {
    fn name(&self) -> &str {
        "slow_typer"
    }

    fn description(&self) -> &str {
        "Types out a message slowly, character by character. Use this when the user asks for a slow or dramatic message. Takes a 'message' parameter with the text to type."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to type out slowly"
                }
            },
            "required": ["message"]
        })
    }

    async fn execute(
        &self,
        params: Option<Value>,
        _agent_context: Option<&stood::agent::AgentContext>,
    ) -> std::result::Result<ToolResult, ToolError> {
        let params = params.unwrap_or(Value::Null);
        let message = params
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("Hello, world!");

        println!();
        println!(
            "{} {}",
            "[TYPING]".bright_magenta(),
            "(Press Ctrl+C to cancel)".dimmed()
        );
        print!("  ");
        io::stdout().flush().ok();

        let mut typed = String::new();
        for ch in message.chars() {
            // Check for cancellation
            if self.state.is_cancelled() {
                println!();
                println!(
                    "{} Typing cancelled after {} characters",
                    "[CANCELLED]".bright_red().bold(),
                    typed.len()
                );
                return Ok(ToolResult::success(json!({
                    "status": "cancelled",
                    "typed_so_far": typed,
                    "remaining": message.len() - typed.len()
                })));
            }

            print!("{}", ch);
            io::stdout().flush().ok();
            typed.push(ch);

            // Delay between characters (80ms = ~12 chars/second, slower for better cancellation window)
            tokio::time::sleep(Duration::from_millis(80)).await;
        }
        println!();

        Ok(ToolResult::success(json!({
            "status": "completed",
            "message": message,
            "characters_typed": message.len()
        })))
    }
}

/// Middleware that prompts the user for approval before executing any tool
struct ToolApprovalMiddleware {
    /// Tools that are always allowed without prompting
    auto_approved: Vec<String>,
}

impl std::fmt::Debug for ToolApprovalMiddleware {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolApprovalMiddleware")
            .field("auto_approved", &self.auto_approved)
            .finish()
    }
}

impl ToolApprovalMiddleware {
    fn new() -> Self {
        Self {
            auto_approved: vec![],
        }
    }

    /// Prompt user for tool approval
    fn prompt_user(&self, tool_name: &str, params: &Value) -> bool {
        // Show compact parameter summary
        let param_summary = if let Some(obj) = params.as_object() {
            let parts: Vec<String> = obj
                .iter()
                .map(|(k, v)| {
                    let val = match v {
                        Value::String(s) if s.len() > 30 => format!("{}...", &s[..30]),
                        Value::String(s) => s.clone(),
                        _ => v.to_string(),
                    };
                    format!("{}={}", k, val)
                })
                .collect();
            if parts.is_empty() {
                String::new()
            } else {
                format!(" ({})", parts.join(", "))
            }
        } else {
            String::new()
        };

        print!(
            "Execute {}{}? [y/N]: ",
            tool_name.bright_cyan(),
            param_summary.dimmed()
        );
        io::stdout().flush().ok();

        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(_) => {
                let input = input.trim().to_lowercase();
                input == "y" || input == "yes"
            }
            Err(_) => false,
        }
    }
}

#[async_trait]
impl ToolMiddleware for ToolApprovalMiddleware {
    async fn before_tool(
        &self,
        tool_name: &str,
        params: &Value,
        ctx: &ToolContext,
    ) -> ToolMiddlewareAction {
        // Check if this tool is auto-approved
        if self.auto_approved.contains(&tool_name.to_string()) {
            println!("Executing {} (auto)", tool_name.bright_cyan());
            return ToolMiddlewareAction::Continue;
        }

        tracing::debug!(
            "Tool approval check for {} (agent: {}, tools this turn: {})",
            tool_name,
            ctx.agent_id,
            ctx.tool_count_this_turn
        );

        // Prompt user for approval
        if self.prompt_user(tool_name, params) {
            println!("Executing {}", tool_name.bright_cyan());
            ToolMiddlewareAction::Continue
        } else {
            println!("Cancelled {}", tool_name.bright_red());
            ToolMiddlewareAction::Abort {
                reason: format!("User denied permission to execute tool: {}", tool_name),
                synthetic_result: Some(ToolResult::error(format!(
                    "Permission denied: User declined to allow execution of '{}'. \
                     Please ask the user if they want to proceed differently.",
                    tool_name
                ))),
            }
        }
    }

    async fn after_tool(
        &self,
        tool_name: &str,
        result: &ToolResult,
        ctx: &ToolContext,
    ) -> AfterToolAction {
        if result.success {
            println!(
                "Completed {} ({}ms)",
                tool_name.bright_green(),
                ctx.elapsed_ms()
            );
        } else {
            println!(
                "Failed {} ({}ms)",
                tool_name.bright_red(),
                ctx.elapsed_ms()
            );
        }
        AfterToolAction::PassThrough
    }

    fn name(&self) -> &str {
        "tool_approval"
    }
}

/// Debug level for the application
#[derive(Debug, Clone, Copy, PartialEq)]
enum DebugLevel {
    None,
    Info,
    Debug,
    Trace,
}

impl DebugLevel {
    fn to_filter(&self) -> &'static str {
        match self {
            DebugLevel::None => "off",      // Suppress all logs
            DebugLevel::Info => "stood=info,warn",
            DebugLevel::Debug => "stood=debug",
            DebugLevel::Trace => "stood=trace",
        }
    }
}

/// Prompt user for debug level
fn prompt_debug_level() -> DebugLevel {
    println!("{}", "Select debug level:".bright_cyan());
    println!("  {} - No debug output (default)", "0".bright_white());
    println!("  {} - Info level", "1".bright_white());
    println!("  {} - Debug level", "2".bright_white());
    println!("  {} - Trace level (verbose)", "3".bright_white());
    println!();
    print!("{} ", "Enter choice [0-3, default=0]:".bright_yellow());
    io::stdout().flush().ok();

    let mut input = String::new();
    match io::stdin().read_line(&mut input) {
        Ok(_) => match input.trim() {
            "1" => DebugLevel::Info,
            "2" => DebugLevel::Debug,
            "3" => DebugLevel::Trace,
            _ => DebugLevel::None,
        },
        Err(_) => DebugLevel::None,
    }
}

/// Interactive chat application with tool approval
struct ApprovalChat {
    agent: Agent,
    state: Arc<CancellationState>,
}

impl ApprovalChat {
    async fn new(state: Arc<CancellationState>) -> Result<Self> {
        let system_prompt = r#"You are a helpful assistant with access to tools.
When a tool execution is denied by the user, acknowledge it gracefully and offer alternatives.
Be concise in your responses.

You have access to a special 'slow_typer' tool that types messages character by character.
Use it when the user asks for something to be typed slowly or dramatically."#;

        // Create the approval middleware
        let approval_middleware = Arc::new(ToolApprovalMiddleware::new());

        // Create agent with tools and middleware
        let agent = Agent::builder()
            .provider("bedrock")
            .model_str("us.anthropic.claude-haiku-4-5-20251001-v1:0")
            .temperature(0.7)
            .max_tokens(4096)
            .system_prompt(system_prompt)
            .tools(Self::setup_tools(state.clone())?)
            .with_middleware(approval_middleware)
            .build()
            .await?;

        Ok(Self { agent, state })
    }

    fn setup_tools(state: Arc<CancellationState>) -> Result<Vec<Box<dyn Tool>>> {
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(stood::tools::builtin::CalculatorTool::new()),
            Box::new(stood::tools::builtin::CurrentTimeTool::new()),
            Box::new(SlowTyperTool::new(state)),
        ];

        Ok(tools)
    }

    async fn run(&mut self) -> Result<()> {
        println!();
        println!("{}", "Tool Approval Middleware Demo".bright_cyan().bold());
        println!("{}", "=".repeat(50).bright_cyan());
        println!();
        println!(
            "{}",
            "This chat requires your approval before executing any tool.".dimmed()
        );
        println!();
        println!("{}", "Cancellation:".bright_red().bold());
        println!(
            "{}",
            "  Press Ctrl+C once  = Cancel current operation".dimmed()
        );
        println!(
            "{}",
            "  Press Ctrl+C twice = Exit program".dimmed()
        );
        println!();
        println!("{}", "Try these prompts:".bright_cyan());
        println!("{}", "  - What's 15 * 23 + 7?".dimmed());
        println!("{}", "  - What time is it?".dimmed());
        println!(
            "{}",
            "  - Type 'Hello World' slowly (then Ctrl+C to cancel)".dimmed()
        );
        println!();
        println!(
            "{}",
            "Commands: exit/quit, help, clear".bright_yellow()
        );
        println!();

        let mut rl = DefaultEditor::new().map_err(|e| {
            StoodError::configuration_error(format!("Failed to initialize readline: {}", e))
        })?;

        loop {
            // Check if we should exit (double Ctrl+C)
            if self.state.should_exit() {
                println!();
                println!("{}", "Exiting...".bright_yellow());
                break;
            }

            let readline = rl.readline(&format!("{} ", "You:".bright_green().bold()));

            match readline {
                Ok(line) => {
                    let line = line.trim();

                    if line.is_empty() {
                        continue;
                    }

                    match line.to_lowercase().as_str() {
                        "exit" | "quit" => {
                            println!("{}", "Goodbye!".bright_yellow());
                            break;
                        }
                        "help" => {
                            self.show_help();
                            continue;
                        }
                        "clear" => {
                            self.agent.clear_history();
                            println!("{}", "Conversation history cleared.".bright_green());
                            continue;
                        }
                        _ => {}
                    }

                    rl.add_history_entry(line).ok();
                    self.process_message(line).await;
                    println!();

                    // Check for exit after processing
                    if self.state.should_exit() {
                        println!();
                        println!("{}", "Exiting...".bright_yellow());
                        break;
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    // Ctrl+C during input prompt
                    if self.state.handle_ctrl_c() {
                        println!();
                        println!("{}", "Exiting...".bright_yellow());
                        break;
                    }
                    println!();
                    println!(
                        "{}",
                        "(Press Ctrl+C again within 2 seconds to exit)".dimmed()
                    );
                    continue;
                }
                Err(ReadlineError::Eof) => {
                    println!("{}", "Goodbye!".bright_yellow());
                    break;
                }
                Err(err) => {
                    println!("{} {}", "Error:".bright_red().bold(), err);
                    break;
                }
            }
        }

        Ok(())
    }

    fn show_help(&self) {
        println!("{}", "Commands:".bright_cyan());
        println!("{}", "  exit/quit - Exit the program".dimmed());
        println!("{}", "  help      - Show this help".dimmed());
        println!("{}", "  clear     - Clear conversation history".dimmed());
        println!();
        println!("{}", "Tool Approval:".bright_cyan());
        println!(
            "{}",
            "  When a tool is about to execute, you'll be prompted.".dimmed()
        );
        println!(
            "{}",
            "  Type 'y' or 'yes' to approve, anything else to deny.".dimmed()
        );
        println!();
        println!("{}", "Cancellation:".bright_cyan());
        println!(
            "{}",
            "  Press Ctrl+C once during tool execution to cancel.".dimmed()
        );
        println!(
            "{}",
            "  Press Ctrl+C twice quickly to exit the program.".dimmed()
        );
        println!();
        println!("{}", "Try these prompts:".bright_cyan());
        println!("{}", "  - Calculate 2^10".dimmed());
        println!("{}", "  - What's the current time?".dimmed());
        println!(
            "{}",
            "  - Type 'The quick brown fox jumps over the lazy dog' slowly".dimmed()
        );
    }

    async fn process_message(&mut self, message: &str) {
        println!();

        // Reset cancellation flag before processing
        self.state.reset();

        let start = Instant::now();

        // Execute the agent
        let result = self.agent.execute(message).await;

        let elapsed = start.elapsed();

        // Reset cancellation state
        self.state.reset();

        match result {
            Ok(result) => {
                if result.response.is_empty() {
                    println!(
                        "{} {}",
                        "Warning:".bright_yellow().bold(),
                        "Empty response received"
                    );
                } else {
                    println!(
                        "{} {}",
                        "Assistant:".bright_blue().bold(),
                        result.response
                    );
                }

                if result.used_tools {
                    println!();
                    println!(
                        "{} Tools: {:?}, Time: {:.1}s",
                        "[INFO]".dimmed(),
                        result.tools_called,
                        elapsed.as_secs_f64()
                    );
                }
            }
            Err(e) => {
                let error_str = e.to_string();
                if error_str.contains("cancelled") || error_str.contains("Cancelled") {
                    println!(
                        "{} Operation cancelled",
                        "[CANCELLED]".bright_red().bold()
                    );
                } else {
                    println!("{} {}", "Error:".bright_red().bold(), e);
                }
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("{}", "Tool Approval Middleware Demo".bright_cyan().bold());
    println!("{}", "=".repeat(50).bright_cyan());
    println!();

    // Prompt for debug level
    let debug_level = prompt_debug_level();
    println!();

    // Initialize tracing based on selected level
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(debug_level.to_filter().parse().unwrap()),
        )
        .init();

    // Check for AWS credentials
    if std::env::var("AWS_PROFILE").is_err() && std::env::var("AWS_ACCESS_KEY_ID").is_err() {
        eprintln!(
            "{}",
            "Warning: AWS credentials may be required for Bedrock".bright_yellow()
        );
        eprintln!("{}", "Configure using:".dimmed());
        eprintln!("{}", "  export AWS_PROFILE=your-profile".dimmed());
        eprintln!();
    }

    // Create shared cancellation state
    let state = Arc::new(CancellationState::new());

    // Set up Ctrl+C handler
    let handler_state = state.clone();
    ctrlc::set_handler(move || {
        if handler_state.handle_ctrl_c() {
            // Double press - exit immediately
            println!();
            println!("{}", "Force exit (double Ctrl+C)".bright_red());
            std::process::exit(0);
        } else {
            // Single press - just cancel
            println!();
            println!(
                "{}",
                "(Ctrl+C: cancelling... press again to exit)".bright_yellow()
            );
        }
    })
    .expect("Error setting Ctrl+C handler");

    let mut chat = ApprovalChat::new(state).await?;
    chat.run().await?;

    Ok(())
}
