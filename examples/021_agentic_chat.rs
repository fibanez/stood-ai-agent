//! Example 021: Interactive Agentic Chat
//!
//! Simple interactive chat with AWS Bedrock Claude models and built-in tools.
//! The agent automatically selects and uses appropriate tools based on user requests.
//!
//! Usage:
//! ```bash
//! cargo run --example 021_agentic_chat
//! ```

use colored::*;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

use stood::agent::Agent;
use stood::{Result, StoodError};

/// Simple interactive chat application
struct AgenticChat {
    agent: Agent,
}

impl AgenticChat {
    async fn new() -> Result<Self> {
        // Default system prompt
        let system_prompt = "Answer questions directly and concisely. Do not explain your reasoning or why you are or are not using tools. Provide the most precise, immediate response to the query.";

        // Create agent with Claude Haiku 4.5 (default model) and built-in tools
        let agent = Agent::builder()
            .provider("bedrock")
            .model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
            .temperature(0.7)
            .max_tokens(4096)
            .system_prompt(system_prompt)
            .tools(Self::setup_tools().await?)
            .build()
            .await?;

        Ok(Self { agent })
    }

    async fn setup_tools() -> Result<Vec<Box<dyn stood::tools::Tool>>> {
        // Create essential built-in tools
        let tools: Vec<Box<dyn stood::tools::Tool>> = vec![
            Box::new(stood::tools::builtin::CalculatorTool::new()),
            Box::new(stood::tools::builtin::FileReadTool::new()),
            Box::new(stood::tools::builtin::FileWriteTool::new()),
            Box::new(stood::tools::builtin::FileListTool::new()),
            Box::new(stood::tools::builtin::CurrentTimeTool::new()),
        ];

        Ok(tools)
    }

    async fn run_interactive_chat(&mut self) -> Result<()> {
        println!("{}", "🤖 Interactive Chat with Claude".bright_cyan().bold());
        println!("{}", "Commands: exit/quit, help, clear".dimmed());
        println!();

        let mut rl = DefaultEditor::new().map_err(|e| {
            StoodError::configuration_error(format!("Failed to initialize readline: {}", e))
        })?;

        loop {
            let readline = rl.readline(&format!("{} ", "You:".bright_green().bold()));

            match readline {
                Ok(line) => {
                    let line = line.trim();

                    if line.is_empty() {
                        continue;
                    }

                    match line.to_lowercase().as_str() {
                        "exit" | "quit" => {
                            println!("{}", "Goodbye! 👋".bright_yellow());
                            break;
                        }
                        "help" => {
                            println!("{}", "Available commands:".bright_cyan());
                            println!("{}", "  exit/quit - Exit the program".dimmed());
                            println!("{}", "  help      - Show this help".dimmed());
                            println!("{}", "  clear     - Clear conversation history".dimmed());
                            continue;
                        }
                        "clear" => {
                            self.agent.clear_history();
                            println!("{}", "✅ Conversation history cleared.".bright_green());
                            continue;
                        }
                        _ => {}
                    }

                    rl.add_history_entry(line).ok();
                    self.process_message(line).await;
                    println!();
                }
                Err(ReadlineError::Interrupted) => {
                    println!("{}", "Goodbye! 👋".bright_yellow());
                    break;
                }
                Err(ReadlineError::Eof) => {
                    println!("{}", "Goodbye! 👋".bright_yellow());
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

    async fn process_message(&mut self, message: &str) {
        match self.agent.execute(message).await {
            Ok(result) => {
                if result.response.is_empty() {
                    println!("{} {}", "Warning:".bright_yellow().bold(), "Empty response received. This likely indicates AWS credentials are not configured properly.");
                    println!("Please ensure you have set up AWS credentials via:");
                    println!("  • AWS_PROFILE environment variable");
                    println!(
                        "  • AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables"
                    );
                    println!("  • AWS CLI credentials file");
                } else {
                    println!("{} {}", "Assistant:".bright_blue().bold(), result.response);
                }
            }
            Err(e) => {
                println!("{} {}", "Error:".bright_red().bold(), e);
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Check for AWS credentials
    if std::env::var("AWS_PROFILE").is_err() && std::env::var("AWS_ACCESS_KEY_ID").is_err() {
        eprintln!(
            "{}",
            "⚠️  AWS credentials required for Bedrock integration".bright_yellow()
        );
        eprintln!("{}", "Configure using:".dimmed());
        eprintln!("{}", "  • export AWS_PROFILE=your-profile".dimmed());
        eprintln!(
            "{}",
            "  • export AWS_ACCESS_KEY_ID=... && export AWS_SECRET_ACCESS_KEY=...".dimmed()
        );
        eprintln!();
    }

    let mut chat = AgenticChat::new().await?;
    chat.run_interactive_chat().await?;

    Ok(())
}
