//! Example 017: Multi-Round Parallel vs Sequential Execution
//!
//! This example demonstrates the power of parallel execution at scale:
//! - Parallel Agent: max_parallel_tools = 8 (up to 8 tools run concurrently)
//! - Sequential Agent: max_parallel_tools = 1 (tools run one at a time)
//!
//! Executes multiple rounds of various task types to show how parallelism
//! provides dramatic performance improvements as volume increases.
//! Shows real-time feedback including model interactions and progress.
//!
//! Usage:
//! ```bash
//! cargo run --example 017_parallel_execution
//! ```

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tokio::time::sleep;

use stood::agent::callbacks::{CallbackError, CallbackEvent, CallbackHandler};
use stood::agent::Agent;
use stood::tool;

/// Simple task simulation tool
#[tool]
/// Execute a task with simulated work
async fn task_executor(task_name: String, work_duration_ms: u64) -> Result<String, String> {
    // Simulate work with sleep
    sleep(Duration::from_millis(work_duration_ms)).await;

    Ok(format!(
        "Task '{}' completed in {}ms",
        task_name, work_duration_ms
    ))
}

/// Track task execution timing
#[derive(Debug, Clone)]
struct TaskTiming {
    task_type: String,
    start_time: Option<Instant>,
    duration: Option<Duration>,
}

/// Handler that tracks multi-round execution and displays real-time feedback
#[derive(Debug)]
struct MultiRoundHandler {
    tasks: Arc<Mutex<Vec<TaskTiming>>>,
    start_index: Arc<Mutex<usize>>,
    complete_index: Arc<Mutex<usize>>,
    mode: String,
    rounds_per_task: usize,
    completed_by_type: Arc<Mutex<std::collections::HashMap<String, Vec<Duration>>>>,
    model_start_time: Arc<Mutex<Option<Instant>>>,
}

impl MultiRoundHandler {
    fn new(tasks: Arc<Mutex<Vec<TaskTiming>>>, mode: &str, rounds_per_task: usize) -> Self {
        Self {
            tasks,
            start_index: Arc::new(Mutex::new(0)),
            complete_index: Arc::new(Mutex::new(0)),
            mode: mode.to_string(),
            rounds_per_task,
            completed_by_type: Arc::new(Mutex::new(std::collections::HashMap::new())),
            model_start_time: Arc::new(Mutex::new(None)),
        }
    }
}

#[async_trait::async_trait]
impl CallbackHandler for MultiRoundHandler {
    async fn handle_event(&self, event: CallbackEvent) -> Result<(), CallbackError> {
        match event {
            CallbackEvent::ModelStart { .. } => {
                let mut model_start_time = self.model_start_time.lock().await;
                *model_start_time = Some(Instant::now());
                println!("  {} 🤖 Sending prompt to model (32 tasks)...", self.mode);
            }
            CallbackEvent::ModelComplete { .. } => {
                let model_start_time = self.model_start_time.lock().await;
                if let Some(start_time) = *model_start_time {
                    let model_duration = start_time.elapsed();
                    println!(
                        "  {} ✅ Model response received ({:.1}s) - Starting tool execution...",
                        self.mode,
                        model_duration.as_secs_f64()
                    );
                }
            }
            CallbackEvent::ToolStart { tool_name, .. } => {
                if tool_name == "task_executor" {
                    let mut start_index = self.start_index.lock().await;
                    let mut tasks = self.tasks.lock().await;

                    if *start_index < tasks.len() {
                        let task = &mut tasks[*start_index];
                        task.start_time = Some(Instant::now());

                        // Show progress every 8 tasks (1 complete round) for parallel, every 1 for sequential
                        if self.mode == "PARALLEL" && (*start_index + 1) % 8 == 0 {
                            let round = (*start_index + 1) / 8;
                            println!(
                                "  {} ⚡ Round {} tools started ({}/32 tasks)",
                                self.mode,
                                round,
                                *start_index + 1
                            );
                        } else if self.mode == "SEQUENTIAL" {
                            println!(
                                "  {} ⚡ Tool {} started: {}",
                                self.mode,
                                *start_index + 1,
                                task.task_type
                            );
                        }

                        *start_index += 1;
                    }
                }
            }
            CallbackEvent::ToolComplete {
                tool_name,
                duration,
                error,
                ..
            } => {
                if tool_name == "task_executor" && error.is_none() {
                    let mut complete_index = self.complete_index.lock().await;
                    let mut tasks = self.tasks.lock().await;

                    if *complete_index < tasks.len() {
                        let task = &mut tasks[*complete_index];
                        task.duration = Some(duration);

                        // Track completion by task type
                        let mut completed_by_type = self.completed_by_type.lock().await;
                        let durations = completed_by_type
                            .entry(task.task_type.clone())
                            .or_insert_with(Vec::new);
                        durations.push(duration);

                        // Show progress every 8 completions for parallel, every 1 for sequential
                        if self.mode == "PARALLEL" && (*complete_index + 1) % 8 == 0 {
                            let round = (*complete_index + 1) / 8;
                            println!(
                                "  {} 🏁 Round {} complete ({}/32 tasks)",
                                self.mode,
                                round,
                                *complete_index + 1
                            );
                        } else if self.mode == "SEQUENTIAL" {
                            println!(
                                "  {} ✅ Tool {} complete: {} ({:.1}s)",
                                self.mode,
                                *complete_index + 1,
                                task.task_type,
                                duration.as_secs_f64()
                            );
                        }

                        // Check if all rounds for this task type are complete
                        if durations.len() == self.rounds_per_task {
                            let total_duration: Duration = durations.iter().sum();
                            let avg_duration = total_duration / self.rounds_per_task as u32;

                            println!(
                                "  {} ✅ {} - {} rounds complete | Avg: {:.1}s | Total: {:.1}s",
                                self.mode,
                                task.task_type,
                                self.rounds_per_task,
                                avg_duration.as_secs_f64(),
                                total_duration.as_secs_f64()
                            );
                        }

                        *complete_index += 1;
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// Create multi-round task list for tracking
fn create_multi_round_task_list(rounds: usize) -> Arc<Mutex<Vec<TaskTiming>>> {
    let task_types = [
        "File_Processing",
        "Database_Query",
        "Image_Resize",
        "Data_Analysis",
        "Email_Send",
        "Report_Generation",
        "Cache_Update",
        "Backup_Creation",
    ];

    let mut tasks = Vec::new();

    // Create multiple rounds of each task type
    for _round in 1..=rounds {
        for &task_type in &task_types {
            tasks.push(TaskTiming {
                task_type: task_type.to_string(),
                start_time: None,
                duration: None,
            });
        }
    }

    Arc::new(Mutex::new(tasks))
}

/// Setup agent for multi-round execution
async fn setup_multi_round_agent(
    mode: &str,
    tasks: Arc<Mutex<Vec<TaskTiming>>>,
    max_parallel: usize,
    rounds: usize,
) -> Result<Agent, Box<dyn std::error::Error>> {
    let timing_handler = MultiRoundHandler::new(tasks, mode, rounds);

    let agent = Agent::builder()
        .provider("bedrock")
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .temperature(0.0)
        .max_tokens(1500)  // Increased for larger prompts
        .tools(vec![task_executor()])
        .with_callback_handler(timing_handler)
        .max_parallel_tools(max_parallel)  // Controls parallel execution!
        .with_streaming(false)
        .with_timeout(Duration::from_secs(600))  // Increased timeout
        .system_prompt("You are a task executor. Execute all requested tasks using the task_executor tool. Process ALL tool calls in a single response.")
        .build()
        .await?;

    Ok(agent)
}

/// Generate multi-round task prompt
fn create_multi_round_prompt(rounds: usize) -> String {
    let task_configs = [
        ("File_Processing", 400),
        ("Database_Query", 600),
        ("Image_Resize", 300),
        ("Data_Analysis", 800),
        ("Email_Send", 450),
        ("Report_Generation", 550),
        ("Cache_Update", 350),
        ("Backup_Creation", 700),
    ];

    let total_tasks = task_configs.len() * rounds;

    let mut prompt = format!("Execute {} tasks across {} rounds. Please call ALL task_executor tools at once since they can run independently:\n\n", total_tasks, rounds);

    let mut task_number = 1;
    for round in 1..=rounds {
        prompt.push_str(&format!("=== ROUND {} ===\n", round));
        for (task_type, duration) in &task_configs {
            prompt.push_str(&format!(
                "{}. Execute task_executor with task_name='{}' and work_duration_ms={}\n",
                task_number, task_type, duration
            ));
            task_number += 1;
        }
        prompt.push_str("\n");
    }

    prompt.push_str(&format!("\nIMPORTANT: Please make ALL {} tool calls in your SINGLE response since these tasks are independent and can run concurrently.", total_tasks));
    prompt
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Disable all logging for clean output
    tracing_subscriber::fmt()
        .with_env_filter("stood=off")
        .with_target(false)
        .init();

    println!("🚀 Multi-Round Parallel vs Sequential Execution Demo");
    println!("═══════════════════════════════════════════════════════");

    // Configuration
    let rounds = 4;
    let max_parallel = 8;
    let task_types = 8;
    let total_tasks = rounds * task_types;

    println!(
        "Executing {} tasks ({} rounds × {} task types):",
        total_tasks, rounds, task_types
    );
    println!(
        "- Parallel Agent:   max_parallel_tools = {} (up to {} concurrent)",
        max_parallel, max_parallel
    );
    println!("- Sequential Agent: max_parallel_tools = 1 (one at a time)");
    println!("Both agents get identical prompts - only max_parallel_tools differs!");
    println!("");
    println!("📊 Real-time progress: model calls, round completion, and task type summaries");
    println!("──────────────────────────────────────────────────────────────────────────");

    // Create task tracking
    let parallel_tasks = create_multi_round_task_list(rounds);
    let sequential_tasks = create_multi_round_task_list(rounds);

    // Create agents
    let mut parallel_agent =
        setup_multi_round_agent("PARALLEL", parallel_tasks.clone(), max_parallel, rounds).await?;
    let mut sequential_agent =
        setup_multi_round_agent("SEQUENTIAL", sequential_tasks.clone(), 1, rounds).await?;

    let task_prompt = create_multi_round_prompt(rounds);

    // 1. Run Parallel Execution First
    println!("");
    println!(
        "⚡ PARALLEL EXECUTION (max_parallel_tools = {})",
        max_parallel
    );
    println!("────────────────────────────────────────────");

    let parallel_start = Instant::now();
    let parallel_result = parallel_agent.execute(task_prompt.clone()).await;
    let parallel_duration = parallel_start.elapsed();

    match parallel_result {
        Ok(_) => {
            println!(
                "✅ PARALLEL execution completed in {:.2}s",
                parallel_duration.as_secs_f64()
            );
        }
        Err(e) => {
            println!(
                "❌ PARALLEL execution failed in {:.2}s: {}",
                parallel_duration.as_secs_f64(),
                e
            );
        }
    }

    println!("");
    sleep(Duration::from_millis(3000)).await;

    // 2. Run Sequential Execution Second
    println!("🔄 SEQUENTIAL EXECUTION (max_parallel_tools = 1)");
    println!("────────────────────────────────────────────");

    let sequential_start = Instant::now();
    let sequential_result = sequential_agent.execute(task_prompt.clone()).await;
    let sequential_duration = sequential_start.elapsed();

    match sequential_result {
        Ok(_) => {
            println!(
                "✅ SEQUENTIAL execution completed in {:.2}s",
                sequential_duration.as_secs_f64()
            );
        }
        Err(e) => {
            println!(
                "❌ SEQUENTIAL execution failed in {:.2}s: {}",
                sequential_duration.as_secs_f64(),
                e
            );
        }
    }

    // Final summary
    println!("");
    println!("📊 MULTI-ROUND EXECUTION SUMMARY");
    println!("═════════════════════════════════");
    println!(
        "Total tasks executed: {} ({} rounds × {} task types)",
        total_tasks, rounds, task_types
    );
    println!(
        "Sequential execution: {:.2}s",
        sequential_duration.as_secs_f64()
    );
    println!(
        "Parallel execution:   {:.2}s",
        parallel_duration.as_secs_f64()
    );
    println!("");

    if parallel_duration < sequential_duration {
        let speedup = sequential_duration.as_secs_f64() / parallel_duration.as_secs_f64();
        let saved = sequential_duration - parallel_duration;
        println!("🚀 Parallel speedup:  {:.2}x faster", speedup);
        println!(
            "⏱️  Time saved:       {:.2}s ({:.1}%)",
            saved.as_secs_f64(),
            (saved.as_secs_f64() / sequential_duration.as_secs_f64()) * 100.0
        );
        println!(
            "📈 Throughput improvement: {:.1} tasks/second → {:.1} tasks/second",
            total_tasks as f64 / sequential_duration.as_secs_f64(),
            total_tasks as f64 / parallel_duration.as_secs_f64()
        );
    } else {
        println!("⚠️  No speedup observed (parallel overhead exceeded benefits)");
    }

    println!("");
    println!(
        "🎉 Multi-round demo complete! Higher parallelism shows dramatic improvements at scale."
    );

    Ok(())
}
