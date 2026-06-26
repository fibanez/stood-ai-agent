//! Example 018: Task Evaluation Strategy - Travel Planning
//!
//! This example demonstrates the default task evaluation strategy where
//! the agent evaluates whether it has fully satisfied the user's request
//! and intent to determine if it should continue working.
//!
//! The task evaluation strategy allows agents to:
//! - Focus on user satisfaction and request completion
//! - Identify missing aspects that would better serve the user
//! - Continue working until the user's needs are fully addressed
//! - Provide comprehensive responses that match user intent
//!
//! This example shows how basic destination info is enhanced into a
//! comprehensive travel plan through task evaluation.

use std::sync::Arc;
use tokio::sync::Mutex;

use stood::agent::callbacks::{CallbackError, CallbackEvent, CallbackHandler, PrintingConfig};
use stood::{agent::Agent, tool};

#[tool]
/// Get basic tourist information for a destination
/// Provides essential travel info but may lack practical planning details
async fn get_destination_info(destination: String) -> Result<String, String> {
    let travel_info = format!(
        "🌍 Travel Information for {}\n\n\
        📍 OVERVIEW:\n\
        - Popular tourist destination with rich cultural heritage\n\
        - Best time to visit: April-October (mild weather)\n\
        - Language: Local language + English widely spoken\n\
        - Currency: Local currency (exchange rate varies)\n\n\
        🎯 TOP ATTRACTIONS:\n\
        - Historic city center and museums\n\
        - Local markets and shopping districts\n\
        - Natural landmarks and scenic viewpoints\n\
        - Cultural sites and entertainment venues\n\n\
        🍽️ FOOD & CULTURE:\n\
        - Traditional cuisine featuring local specialties\n\
        - Vibrant nightlife and cultural events\n\
        - Friendly locals and welcoming atmosphere\n\
        - Rich history and architectural heritage\n\n\
        📋 BASIC INFO:\n\
        - Generally safe for tourists\n\
        - Good public transportation available\n\
        - Various accommodation options\n\
        - Tourist information centers in city center\n\n\
        ⚠️ Note: This is basic destination information. For a complete travel plan, \
        you'll want specific restaurant recommendations, detailed itineraries, \
        practical tips, and insider knowledge.",
        destination
    );
    Ok(travel_info)
}

/// Enhanced callback handler that tracks evaluation events
#[derive(Debug)]
struct EvaluationTrackingHandler {
    last_evaluation_decision: Arc<Mutex<Option<bool>>>,
    last_evaluation_reasoning: Arc<Mutex<Option<String>>>,
}

impl EvaluationTrackingHandler {
    fn new() -> Self {
        Self {
            last_evaluation_decision: Arc::new(Mutex::new(None)),
            last_evaluation_reasoning: Arc::new(Mutex::new(None)),
        }
    }
}

#[async_trait::async_trait]
impl CallbackHandler for EvaluationTrackingHandler {
    async fn handle_event(&self, event: CallbackEvent) -> Result<(), CallbackError> {
        match event {
            CallbackEvent::EvaluationStart { strategy, .. } => {
                println!("🤔 Starting evaluation using {} strategy...", strategy);
            }
            CallbackEvent::EvaluationComplete {
                strategy,
                decision,
                reasoning,
                duration,
            } => {
                // Store evaluation results
                let mut last_decision = self.last_evaluation_decision.lock().await;
                let mut last_reasoning = self.last_evaluation_reasoning.lock().await;
                *last_decision = Some(decision);
                *last_reasoning = Some(reasoning.clone());

                // Display evaluation decision with reasoning
                let decision_text = if decision { "CONTINUE" } else { "STOP" };
                println!(
                    "🤔 Evaluation ({}) → {} (took {:.1}s)",
                    strategy,
                    decision_text,
                    duration.as_secs_f64()
                );

                if !reasoning.trim().is_empty() {
                    println!(
                        "   Reasoning: {}",
                        reasoning.chars().take(200).collect::<String>()
                    );
                }
            }
            _ => {} // Ignore other events
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Disable telemetry to avoid OTLP warnings in example
    std::env::set_var("OTEL_ENABLED", "false");

    println!("✈️ Task Evaluation Strategy Demo - Travel Planning");
    println!("====================================================");
    println!("This example shows how task evaluation guides an agent to create");
    println!("comprehensive travel plans using model knowledge.\n");

    // Create travel planning tool
    let travel_tools = vec![get_destination_info()];

    // Create enhanced callback handler for evaluation tracking
    let evaluation_handler = EvaluationTrackingHandler::new();

    // Create agent with task evaluation strategy (this is the DEFAULT behavior)
    let mut travel_agent = Agent::builder()
        .provider("bedrock")
        .model_str("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .system_prompt(
            "You are a knowledgeable travel advisor specializing in comprehensive trip planning. \
            Your role is to help travelers create detailed, practical travel plans.\n\n\
            TOOL USAGE WORKFLOW:\n\
            1. FIRST REQUEST: Use get_destination_info tool ONCE to gather basic information\n\
            2. EVALUATION RESPONSE: When evaluation identifies missing details, enhance your \
               response using your extensive travel knowledge - DO NOT call tools again\n\
            3. Use your knowledge to add practical details about:\n\
               - Specific attractions with visiting tips and timing\n\
               - Local transportation options and costs\n\
               - Neighborhood recommendations and where to stay\n\
               - Local cuisine, must-try dishes, and restaurant recommendations\n\
               - Cultural insights, customs, and etiquette\n\
               - Practical tips for weather, packing, and safety\n\
               - Budget estimates and money-saving tips\n\
               - Sample itineraries and time management\n\n\
            RESPONSE REQUIREMENTS:\n\
            - Provide complete, actionable travel advice\n\
            - Never ask if user wants more details or elaboration\n\
            - Be definitive and confident in recommendations\n\
            - NO phrases like 'Would you like...', 'Should I...', 'Do you want...'\n\
            - End with 'Your comprehensive travel plan is ready!' or similar definitive statement"
        )
        .tools(travel_tools)
        .with_callback_handler(evaluation_handler)
        .with_task_evaluation(
            "As a travel expert, evaluate the completeness of this travel plan:\n\n\
            TRAVEL PLANNING EVALUATION CRITERIA:\n\
            1. PRACTICAL INFORMATION (Required):\n\
               - Are specific attractions, restaurants, and activities mentioned?\n\
               - Is transportation information detailed (costs, routes, timing)?\n\
               - Are accommodation recommendations provided with area suggestions?\n\n\
            2. LOCAL INSIGHTS (Required):\n\
               - Are cultural customs and etiquette tips included?\n\
               - Is local cuisine properly described with specific dish recommendations?\n\
               - Are insider tips and local secrets shared?\n\n\
            3. PRACTICAL PLANNING (Required):\n\
               - Are budget estimates provided for different spending levels?\n\
               - Is weather information detailed with packing suggestions?\n\
               - Are sample itineraries or time management tips included?\n\n\
            4. COMPREHENSIVE COVERAGE (Required):\n\
               - Does this cover both must-see highlights AND hidden gems?\n\
               - Are different traveler types considered (budget, luxury, family)?\n\
               - Is safety information and practical advice included?\n\n\
            COMPLETION CRITERIA (STOP if ALL are met):\n\
            ✅ 1. SPECIFIC DETAILS: Contains specific restaurant names, neighborhoods, attractions\n\
            ✅ 2. PRACTICAL INFO: Includes transportation details, costs, and timing\n\
            ✅ 3. INSIDER KNOWLEDGE: Has local tips, cultural insights, and hidden gems\n\
            ✅ 4. ACTIONABLE PLAN: Provides sample itinerary with realistic scheduling\n\
            ✅ 5. COMPREHENSIVE: Covers dining, activities, transport, budget, safety\n\n\
            CONTINUATION DECISION:\n\
            - STOP: If response contains 4+ specific restaurant/attraction names, \
                   detailed transport info, cultural tips, AND a sample itinerary\n\
            - CONTINUE: Only if missing 2+ of the completion criteria above\n\n\
            MAXIMUM ONE ENHANCEMENT: This evaluation should only continue ONCE. \
            If this is the second evaluation cycle, automatically STOP regardless of completeness.\n\n\
            CRITICAL: If you decide to CONTINUE, start your response field with 'EVALUATION FEEDBACK:' \
            followed by specific guidance on what travel details are missing. \
            Emphasize using travel expertise and local knowledge, not calling tools again.\n\n\
            RESPONSE REQUIREMENTS:\n\
            - If CONTINUE: Provide specific guidance to enhance the travel plan\n\
            - If STOP: Leave response field empty\n\
            - Focus on practical, actionable travel advice\n\
            - Never suggest calling more tools - use travel expertise instead"
        )
        .with_printing_callbacks_config(PrintingConfig {
            show_reasoning: true,
            show_tools: true,
            show_performance: true,
            stream_output: false, // Disable streaming for cleaner output in examples
        })
        .build()
        .await?;

    println!("✅ Agent created with task evaluation strategy (DEFAULT)");
    println!("   - Model: Claude 3.5 Haiku");
    println!("   - Domain: Travel Planning");
    println!("   - Evaluation: Task completion criteria with practical travel requirements");
    println!("   - Max iterations: 25");
    println!("   - Tools: get_destination_info");
    println!("   - Enhancement: Uses model's travel knowledge for detailed planning");
    println!("   - Callbacks: Tool execution tracking and performance metrics");
    println!("   - Evaluation Display: Shows evaluation reasoning and feedback to user\n");

    // Test the task evaluation with a travel planning request
    println!("=== Comprehensive Travel Planning Task ===");
    let travel_task = "Help me plan a 5-day trip to Barcelona, Spain. I want to experience \
                      the best of the city including architecture, food, culture, and local life. \
                      I'm interested in both famous attractions and authentic local experiences. \
                      Please provide a comprehensive travel plan with practical details \
                      for planning and enjoying my visit.";

    println!("Task: {}\n", travel_task);
    println!("✈️ Agent will use task evaluation criteria...\n");

    // Add conversation tracking before execution
    println!("🔍 CONVERSATION TRACKING DEBUG:");
    println!(
        "   📝 Messages before execution: {}",
        travel_agent.conversation().message_count()
    );
    println!(
        "   🗨️  Initial conversation state: {}",
        if travel_agent.conversation().is_empty() {
            "EMPTY"
        } else {
            "HAS_MESSAGES"
        }
    );
    println!("");

    let result = travel_agent.execute(travel_task).await?;

    // Add conversation tracking after execution
    println!("\n🔍 CONVERSATION TRACKING RESULTS:");
    println!(
        "   📝 Messages after execution: {}",
        travel_agent.conversation().message_count()
    );
    println!("   🔄 Execution cycles: {}", result.execution.cycles);
    println!(
        "   🗨️  Final conversation state: {}",
        if travel_agent.conversation().is_empty() {
            "EMPTY"
        } else {
            "HAS_MESSAGES"
        }
    );

    // Display conversation messages to verify continuity and identify evaluation feedback
    println!("\n💬 CONVERSATION HISTORY VERIFICATION:");
    for (index, message) in travel_agent
        .conversation()
        .messages()
        .messages
        .iter()
        .enumerate()
    {
        let text = message
            .text()
            .map(|s| s.to_string())
            .unwrap_or_else(|| "[NO TEXT]".to_string());
        let preview = text.chars().take(100).collect::<String>();

        // Check if this looks like evaluation feedback
        let is_evaluation = match message.role {
            stood::types::MessageRole::User => {
                text.contains("EVALUATION FEEDBACK:")
                    || text.contains("Based on my travel evaluation")
                    || text.contains("Focus on completing any missing travel details")
            }
            _ => false,
        };

        let marker = if is_evaluation { "🔍 [EVAL]" } else { "" };

        println!(
            "   {}. {:?}: {}{} {}",
            index + 1,
            message.role,
            marker,
            preview,
            if text.len() > 100 { "..." } else { "" }
        );
    }

    println!("=== Comprehensive Travel Plan ===");
    println!("{}", result.response);

    // Show execution metrics
    println!("\n=== Execution Metrics ===");
    println!("Duration: {:?}", result.duration);
    println!("Execution cycles: {}", result.execution.cycles);
    println!("Model calls: {}", result.execution.model_calls);
    println!("Used tools: {}", result.used_tools);

    if result.used_tools {
        println!("Tools called: {}", result.tools_called.join(", "));
        println!(
            "Total tool calls: {}",
            result.tool_call_summary.total_attempts
        );
        println!(
            "Successful tool calls: {}",
            result.tool_call_summary.successful
        );

        if result.tool_call_summary.failed > 0 {
            println!("Failed tool calls: {}", result.tool_call_summary.failed);
        }
    }

    if let Some(tokens) = &result.execution.tokens {
        println!(
            "Token usage: input={}, output={}, total={}",
            tokens.input_tokens, tokens.output_tokens, tokens.total_tokens
        );
    }

    println!("\n=== Task Evaluation Benefits Demonstrated ===");
    println!("✅ User Intent Focus: Evaluation based on satisfying the traveler's needs");
    println!("✅ Practical Criteria: Specific requirements for actionable travel plans");
    println!("✅ Knowledge Enhancement: Model expertise supplements basic tool data");
    println!("✅ Clear Completion: Definitive criteria prevent endless iterations");
    println!("✅ Default Behavior: Multi-cycle execution enabled automatically");
    println!("✅ Quality Assurance: Ensures comprehensive, practical travel advice");

    println!("\n🎉 Task evaluation demonstration complete!");
    println!(
        "The agent used {} execution cycles with travel-focused evaluation",
        result.execution.cycles
    );

    Ok(())
}
