//! Example 019: Agent-Based Evaluation - Recipe Development
//!
//! This example demonstrates the agent-based evaluation strategy where
//! a separate chef evaluator assesses recipe completeness to determine
//! if the recipe developer should continue working.
//!
//! Key design: Tools provide basic ingredient data, but evaluation pushes
//! for cooking techniques, timing, and tips that come from culinary knowledge,
//! not tool calls. This prevents tool repetition issues.
//!
//! The agent-based strategy demonstrates:
//! - Independent culinary assessment by specialized chef evaluator
//! - Tools for data gathering vs knowledge for recipe enhancement
//! - Separation between ingredient lookup and cooking expertise
//! - Quality assurance through professional chef review

use stood::agent::callbacks::PrintingConfig;
use stood::{agent::Agent, tool};

#[tool]
/// Get basic nutritional information for an ingredient
async fn get_nutrition_info(ingredient: String) -> Result<String, String> {
    let nutrition = format!(
        "🥕 Nutritional Information: {}\n\n\
        BASIC NUTRITION (per 100g):\n\
        - Calories: ~45-85 kcal\n\
        - Protein: 1-3g\n\
        - Carbohydrates: 8-15g\n\
        - Fat: 0.1-0.5g\n\
        - Fiber: 2-4g\n\n\
        GENERAL PROPERTIES:\n\
        - Season: Available year-round\n\
        - Storage: Refrigerate for freshness\n\
        - Shelf life: 5-10 days\n\
        - Common uses: Various cooking methods\n\n\
        ⚠️ Note: Generic nutritional data - does not include cooking methods, \
        flavor combinations, preparation techniques, or recipe-specific guidance",
        ingredient
    );
    Ok(nutrition)
}

#[tool]
/// Get ingredient substitution options
async fn get_substitutions(ingredient: String) -> Result<String, String> {
    let substitutions = format!(
        "🔄 Substitution Options for: {}\n\n\
        COMMON SUBSTITUTES:\n\
        - Option A: Similar flavor profile ingredient\n\
        - Option B: Different texture but compatible taste\n\
        - Option C: Alternative with adjusted quantities\n\n\
        SUBSTITUTION RATIOS:\n\
        - 1:1 ratio for most basic substitutions\n\
        - Adjust quantities based on intensity\n\
        - Consider texture and cooking time differences\n\n\
        GENERAL NOTES:\n\
        - Flavor may vary slightly with substitutions\n\
        - Test small amounts before full recipe\n\
        - Some substitutes work better in specific dishes\n\n\
        ⚠️ Note: Basic substitution data - does not include cooking techniques, \
        timing adjustments, or professional chef tips for best results",
        ingredient
    );
    Ok(substitutions)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Disable telemetry to avoid OTLP warnings in example
    std::env::set_var("OTEL_ENABLED", "false");

    println!("👨‍🍳 Agent-Based Evaluation Demo - Recipe Development");
    println!("======================================================");
    println!("This example shows how a chef evaluator can assess recipe completeness");
    println!("while the main agent uses basic ingredient tools + culinary knowledge.\n");

    // Create ingredient data tools (unrelated to what chef evaluator will want)
    let ingredient_tools = vec![get_nutrition_info(), get_substitutions()];

    // Create the chef evaluator agent (specialized for culinary assessment)
    let chef_evaluator = Agent::builder()
        .provider("bedrock")
        .model_str("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .system_prompt(
            "You are a professional chef evaluating recipe completeness. \
            Assess if this recipe meets restaurant-quality standards for home cooks.\n\n\
            RECIPE COMPLETION CRITERIA (STOP if ALL are met):\n\
            ✅ 1. DETAILED STEPS: Step-by-step instructions with specific timing (e.g., '3-4 minutes')\n\
            ✅ 2. TEMPERATURES: Specific cooking temperatures (e.g., '375°F', 'medium-high heat')\n\
            ✅ 3. CHEF TECHNIQUES: Professional tips for texture, flavor, or presentation\n\
            ✅ 4. COMPLETE INGREDIENTS: All ingredients with measurements and preparation notes\n\
            ✅ 5. TROUBLESHOOTING: At least 2 common issues with solutions\n\n\
            EVALUATION DECISION:\n\
            - STOP: If recipe includes specific temperatures, detailed timing, \
                   professional chef tips, AND troubleshooting advice\n\
            - CONTINUE: Only if missing 2+ completion criteria above\n\n\
            MAXIMUM ONE IMPROVEMENT: Allow only ONE continuation cycle. \
            If this is the second evaluation, automatically STOP.\n\n\
            RESPONSE FORMAT:\n\
            If CONTINUE: 'The recipe needs: [specific missing elements]'\n\
            If STOP: 'Recipe is complete and ready for home cooks.'"
        )
        .build()
        .await?;

    // Create the main recipe developer agent with chef evaluation
    let mut recipe_agent = Agent::builder()
        .provider("bedrock")
        .model_str("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .system_prompt(
            "You are a recipe developer creating detailed cooking instructions. \
            Use available tools to gather ingredient data, then apply your culinary \
            knowledge to create comprehensive recipes with:\n\
            - Step-by-step cooking instructions with timing\n\
            - Temperature and technique details\n\
            - Chef tips for best results\n\
            - Flavor notes and variations\n\
            - Troubleshooting common issues\n\n\
            When chef evaluation indicates missing elements, enhance the recipe \
            using your cooking knowledge - do NOT repeat tool calls for the same ingredients.",
        )
        .tools(ingredient_tools)
        .with_agent_based_evaluation(chef_evaluator)
        .with_high_tool_limit(35) // Allow sufficient tool calls for thorough work
        .with_printing_callbacks_config(PrintingConfig {
            show_reasoning: true,
            show_tools: true,
            show_performance: true,
            stream_output: false, // Disable streaming for cleaner output in examples
        })
        .build()
        .await?;

    println!("✅ Agents created with chef-based evaluation");
    println!("   - Main Agent: Recipe developer");
    println!("   - Evaluator Agent: Professional chef reviewer");
    println!("   - Model: Claude 3.5 Haiku (both agents)");
    println!("   - High tool limit: 35 calls");
    println!("   - Tools: get_nutrition_info, get_substitutions (data only)");
    println!("   - Enhancement: Culinary knowledge for cooking techniques");
    println!("   - Evaluation: Independent chef assessment of recipe completeness");
    println!("   - Callbacks: Tool execution tracking and performance metrics\n");

    // Test the chef-based evaluation with a recipe development task
    println!("=== Recipe Development Task ===");
    let recipe_task = "Create a complete recipe for Chicken Parmesan that a home cook \
                      can follow successfully. The recipe should include detailed cooking \
                      instructions, timing, temperatures, and chef tips for achieving \
                      restaurant-quality results at home.";

    println!("Task: {}\n", recipe_task);
    println!("👨‍🍳 Recipe agent will gather ingredient data while chef evaluator ensures completeness...\n");

    let result = recipe_agent.execute(recipe_task).await?;

    println!("=== Final Recipe ===");
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

    println!("\n=== Chef-Based Evaluation Benefits Demonstrated ===");
    println!("✅ Independent Review: Chef evaluator provides culinary expertise assessment");
    println!("✅ Tool Separation: Data tools vs cooking knowledge prevents repetition");
    println!("✅ Multi-Perspective: Recipe developer gathers data, chef judges completeness");
    println!("✅ Quality Assurance: Professional chef standards for recipe quality");
    println!("✅ Knowledge Enhancement: Cooking expertise fills gaps beyond ingredient data");

    println!("\n🎉 Chef-based evaluation demonstration complete!");
    println!(
        "The recipe agent used {} execution cycles with chef oversight",
        result.execution.cycles
    );

    Ok(())
}
