//! Example 020: Multi-Perspective Evaluation
//!
//! This example demonstrates the multi-perspective evaluation strategy where
//! the agent evaluates its work from multiple different viewpoints with
//! weighted scoring to determine if it should continue working.
//!
//! The multi-perspective strategy enables:
//! - Multiple evaluation criteria with different weights
//! - Balanced assessment from various stakeholder perspectives
//! - Weighted scoring system for comprehensive evaluation
//! - Holistic view of work quality and completeness
//! - Balanced decision-making considering multiple viewpoints

use stood::agent::callbacks::PrintingConfig;
use stood::agent::evaluation::PerspectiveConfig;
use stood::{agent::Agent, tool};

// Use wee_alloc as the global allocator for smaller binary size
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[tool]
/// Conduct market research and analysis
async fn market_research(topic: String, scope: Option<String>) -> Result<String, String> {
    let research_scope = scope.unwrap_or_else(|| "comprehensive".to_string());
    let market_data = format!(
        "Market Research: {} (scope: {})\n\n\
        📊 MARKET SIZE & GROWTH:\n\
        - Total Addressable Market (TAM): $12.5B\n\
        - Serviceable Addressable Market (SAM): $3.2B\n\
        - Annual Growth Rate: 15.3%\n\
        - Market Maturity: Growth phase\n\n\
        🎯 TARGET SEGMENTS:\n\
        - Enterprise (65%): High value, long sales cycles\n\
        - Mid-market (25%): Balanced approach, moderate complexity\n\
        - SMB (10%): Volume-based, price-sensitive\n\n\
        🏆 COMPETITIVE LANDSCAPE:\n\
        - Market Leaders: 3 dominant players (45% market share)\n\
        - Emerging Players: 8 growing competitors (30% share)\n\
        - Niche Solutions: 15+ specialized providers (25% share)\n\n\
        💡 KEY OPPORTUNITIES:\n\
        - Underserved segments in mid-market\n\
        - Technology gap in mobile solutions\n\
        - Geographic expansion potential in APAC\n\n\
        Data quality: High confidence (85%)",
        topic, research_scope
    );
    Ok(market_data)
}

#[tool]
/// Analyze financial projections and business models
async fn financial_analysis(business_model: String) -> Result<String, String> {
    let financial_report = format!(
        "Financial Analysis for '{}':\n\n\
        💰 REVENUE PROJECTIONS (3-year):\n\
        - Year 1: $2.5M (baseline scenarios)\n\
        - Year 2: $6.8M (growth acceleration)\n\
        - Year 3: $15.2M (market expansion)\n\n\
        📈 KEY METRICS:\n\
        - Customer Acquisition Cost (CAC): $1,200\n\
        - Customer Lifetime Value (CLV): $8,500\n\
        - Monthly Recurring Revenue (MRR): $185K\n\
        - Gross Margin: 78%\n\
        - Net Margin: 22%\n\n\
        💵 FUNDING REQUIREMENTS:\n\
        - Series A: $5M (product development, team expansion)\n\
        - Series B: $15M (market expansion, scaling)\n\
        - Break-even: Month 18\n\n\
        ⚖️  RISK ASSESSMENT:\n\
        - Market risk: Medium (competitive pressure)\n\
        - Technology risk: Low (proven stack)\n\
        - Execution risk: Medium (team scaling)\n\n\
        Financial model confidence: 80%",
        business_model
    );
    Ok(financial_report)
}

#[tool]
/// Evaluate technical feasibility and requirements
async fn technical_feasibility(concept: String) -> Result<String, String> {
    let tech_assessment = format!(
        "Technical Feasibility Analysis for '{}':\n\n\
        🔧 TECHNICAL REQUIREMENTS:\n\
        - Core Platform: Cloud-native architecture (AWS/Azure)\n\
        - Frontend: React/Vue.js with responsive design\n\
        - Backend: Node.js/Python with microservices\n\
        - Database: PostgreSQL with Redis caching\n\
        - APIs: RESTful with GraphQL for complex queries\n\n\
        ⚡ DEVELOPMENT TIMELINE:\n\
        - MVP: 4-6 months (core features)\n\
        - Beta: 8-10 months (full feature set)\n\
        - Production: 12-15 months (enterprise ready)\n\n\
        👥 TEAM REQUIREMENTS:\n\
        - Senior developers: 4-6 (full-stack expertise)\n\
        - DevOps engineers: 2 (infrastructure automation)\n\
        - QA engineers: 2 (testing and quality assurance)\n\
        - Technical architect: 1 (system design oversight)\n\n\
        🚀 SCALABILITY FACTORS:\n\
        - Performance: Handles 10K concurrent users\n\
        - Storage: Scalable data architecture\n\
        - Security: Enterprise-grade compliance\n\n\
        Technical risk: Low to Medium\n\
        Implementation confidence: 85%",
        concept
    );
    Ok(tech_assessment)
}

#[tool]
/// Assess operational and execution requirements
async fn operational_assessment(plan: String) -> Result<String, String> {
    let ops_evaluation = format!(
        "Operational Assessment for Plan:\n\n\
        🏢 ORGANIZATIONAL STRUCTURE:\n\
        - Leadership team: CEO, CTO, VP Sales, VP Marketing\n\
        - Department heads: 6 key positions\n\
        - Total team size: 25-30 people (12-18 months)\n\
        - Reporting structure: Flat with clear accountability\n\n\
        📋 OPERATIONAL PROCESSES:\n\
        - Product development: Agile with 2-week sprints\n\
        - Customer support: 24/7 with SLA commitments\n\
        - Sales process: Structured pipeline with CRM\n\
        - Marketing: Multi-channel with performance tracking\n\n\
        🎯 SUCCESS METRICS:\n\
        - Customer satisfaction: >90% (NPS score)\n\
        - Employee retention: >85% annually\n\
        - Product uptime: 99.9% availability\n\
        - Sales efficiency: $500K per sales rep\n\n\
        🔄 CONTINUOUS IMPROVEMENT:\n\
        - Monthly business reviews\n\
        - Quarterly strategy adjustments\n\
        - Annual strategic planning\n\n\
        Plan length analyzed: {} characters\n\
        Operational complexity: Medium\n\
        Execution confidence: 78%",
        plan.len()
    );
    Ok(ops_evaluation)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Disable telemetry to avoid OTLP warnings in example
    std::env::set_var("OTEL_ENABLED", "false");

    println!("🎭 Multi-Perspective Evaluation Demo");
    println!("====================================");
    println!("This example shows how agents can evaluate their work from multiple");
    println!("perspectives with weighted scoring for balanced decision-making.\n");

    // Create tools for comprehensive business analysis
    let business_tools = vec![
        market_research(),
        financial_analysis(),
        technical_feasibility(),
        operational_assessment(),
    ];

    // Define multiple evaluation perspectives with different weights
    let perspectives = vec![
        PerspectiveConfig {
            name: "investor_perspective".to_string(),
            prompt: "As a venture capital investor, evaluate this business plan: \
                    Is the market opportunity compelling? Are the financial projections \
                    realistic? Does the team have the right experience? Is this \
                    investable and likely to generate strong returns?"
                .to_string(),
            weight: 0.3, // 30% weight - investor focus
        },
        PerspectiveConfig {
            name: "technical_perspective".to_string(),
            prompt: "As a technical leader, assess this plan: Is the technology \
                    approach sound? Are the development timelines realistic? Can \
                    the team execute on the technical requirements? Are there \
                    significant technical risks or challenges?"
                .to_string(),
            weight: 0.25, // 25% weight - technical feasibility
        },
        PerspectiveConfig {
            name: "market_perspective".to_string(),
            prompt: "As a market analyst, evaluate this opportunity: Is there \
                    real customer demand? Is the competitive analysis thorough? \
                    Are the go-to-market strategies appropriate? Will customers \
                    actually pay for this solution?"
                .to_string(),
            weight: 0.25, // 25% weight - market validation
        },
        PerspectiveConfig {
            name: "operational_perspective".to_string(),
            prompt: "As an operations expert, assess the execution plan: Are the \
                    operational processes well-defined? Is the organizational \
                    structure appropriate? Can the team scale effectively? Are \
                    the success metrics meaningful and achievable?"
                .to_string(),
            weight: 0.2, // 20% weight - operational execution
        },
    ];

    // Create agent with multi-perspective evaluation
    let mut business_agent = Agent::builder()
        .provider("bedrock")
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .system_prompt(
            "You are a comprehensive business strategy consultant. Your role is to \
            create detailed business plans and strategies by conducting thorough \
            market research, financial analysis, technical feasibility studies, \
            and operational assessments. Use available tools to gather comprehensive \
            information and provide actionable insights.",
        )
        .tools(business_tools)
        .with_multi_perspective_evaluation(perspectives)
        .with_high_tool_limit(45) // Allow extensive analysis
        .with_printing_callbacks_config(PrintingConfig {
            show_reasoning: true,
            show_tools: true,
            show_performance: true,
            stream_output: false, // Disable streaming for cleaner output in examples
        })
        .build()
        .await?;

    println!("✅ Agent created with multi-perspective evaluation");
    println!("   - Model: Claude 3.5 Haiku");
    println!("   - Domain: Business Strategy Consulting");
    println!("   - Perspectives: 4 weighted viewpoints");
    println!("     • Investor (30%): ROI and investment attractiveness");
    println!("     • Technical (25%): Feasibility and development risks");
    println!("     • Market (25%): Customer demand and competition");
    println!("     • Operational (20%): Execution and scalability");
    println!("   - High tool limit: 45 calls");
    println!("   - Tools: market_research, financial_analysis, technical_feasibility, operational_assessment");
    println!("   - Callbacks: Tool execution tracking and performance metrics\n");

    // Test the multi-perspective evaluation with a business strategy task
    println!("=== Comprehensive Business Strategy Task ===");
    let strategy_task = "Develop a comprehensive business strategy for launching a new \
                        SaaS platform that uses AI to optimize supply chain management \
                        for mid-market manufacturing companies. The platform should \
                        provide real-time inventory optimization, demand forecasting, \
                        and supplier relationship management. Include market analysis, \
                        financial projections, technical implementation plan, and \
                        operational strategy. The goal is to create a fundable business \
                        plan that can attract Series A investment.";

    println!("Task: {}\n", strategy_task);
    println!("🎭 Agent will evaluate from multiple perspectives with weighted scoring...\n");

    let result = business_agent.execute(strategy_task).await?;

    println!("=== Comprehensive Business Strategy ===");
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

    println!("\n=== Multi-Perspective Evaluation Benefits Demonstrated ===");
    println!("✅ Balanced Assessment: Multiple viewpoints with weighted importance");
    println!("✅ Stakeholder Alignment: Considers all key perspectives");
    println!("✅ Risk Mitigation: Identifies issues from different angles");
    println!("✅ Comprehensive Coverage: No single perspective dominates");
    println!("✅ Weighted Decision Making: Important perspectives have more influence");
    println!("✅ Holistic Quality: Overall assessment considers all factors");

    println!("\n🎉 Multi-perspective evaluation demonstration complete!");
    println!(
        "The agent used {} execution cycles with 4-perspective weighted evaluation",
        result.execution.cycles
    );

    Ok(())
}
