//! Nebula Operations Commander - AgentCore Evaluations Test
//!
//! A complex multi-tool agent designed to generate rich telemetry data
//! for testing AgentCore Evaluations.
//!
//! # Running
//!
//! ```bash
//! cargo run --example 026_nebula_evaluation_test
//! ```
//!
//! CloudWatch telemetry is enabled by default. The traces will be exported to:
//! - Spans: `aws/spans` log group
//! - Log Events: `/aws/bedrock-agentcore/runtimes/nebula-commander-001`
//!
//! # What This Example Does
//!
//! 1. Creates a "Nebula Operations Commander" agent with 9 tools
//! 2. Executes TWO complex multi-step missions with different tools
//! 3. Generates multiple tool calls across multiple cycles with varied delays
//! 4. Exports telemetry to CloudWatch for evaluation testing
//!
//! # Viewing Results
//!
//! After running, view the traces in:
//! - CloudWatch Console > Application Signals > GenAI Observability
//! - Or run: `./scripts/evaluate_agent.py --defaults` to run evaluations

use stood::agent::Agent;
use stood::telemetry::TelemetryConfig;
use stood::tool;

// ============================================================================
// Mission Planning Tools (Round 1)
// ============================================================================

#[tool]
/// Scan a star system for planets, asteroids, and anomalies
async fn scan_star_system(system_name: String) -> Result<String, String> {
    // Simulate sensor array processing time - medium delay
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    Ok(format!(
        "Scan complete for {}. Found: 4 planets (2 habitable), 1 asteroid belt, \
         3 gas giants, and 1 anomalous energy signature at coordinates 47.3, -12.8.",
        system_name
    ))
}

#[tool]
/// Check fuel levels and calculate range for a given ship
async fn check_fuel_status(ship_id: String) -> Result<String, String> {
    // Simulate ship systems query - fast
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    Ok(format!(
        "Ship {} status: Fuel at 73%, maximum range 450 light-years. \
         Recommended refuel at Proxima Station before deep space mission.",
        ship_id
    ))
}

#[tool]
/// Query crew availability and specializations
async fn query_crew_roster(mission_type: String) -> Result<String, String> {
    // Simulate crew database lookup - slow
    tokio::time::sleep(std::time::Duration::from_millis(800)).await;
    Ok(format!(
        "Available crew for {} mission: Commander Chen (navigation), \
         Dr. Patel (xenobiology), Engineer Martinez (propulsion), \
         Security Officer Kim (tactical). All cleared for duty.",
        mission_type
    ))
}

#[tool]
/// Calculate optimal route between two star systems
async fn calculate_route(origin: String, destination: String) -> Result<String, String> {
    // Simulate navigation computer calculations - very slow
    tokio::time::sleep(std::time::Duration::from_millis(1200)).await;
    Ok(format!(
        "Optimal route from {} to {}: Via Kepler-442 waypoint. \
         Total distance: 127 light-years. Estimated travel time: 3.2 days at warp 7. \
         Hazard assessment: Low (2 minor asteroid fields).",
        origin, destination
    ))
}

#[tool]
/// Initiate communication with a space station or outpost
async fn hail_station(station_name: String, message: String) -> Result<String, String> {
    // Simulate subspace communication delay - medium
    tokio::time::sleep(std::time::Duration::from_millis(600)).await;
    Ok(format!(
        "Transmission to {} sent: '{}'. Response received: \
         'Acknowledged, Nebula Command. Docking bay 7 reserved. \
         Local time 14:32 UTC. Welcome.'",
        station_name, message
    ))
}

// ============================================================================
// Emergency Response Tools (Round 2 - Different tools, different delays)
// ============================================================================

#[tool]
/// Analyze an anomaly detected during scanning
async fn analyze_anomaly(coordinates: String) -> Result<String, String> {
    // Deep analysis takes time - very slow
    tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
    Ok(format!(
        "Anomaly analysis at {}: Detected quantum fluctuations consistent with \
         an unstable wormhole. Energy output: 4.7 terawatts. Stability index: 0.3 (UNSTABLE). \
         Recommend maintaining 500km minimum safe distance. Potential navigation hazard.",
        coordinates
    ))
}

#[tool]
/// Deploy a probe to investigate a target
async fn deploy_probe(target: String, probe_type: String) -> Result<String, String> {
    // Probe deployment and initial telemetry - medium
    tokio::time::sleep(std::time::Duration::from_millis(700)).await;
    Ok(format!(
        "Deployed {} probe to {}. Probe status: ACTIVE. \
         Initial telemetry received. Radiation levels nominal. \
         Atmospheric composition analysis in progress. ETA for full report: 2 hours.",
        probe_type, target
    ))
}

#[tool]
/// Check shield and defense system status
async fn check_shields(ship_id: String) -> Result<String, String> {
    // Quick diagnostic - fast
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    Ok(format!(
        "Ship {} defensive systems: Shields at 100% capacity. \
         Deflector array online. Hull integrity 98%. \
         Weapons systems on standby. Ready for hostile environment.",
        ship_id
    ))
}

#[tool]
/// Send emergency alert to fleet command
async fn send_emergency_alert(alert_level: String, situation: String) -> Result<String, String> {
    // Priority transmission - medium-fast
    tokio::time::sleep(std::time::Duration::from_millis(350)).await;
    Ok(format!(
        "EMERGENCY ALERT SENT - Level: {}. Situation: '{}'. \
         Fleet Command acknowledges. Backup vessels USS Horizon and USS Pathfinder \
         dispatched to your coordinates. ETA: 4.5 hours. Maintain position.",
        alert_level, situation
    ))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for local logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    println!("==============================================");
    println!("  Nebula Operations Commander");
    println!("  AgentCore Evaluations Test (Multi-Round)");
    println!("==============================================\n");

    // ========================================================================
    // Step 1: Configure CloudWatch Telemetry (always enabled)
    // ========================================================================

    let region = std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string());

    // CloudWatch telemetry is always enabled for this example
    let telemetry_config = TelemetryConfig::cloudwatch(&region)
        .with_service_name("nebula-ops-service")
        .with_agent_id("nebula-commander-001");

    println!("CloudWatch telemetry: ENABLED");
    println!("Configuration:");
    println!("  Region: {}", region);
    println!("  Service Name: {}", telemetry_config.service_name());
    if let Some(agent_id) = telemetry_config.agent_id() {
        println!("  Agent ID: {}", agent_id);
    }
    if let Some(log_group) = telemetry_config.log_group_name() {
        println!("  Log Group: {}", log_group);
    }
    println!();

    // ========================================================================
    // Step 2: Build Agent with ALL Tools
    // ========================================================================

    println!("Building Nebula Operations Commander...");

    let tools = vec![
        // Round 1 tools
        scan_star_system(),
        check_fuel_status(),
        query_crew_roster(),
        calculate_route(),
        hail_station(),
        // Round 2 tools
        analyze_anomaly(),
        deploy_probe(),
        check_shields(),
        send_emergency_alert(),
    ];

    let mut agent = Agent::builder()
        .name("Nebula Operations Commander")
        .provider("bedrock")
        .model_str("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        .system_prompt(
            "You are the Nebula Operations Commander, an advanced AI managing a fleet of \
             exploration vessels. You coordinate missions, analyze data, and ensure crew safety. \
             Always use your tools to gather information before making recommendations. \
             Be thorough and use ALL relevant tools for each situation."
        )
        .tools(tools)
        .sequential_execution() // Run tools one at a time for clearer traces
        .with_telemetry(telemetry_config.clone())
        .build()
        .await?;

    println!("Agent ready with 9 tools!\n");

    // ========================================================================
    // ROUND 1: Mission Planning
    // ========================================================================

    println!("##############################################");
    println!("  ROUND 1: Mission Planning");
    println!("##############################################\n");

    let mission_prompt = r#"
Commander, we need to plan an exploration mission to the Tau Ceti system.

Please:
1. Scan the Tau Ceti system for points of interest
2. Check the fuel status of ship NCC-1701-D
3. Query available crew for an exploration mission
4. Calculate the optimal route from Sol to Tau Ceti

Provide a brief mission readiness summary.
"#;

    println!("Mission: {}\n", mission_prompt.trim());

    let result1 = agent.execute(mission_prompt).await?;

    println!("\n--- Round 1 Response ---\n");
    println!("{}", result1.response);
    println!("\n--- Round 1 Metrics ---");
    println!("  Duration: {:?}", result1.duration);
    println!("  Tool Calls: {}", result1.tool_call_summary.total_attempts);
    println!("  Tools: {}", result1.tools_called.join(", "));

    // ========================================================================
    // ROUND 2: Emergency Response (Different tools!)
    // ========================================================================

    println!("\n##############################################");
    println!("  ROUND 2: Emergency Response");
    println!("##############################################\n");

    let emergency_prompt = r#"
ALERT! During the Tau Ceti mission, we detected the anomalous energy signature.

Immediate actions required:
1. Analyze the anomaly at coordinates 47.3, -12.8
2. Check our shield status on ship NCC-1701-D
3. Deploy a science probe to investigate the anomaly
4. Send an emergency alert to Fleet Command about the unstable wormhole situation

Report your findings and recommended actions.
"#;

    println!("Emergency: {}\n", emergency_prompt.trim());

    let result2 = agent.execute(emergency_prompt).await?;

    println!("\n--- Round 2 Response ---\n");
    println!("{}", result2.response);
    println!("\n--- Round 2 Metrics ---");
    println!("  Duration: {:?}", result2.duration);
    println!("  Tool Calls: {}", result2.tool_call_summary.total_attempts);
    println!("  Tools: {}", result2.tools_called.join(", "));

    // ========================================================================
    // Combined Summary
    // ========================================================================

    println!("\n==============================================");
    println!("  Combined Execution Summary");
    println!("==============================================");
    println!("  Round 1 Duration: {:?}", result1.duration);
    println!("  Round 2 Duration: {:?}", result2.duration);
    println!("  Total Tool Calls: {}",
        result1.tool_call_summary.total_attempts + result2.tool_call_summary.total_attempts);
    println!("  Round 1 Tools: {}", result1.tools_called.join(", "));
    println!("  Round 2 Tools: {}", result2.tools_called.join(", "));

    if let (Some(t1), Some(t2)) = (&result1.execution.tokens, &result2.execution.tokens) {
        println!("  Total Tokens: {} input, {} output",
            t1.input_tokens + t2.input_tokens,
            t1.output_tokens + t2.output_tokens);
    }

    // ========================================================================
    // Verification Instructions
    // ========================================================================

    if telemetry_config.is_enabled() {
        println!("\n==============================================");
        println!("  Next Steps: Verify in CloudWatch");
        println!("==============================================\n");

        println!("Wait 2-3 minutes, then check:");
        println!();
        println!("1. CloudWatch Traces:");
        println!("   aws logs start-query \\");
        println!("     --log-group-name aws/spans \\");
        println!("     --start-time $(date -d '10 minutes ago' +%s) \\");
        println!("     --end-time $(date +%s) \\");
        println!("     --query-string 'fields name, @timestamp | filter @message like /nebula/ | limit 20'");
        println!();
        println!("2. GenAI Dashboard: CloudWatch > Application Signals > GenAI");
        println!();
        println!("You should see TWO separate invoke_agent traces with different tool spans!");
    }

    println!("\nDone!");
    Ok(())
}
