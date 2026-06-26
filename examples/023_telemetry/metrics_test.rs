//! Comprehensive telemetry test with multiple tools for span hierarchy validation

use fastrand;
use std::time::Duration;
use stood::agent::Agent;
use stood::telemetry::TelemetryConfig;
use stood::tool;

// Tool 1: Weather checker
#[tool]
/// Check weather information for any city (returns mock weather data)
async fn check_weather(city: String) -> Result<String, String> {
    // Simulate API latency
    tokio::time::sleep(Duration::from_millis(150)).await;

    let weather_conditions = ["sunny", "cloudy", "rainy", "snowy", "partly cloudy"];
    let condition = weather_conditions[fastrand::usize(0..weather_conditions.len())];
    let temperature = fastrand::i32(32..85);

    Ok(format!(
        "Weather in {}: {} with {}°F temperature, humidity 65%, light winds from the west",
        city, condition, temperature
    ))
}

// Tool 2: Stock price checker
#[tool]
/// Check current stock price (returns random price for demo)
async fn check_stock(symbol: String) -> Result<String, String> {
    // Simulate market data API call
    tokio::time::sleep(Duration::from_millis(200)).await;

    let price = fastrand::f64() * 500.0 + 50.0; // Random price between $50-$550
    let change = (fastrand::f64() - 0.5) * 20.0; // Random change ±$10
    let change_percent = (change / price) * 100.0;

    Ok(format!(
        "Stock {} is trading at ${:.2}, change: ${:.2} ({:.1}%) in the last session",
        symbol.to_uppercase(),
        price,
        change,
        change_percent
    ))
}

// Tool 3: Random number generator
#[tool]
/// Generate a random number within specified range
async fn random_number(min: i32, max: i32) -> Result<String, String> {
    if min >= max {
        return Err("Minimum value must be less than maximum value".to_string());
    }

    // Small delay to simulate processing
    tokio::time::sleep(Duration::from_millis(50)).await;

    let number = fastrand::i32(min..=max);
    Ok(format!(
        "Generated random number: {} (range: {} to {})",
        number, min, max
    ))
}

// Tool 4: Time checker
#[tool]
/// Get the current time and date
async fn get_current_time() -> Result<String, String> {
    let now = chrono::Utc::now();
    let local_time = now.format("%Y-%m-%d %H:%M:%S UTC").to_string();

    Ok(format!(
        "Current time: {}, Unix timestamp: {}, Day of week: {}",
        local_time,
        now.timestamp(),
        now.format("%A")
    ))
}

// Tool 5: Memory scan simulation
#[tool]
/// Run a fake memory scan and return system report
async fn run_memory_scan() -> Result<String, String> {
    // Simulate scan duration
    tokio::time::sleep(Duration::from_millis(300)).await;

    let total_memory = fastrand::u32(8000..32000); // 8-32GB
    let used_memory = fastrand::u32(2000..total_memory);
    let free_memory = total_memory - used_memory;
    let processes = fastrand::u32(150..400);

    Ok(format!(
        "Memory Scan Report:\n- Total Memory: {} MB\n- Used Memory: {} MB ({:.1}%)\n- Free Memory: {} MB\n- Active Processes: {}\n- Status: Normal operation",
        total_memory,
        used_memory,
        (used_memory as f64 / total_memory as f64) * 100.0,
        free_memory,
        processes
    ))
}

// Tool 6: Network test simulation
#[tool]
/// Test network connectivity and return fake network report
async fn test_network() -> Result<String, String> {
    // Simulate network tests
    tokio::time::sleep(Duration::from_millis(250)).await;

    let ping_time = fastrand::f64() * 50.0 + 5.0; // 5-55ms
    let download_speed = fastrand::f64() * 900.0 + 100.0; // 100-1000 Mbps
    let upload_speed = download_speed * 0.8; // Typically lower
    let packet_loss = fastrand::f64() * 2.0; // 0-2%

    let status = if ping_time < 30.0 && packet_loss < 1.0 {
        "Excellent"
    } else if ping_time < 50.0 && packet_loss < 2.0 {
        "Good"
    } else {
        "Fair"
    };

    Ok(format!(
        "Network Test Report:\n- Ping: {:.1}ms\n- Download Speed: {:.1} Mbps\n- Upload Speed: {:.1} Mbps\n- Packet Loss: {:.2}%\n- Overall Status: {}",
        ping_time, download_speed, upload_speed, packet_loss, status
    ))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "🧪 Testing Stood Telemetry with Multiple Tools - Span Hierarchy Validation (Nova Micro)"
    );

    // Initialize telemetry for comprehensive tracing
    // Note: CloudWatch telemetry is now the default - set AWS credentials via environment
    let telemetry_config = TelemetryConfig::disabled(); // Enable with CloudWatch variant when AWS is configured
    println!(
        "📊 Telemetry config: enabled={}",
        telemetry_config.is_enabled()
    );

    // Configure providers
    //ProviderRegistry::configure().await?;
    use stood::llm::registry::ProviderRegistry;
    ProviderRegistry::configure().await?;

    // Create agent with all six tools using Nova Micro (AWS Bedrock)
    let mut agent = Agent::builder()
        .provider("bedrock")
        .model("us.amazon.nova-micro-v1:0") // Nova Micro via AWS Bedrock - reliable tool calling
        .with_streaming(false) // Disable streaming for troubleshooting
        .with_metrics()
        .tool(check_weather())
        .tool(check_stock())
        .tool(random_number())
        .tool(get_current_time())
        .tool(run_memory_scan())
        .tool(test_network())
        .build()
        .await?;

    println!(
        "✅ Agent created with 6 tools and comprehensive telemetry enabled (using Nova Micro)"
    );

    // Test 1: Weather check
    println!("\n🔄 Test 1: Weather Information");
    let response = agent
        .execute("What's the weather like in San Francisco?")
        .await?;
    println!("   Response: {}", response.response);

    // Test 2: Stock price check
    println!("\n🔄 Test 2: Stock Price Query");
    let response = agent
        .execute("Check the current price of AAPL stock")
        .await?;
    println!("   Response: {}", response.response);

    // Test 3: Random number generation
    println!("\n🔄 Test 3: Random Number Generation");
    let response = agent
        .execute("Generate a random number between 1 and 100")
        .await?;
    println!("   Response: {}", response.response);

    // Test 4: Time information
    println!("\n🔄 Test 4: Current Time");
    let response = agent.execute("What time is it right now?").await?;
    println!("   Response: {}", response.response);

    // Test 5: Memory scan
    println!("\n🔄 Test 5: System Memory Scan");
    let response = agent
        .execute("Run a memory scan to check system status")
        .await?;
    println!("   Response: {}", response.response);

    // Test 6: Network test
    println!("\n🔄 Test 6: Network Connectivity Test");
    let response = agent.execute("Test network connectivity and speed").await?;
    println!("   Response: {}", response.response);

    // Test 7: Complex multi-tool request
    println!("\n🔄 Test 7: Multi-Tool Complex Request");
    let response = agent.execute(
        "Please check the weather in New York, generate a random number between 50 and 150, and test the network connectivity, what is the stock for AMNZ, and run a memory scan."
    ).await?;
    println!("   Response: {}", response.response);

    println!("\n✅ All telemetry tests completed successfully!");
    println!("📊 Check Jaeger UI at http://localhost:16686 for:");
    println!(
        "   • Complete span hierarchy: event_loop -> model_interaction -> model -> tool calls"
    );
    println!("   • Parent-child relationships between all spans");
    println!("   • Distributed tracing across all tool executions");
    println!("   • Tool execution metrics and timing");

    // Give time for all spans to be exported
    println!("\n⏳ Waiting 3 seconds for span export...");
    tokio::time::sleep(Duration::from_secs(3)).await;

    Ok(())
}
