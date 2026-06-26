# Example 022: AWS Documentation MCP Server Integration (NEW Simple Method)

This example demonstrates the NEW simple way to integrate with the AWS Documentation MCP server using Docker and the `with_mcp_client()` builder method. It shows how to use the new simplified approach for connecting to MCP servers and verifying that the tools are working correctly.

## 🚀 Quick Start

### Step 1: Docker Setup

**Important**: The pre-built Docker image is not available in Docker Hub. You need to build it from source.

#### Option A: Run the Setup Guide
```bash
# Make setup script executable and run it for guidance
chmod +x ./docker_mcp_setup.sh
./docker_mcp_setup.sh
```

#### Option B: Manual Build Steps
```bash
# 1. Clone the MCP repository
git clone https://github.com/awslabs/mcp.git

# 2. Navigate to the AWS Documentation MCP server directory
cd mcp/src/aws-documentation-mcp-server/

# 3. Build the Docker image (this may take several minutes)
docker build -t awslabs/aws-documentation-mcp-server .

# 4. Verify the image was built successfully
docker images | grep awslabs/aws-documentation-mcp-server

# 5. Return to the example directory
cd path/to/stood/examples/022_aws_doc_mcp
```

### Step 2: Run the Example
```bash
# Navigate to the example directory
cd examples/022_aws_doc_mcp

# Basic run
cargo run --example 022_aws_doc_mcp

# With debug logging to see MCP tool calls
RUST_LOG=debug cargo run --example 022_aws_doc_mcp
```

## 📋 Prerequisites

- **Docker**: Must be installed and running
- **Git**: For cloning the MCP repository
- **Rust**: Latest stable version
- **AWS Knowledge**: Basic understanding helpful but not required
- **Network**: Internet connection for cloning repository and building Docker image
- **Storage**: ~500MB free space for Docker image and build artifacts

## 🎯 What This Example Demonstrates

### Core Functionality
- ✅ **NEW Simple MCP Integration**: Using the `with_mcp_client()` builder method
- ✅ **Docker-based MCP Server**: Starting and managing containerized MCP servers
- ✅ **AWS Documentation Access**: Querying comprehensive AWS service documentation  
- ✅ **Automatic Tool Discovery**: Automatic discovery and registration of MCP tools
- ✅ **Namespace Management**: Prefixing tools to avoid naming conflicts
- ✅ **Verification**: Direct testing that MCP tools are working correctly

### Advanced Features
- ✅ **One-Line Integration**: Simple agent integration with minimal code
- ✅ **Error Handling**: Robust error handling with helpful error messages
- ✅ **Clear Output**: Step-by-step verification that shows MCP usage

## 🔧 Configuration Details

### MCP Server Configuration
The example translates this mcp.json configuration:
```json
{
  "mcpServers": {
    "awslabs.aws-documentation-mcp-server": {
      "command": "docker",
      "args": [
        "run", "--rm", "--interactive",
        "--env", "FASTMCP_LOG_LEVEL=ERROR",
        "--env", "AWS_DOCUMENTATION_PARTITION=aws",
        "awslabs/aws-documentation-mcp-server:latest"
      ],
      "env": {},
      "disabled": false,
      "autoApprove": []
    }
  }
}
```

### NEW Simple Agent Integration
The example demonstrates the new simplified agent integration pattern using `with_mcp_client()`:
```rust
// Step 1: Create and connect MCP client
let mut mcp_client = create_aws_docs_mcp_client().await?;

// Step 2: Create agent using NEW simple method
let mut agent = Agent::builder()
    .provider("bedrock").model_str("us.anthropic.claude-haiku-4-5-20251001-v1:0")
    .system_prompt("You are an AWS expert with access to comprehensive AWS documentation...")
    .with_mcp_client(mcp_client, Some("aws_docs_".to_string())).await? // ONE LINE!
    .build()
    .await?;

// Step 3: Use the agent - MCP tools are automatically available
let result = agent.execute("How do I create a DynamoDB table?").await?;
println!("Agent: {}", result.response);
```

## 📚 Example Queries

The example demonstrates these types of AWS documentation queries:

### CloudFormation
- "What are the key components of a CloudFormation template and how do I define resources?"
- "What are the best practices for IAM role policies in CloudFormation?"

### DynamoDB  
- "How do I create a DynamoDB table with global secondary indexes using the AWS CLI?"
- "What are the performance considerations for DynamoDB partition keys?"

### Rust SDK
- "Show me examples of using the AWS SDK for Rust to interact with S3 buckets"
- "How do I handle errors in the AWS SDK for Rust?"

### General AWS
- "What are the different AWS compute services and when should I use each?"
- "How do I implement least privilege access with IAM policies?"

## 🐛 Troubleshooting

### Common Issues

**Docker Image Not Found**
```bash
# Solution: Build the image from source
git clone https://github.com/awslabs/mcp.git
cd mcp/src/aws-documentation-mcp-server/
docker build -t awslabs/aws-documentation-mcp-server .
```

**Docker Build Fails**
```bash
# Solution: Ensure you have enough disk space and retry
docker system prune  # Clean up old containers/images
docker build -t awslabs/aws-documentation-mcp-server . --no-cache
```

**Connection Timeout**
```bash
# Solution: Increase timeout in the configuration
# Or check Docker daemon is running
docker info
```

**Permission Denied**
```bash
# Solution: Ensure Docker is running and accessible
sudo systemctl start docker  # Linux
# Or restart Docker Desktop    # macOS/Windows
```

**Tool Registration Fails**
```bash
# Solution: Check MCP server logs and increase timeout
RUST_LOG=debug cargo run
```

### Debug Steps

1. **Verify Docker Setup**:
   ```bash
   docker --version
   docker images | grep awslabs/aws-documentation-mcp-server
   ```

2. **Verify Git and Clone**:
   ```bash
   git --version
   git clone https://github.com/awslabs/mcp.git
   ls mcp/src/aws-documentation-mcp-server/
   ```

3. **Test Container Manually**:
   ```bash
   docker run --rm --interactive \
     --env FASTMCP_LOG_LEVEL=DEBUG \
     --env AWS_DOCUMENTATION_PARTITION=aws \
     awslabs/aws-documentation-mcp-server
   ```

3. **Check Logs**:
   ```bash
   RUST_LOG=trace cargo run --example 022_aws_doc_mcp 2>&1 | tee debug.log
   ```

## 🔍 Code Structure

```
022_aws_documentation_mcp.rs
├── verify_mcp_tools()    # Direct MCP tool verification
├── create_aws_docs_mcp_client()  # MCP client creation and connection
├── main()               # NEW simple with_mcp_client() integration
└── Agent execution      # Automatic tool usage verification
```

## 📊 Performance Notes

- **Docker Build Time**: ~5-15 minutes for initial image build from source
- **Startup Time**: ~10-30 seconds for Docker container initialization  
- **Query Response**: ~2-5 seconds per documentation query
- **Memory Usage**: ~100-200MB for the MCP server container
- **Storage**: ~500MB for Docker image and build artifacts
- **Network**: Requires internet for cloning repository and build dependencies

## 🚀 Next Steps

After running this example:

1. **Extend Queries**: Add your own AWS service questions
2. **Custom Tools**: Integrate additional MCP servers
3. **Production Use**: Add proper error handling and monitoring
4. **Integration**: Incorporate into your own applications

## 🔍 Verifying MCP Tool Usage

To confirm that your agent is actually using MCP tools rather than built-in knowledge:

### Quick Verification
```bash
# Run with debug logging to see tool calls
RUST_LOG=debug cargo run

# Look for these log messages:
# [DEBUG stood::mcp::client] Calling MCP tool 'search_documentation'
# [INFO  022_aws_doc_mcp] 🔧 Testing MCP tool directly
# "🎯 SUCCESS: AWS Documentation MCP tools were called!"
```

### Built-in Verification Tests
The example includes automatic verification tests that:
- ✅ Test direct MCP tool calls
- ✅ Analyze agent responses for MCP usage indicators  
- ✅ Provide verification summary and guidance

### Manual Verification Strategies
1. **Ask about very recent AWS features** (newer than model training data)
2. **Request specific documentation URLs** (only MCP server would know)
3. **Compare responses with/without MCP server running**

See `VERIFICATION_GUIDE.md` for comprehensive testing strategies.

