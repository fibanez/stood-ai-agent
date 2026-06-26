use std::io::{self, Write};

#[derive(Debug, Clone)]
pub enum SelectedModel {
    ClaudeHaiku45,
    ClaudeSonnet45,
    ClaudeOpus45,
    NovaMicro,
    NovaLite,
    NovaPro,
}

impl SelectedModel {
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::ClaudeHaiku45 => "Claude Haiku 4.5",
            Self::ClaudeSonnet45 => "Claude Sonnet 4.5",
            Self::ClaudeOpus45 => "Claude Opus 4.5",
            Self::NovaMicro => "Nova Micro",
            Self::NovaLite => "Nova Lite",
            Self::NovaPro => "Nova Pro",
        }
    }

    pub fn provider(&self) -> &'static str {
        "bedrock"
    }

    pub fn model_id(&self) -> &'static str {
        match self {
            Self::ClaudeHaiku45 => "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            Self::ClaudeSonnet45 => "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            Self::ClaudeOpus45 => "us.anthropic.claude-opus-4-5-20251101-v1:0",
            Self::NovaMicro => "us.amazon.nova-micro-v1:0",
            Self::NovaLite => "us.amazon.nova-lite-v1:0",
            Self::NovaPro => "us.amazon.nova-pro-v1:0",
        }
    }
}

pub fn select_model_interactively() -> SelectedModel {
    println!("Select a model:");
    println!("1. Claude Haiku 4.5 (fast, cost-effective)");
    println!("2. Claude Sonnet 4.5 (balanced performance)");
    println!("3. Claude Opus 4.5 (maximum intelligence)");
    println!("4. Nova Micro (AWS optimized)");
    println!("5. Nova Lite (AWS mid-tier)");
    println!("6. Nova Pro (AWS high-performance)");

    loop {
        print!("Enter your choice (1-6) [default: 1]: ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        // Default to Haiku if empty
        if input.is_empty() {
            println!("Selected: Claude Haiku 4.5");
            return SelectedModel::ClaudeHaiku45;
        }

        match input {
            "1" => {
                println!("Selected: Claude Haiku 4.5");
                return SelectedModel::ClaudeHaiku45;
            },
            "2" => {
                println!("Selected: Claude Sonnet 4.5");
                return SelectedModel::ClaudeSonnet45;
            },
            "3" => {
                println!("Selected: Claude Opus 4.5");
                return SelectedModel::ClaudeOpus45;
            },
            "4" => {
                println!("Selected: Nova Micro");
                return SelectedModel::NovaMicro;
            },
            "5" => {
                println!("Selected: Nova Lite");
                return SelectedModel::NovaLite;
            },
            "6" => {
                println!("Selected: Nova Pro");
                return SelectedModel::NovaPro;
            },
            _ => {
                println!("Invalid choice. Please enter 1-6.");
                continue;
            }
        }
    }
}
