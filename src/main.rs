use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use llama_cpp_profiler::environment::capture_environment;
use llama_cpp_profiler::gguf;
use llama_cpp_profiler::profile::{Preset, SafetyLimits, WorkloadGoal};
use llama_cpp_profiler::report::{self, ExportOptions, ReportOptions};
use llama_cpp_profiler::runner::{
    self, FullCtxOptions, ProbeMode, RecommendOptions, ServeOptions, TuneOptions,
};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Empirically profile GGUF models with llama.cpp server"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Discover model GGUF files under a directory.
    Scan(ScanArgs),
    /// Inspect GGUF metadata and saved profiler state.
    Inspect(InspectArgs),
    /// Search bounded llama-server candidates and save the best observed configurations.
    Tune(TuneArgs),
    /// Run bounded tuning, then print a best observed server configuration.
    Recommend(RecommendArgs),
    /// Run an explicit near-full-context TTFT and stability probe.
    Fullctx(FullCtxArgs),
    /// Print ranked profiler summaries.
    Report(ReportArgs),
    /// Run or print a saved recommended llama-server profile.
    Serve(ServeArgs),
    /// Print current profiler, hardware, and llama-server environment.
    Doctor(DoctorArgs),
    /// Export Markdown and optional client harness snippets.
    Export(ExportArgs),
}

#[derive(Debug, Args)]
struct ScanArgs {
    path: PathBuf,
    /// Force plain table output instead of the TTY picker.
    #[arg(long)]
    no_tui: bool,
}

#[derive(Debug, Args)]
struct InspectArgs {
    path: PathBuf,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct TuneArgs {
    path: PathBuf,
    #[arg(long)]
    ctx: Option<u64>,
    #[arg(long, default_value = "quick")]
    preset: Preset,
    #[arg(long, value_enum, default_value_t = ProbeMode::Thinking)]
    probe_mode: ProbeMode,
    #[arg(long)]
    max_runs: Option<usize>,
    #[arg(long, default_value_t = 512)]
    min_vram_free_mib: u64,
    #[arg(long, default_value_t = 1024)]
    max_swap_delta_mib: u64,
    #[arg(long, default_value_t = 18_180)]
    port_start: u16,
    #[arg(long)]
    gpu_index: Option<u32>,
    /// Prepend explicit MoE --n-cpu-moe values, comma-separated, using q8_0 16k/4k candidates.
    #[arg(long, value_delimiter = ',')]
    n_cpu_moe_values: Vec<u64>,
    /// Add an opt-in near-full prompt-ingest probe just below the requested context.
    #[arg(long)]
    near_full_ingest: bool,
    /// Override the near-full ingest target token estimate.
    #[arg(long)]
    near_full_target_tokens: Option<u64>,
    /// Validate the selected observed candidate with a long prompt and up to 1024 output tokens.
    #[arg(long)]
    validate_best: bool,
    /// Rerun promising candidates and rank their measurements by median.
    #[arg(long)]
    confirm_best: bool,
    /// Optimize the primary recommendation for generation, prompt ingest, or a balanced workload.
    #[arg(long, value_enum, default_value_t = WorkloadGoal::Balanced)]
    goal: WorkloadGoal,
    /// Print the candidate plan as JSON and do not start llama-server.
    #[arg(long)]
    plan: bool,
    /// Use JSON output with --plan.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct RecommendArgs {
    path: PathBuf,
    #[arg(long)]
    ctx: Option<u64>,
    #[arg(long, default_value = "quick")]
    preset: Preset,
    #[arg(long, value_enum, default_value_t = ProbeMode::Thinking)]
    probe_mode: ProbeMode,
    #[arg(long)]
    max_runs: Option<usize>,
    /// Use a saved profile explicitly instead of the profile selected by --goal.
    #[arg(long)]
    profile: Option<String>,
    /// Optimize the primary recommendation for generation, prompt ingest, or a balanced workload.
    #[arg(long, value_enum, default_value_t = WorkloadGoal::Balanced)]
    goal: WorkloadGoal,
    #[arg(long, default_value_t = 18_080)]
    port: u16,
    #[arg(long, default_value_t = 512)]
    min_vram_free_mib: u64,
    #[arg(long, default_value_t = 1024)]
    max_swap_delta_mib: u64,
    #[arg(long, default_value_t = 18_180)]
    port_start: u16,
    #[arg(long)]
    gpu_index: Option<u32>,
    /// Prepend explicit MoE --n-cpu-moe values, comma-separated, using q8_0 16k/4k candidates.
    #[arg(long, value_delimiter = ',')]
    n_cpu_moe_values: Vec<u64>,
    /// Add an opt-in near-full prompt-ingest probe just below the requested context.
    #[arg(long)]
    near_full_ingest: bool,
    /// Override the near-full ingest target token estimate.
    #[arg(long)]
    near_full_target_tokens: Option<u64>,
    /// Validate the selected observed candidate with a long prompt and up to 1024 output tokens.
    #[arg(long)]
    validate_best: bool,
    /// Rerun promising candidates and rank their measurements by median.
    #[arg(long)]
    confirm_best: bool,
    /// Emit compact JSON for agents instead of human text.
    #[arg(long)]
    agent: bool,
}

#[derive(Debug, Args)]
struct FullCtxArgs {
    path: PathBuf,
    #[arg(long, default_value = "interactive-fast")]
    profile: String,
    /// Override the default prompt target of 80% of the active server context.
    #[arg(long)]
    target_tokens: Option<u64>,
    #[arg(long)]
    ctx: Option<u64>,
    #[arg(long, value_enum, default_value_t = ProbeMode::Thinking)]
    probe_mode: ProbeMode,
    #[arg(long, default_value_t = 512)]
    min_vram_free_mib: u64,
    #[arg(long, default_value_t = 1024)]
    max_swap_delta_mib: u64,
    #[arg(long, default_value_t = 18_180)]
    port_start: u16,
    #[arg(long)]
    gpu_index: Option<u32>,
}

#[derive(Debug, Args)]
struct ReportArgs {
    path: PathBuf,
    #[arg(long)]
    agent: bool,
    #[arg(long)]
    include_stale: bool,
}

#[derive(Debug, Args)]
struct ServeArgs {
    path: PathBuf,
    #[arg(long, default_value = "interactive-fast")]
    profile: String,
    #[arg(long, default_value_t = 18_080)]
    port: u16,
    #[arg(long)]
    print: bool,
    #[arg(long)]
    allow_stale: bool,
}

#[derive(Debug, Args)]
struct DoctorArgs {
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct ExportArgs {
    path: PathBuf,
    #[arg(long)]
    markdown: bool,
    /// Optional opencode config adapter. This is not required for llama.cpp profiling.
    #[arg(long)]
    opencode: Option<PathBuf>,
    /// Print the would-be export. This is the default unless --write is present.
    #[arg(long)]
    dry_run: bool,
    /// Actually patch managed Markdown/opencode outputs.
    #[arg(long)]
    write: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Scan(args) => {
            if runner::is_interactive_stdout() && !args.no_tui {
                let entries = gguf::scan_entries(&args.path)?;
                if let Some(path) = llama_cpp_profiler::ui::pick_scan_entry(&entries)? {
                    println!("{}", path.display());
                }
            } else {
                report::print_scan_table(&args.path)?;
            }
        }
        Commands::Inspect(args) => {
            let value = report::inspect_model(&args.path)?;
            if args.json {
                println!("{}", serde_json::to_string_pretty(&value)?);
            } else {
                println!("path: {}", json_str(&value, "path"));
                println!("name: {}", json_str(&value, "name"));
                println!("architecture: {}", json_str(&value, "architecture"));
                println!("kind: {}", value["kind"].as_str().unwrap_or("-"));
                println!("quant: {}", json_str(&value, "quant"));
                println!("size: {}", json_str(&value, "file_size"));
                println!(
                    "native context: {}",
                    value["native_context"]
                        .as_u64()
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "-".to_string())
                );
                println!(
                    "chat template: {}",
                    if value["tokenizer_has_chat_template"]
                        .as_bool()
                        .unwrap_or(false)
                    {
                        "present"
                    } else {
                        "missing"
                    }
                );
                println!(
                    "MoE experts: {}/{}",
                    value["expert_used_count"]
                        .as_u64()
                        .map_or_else(|| "-".to_string(), |v| v.to_string()),
                    value["expert_count"]
                        .as_u64()
                        .map_or_else(|| "-".to_string(), |v| v.to_string())
                );
                println!(
                    "prior runs: {}",
                    value["prior_run_count"]
                        .as_u64()
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "0".to_string())
                );
            }
        }
        Commands::Tune(args) => {
            if args.json && !args.plan {
                anyhow::bail!("--json is only supported with --plan");
            }
            let plan_only = args.plan;
            let recs = runner::run_tune(
                &args.path,
                TuneOptions {
                    ctx_cap: args.ctx,
                    preset: args.preset,
                    max_runs: args.max_runs,
                    safety: SafetyLimits {
                        min_vram_free_mib: args.min_vram_free_mib,
                        max_swap_delta_mib: args.max_swap_delta_mib,
                    },
                    port_start: args.port_start,
                    gpu_index: args.gpu_index,
                    n_cpu_moe_values: args.n_cpu_moe_values,
                    plan_only,
                    near_full_ingest: args.near_full_ingest,
                    near_full_target_tokens: args.near_full_target_tokens,
                    validate_best: args.validate_best,
                    confirm_best: args.confirm_best,
                    goal: args.goal,
                    probe_mode: args.probe_mode,
                },
            )
            .await?;
            if !plan_only {
                print!("{}", report::markdown_for_recommendations(&recs));
                println!("{}", report::tune_summary(&recs));
            }
        }
        Commands::Recommend(args) => {
            let recommendation = runner::run_recommend(
                &args.path,
                RecommendOptions {
                    ctx_cap: args.ctx,
                    preset: args.preset,
                    max_runs: args.max_runs,
                    profile: args.profile,
                    goal: args.goal,
                    port: args.port,
                    safety: SafetyLimits {
                        min_vram_free_mib: args.min_vram_free_mib,
                        max_swap_delta_mib: args.max_swap_delta_mib,
                    },
                    port_start: args.port_start,
                    gpu_index: args.gpu_index,
                    n_cpu_moe_values: args.n_cpu_moe_values,
                    near_full_ingest: args.near_full_ingest,
                    near_full_target_tokens: args.near_full_target_tokens,
                    validate_best: args.validate_best,
                    confirm_best: args.confirm_best,
                    agent: args.agent,
                    probe_mode: args.probe_mode,
                },
            )
            .await?;
            if args.agent {
                println!("{}", serde_json::to_string(&recommendation)?);
            } else {
                println!("profile: {}", recommendation.profile_id);
                println!("confidence: {}", recommendation.confidence);
                println!("command: {}", recommendation.command);
                if let Some(next) = recommendation.next_suggested_test {
                    println!("next: {next}");
                }
            }
        }
        Commands::Fullctx(args) => {
            let result = runner::run_fullctx(
                &args.path,
                FullCtxOptions {
                    profile: args.profile,
                    target_tokens: args.target_tokens,
                    ctx_cap: args.ctx,
                    safety: SafetyLimits {
                        min_vram_free_mib: args.min_vram_free_mib,
                        max_swap_delta_mib: args.max_swap_delta_mib,
                    },
                    port_start: args.port_start,
                    gpu_index: args.gpu_index,
                    probe_mode: args.probe_mode,
                },
            )
            .await?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        Commands::Report(args) => {
            report::print_report(
                &args.path,
                ReportOptions {
                    agent: args.agent,
                    include_stale: args.include_stale,
                },
            )?;
        }
        Commands::Serve(args) => {
            runner::run_serve(
                &args.path,
                ServeOptions {
                    profile: args.profile,
                    port: args.port,
                    print_only: args.print,
                    allow_stale: args.allow_stale,
                },
            )
            .await?;
        }
        Commands::Doctor(args) => {
            let executable =
                std::env::var("LLAMA_SERVER").unwrap_or_else(|_| "llama-server".to_string());
            let help = std::process::Command::new(&executable)
                .arg("--help")
                .output()
                .ok()
                .filter(|output| output.status.success())
                .map(|output| {
                    format!(
                        "{}\n{}",
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    )
                });
            let snapshot = capture_environment(&executable, help.as_deref());
            if args.json {
                println!("{}", serde_json::to_string_pretty(&snapshot)?);
            } else {
                println!("profiler: {}", snapshot.profiler_version);
                println!("os: {}", snapshot.os);
                println!(
                    "llama-server: {}",
                    if snapshot.llama_server.usable {
                        "usable"
                    } else {
                        "unavailable"
                    }
                );
                println!("gpus: {}", snapshot.gpus.len());
            }
        }
        Commands::Export(args) => {
            report::export(
                &args.path,
                ExportOptions {
                    markdown: args.markdown,
                    opencode: args.opencode,
                    dry_run: args.dry_run || !args.write,
                    write: args.write,
                },
            )?;
        }
    }
    Ok(())
}

fn json_str<'a>(value: &'a serde_json::Value, key: &str) -> &'a str {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .unwrap_or("-")
}
