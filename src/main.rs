use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use llama_cpp_profiler::environment::capture_environment;
use llama_cpp_profiler::gguf;
use llama_cpp_profiler::profile::{Preset, SafetyLimits};
use llama_cpp_profiler::report::{self, ExportOptions, ReportOptions};
use llama_cpp_profiler::runner::{self, FullCtxOptions, ServeOptions, TuneOptions};
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
    /// Run bounded llama-server tuning probes.
    Tune(TuneArgs),
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
    #[arg(long, default_value_t = 262_144)]
    ctx: u64,
    #[arg(long, default_value = "standard")]
    preset: Preset,
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
    /// Print the candidate plan as JSON and do not start llama-server.
    #[arg(long)]
    plan: bool,
    /// Use JSON output with --plan.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct FullCtxArgs {
    path: PathBuf,
    #[arg(long, default_value = "interactive-fast")]
    profile: String,
    #[arg(long, default_value_t = 250_000)]
    target_tokens: u64,
    #[arg(long, default_value_t = 262_144)]
    ctx: u64,
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
                    ctx_cap: Some(args.ctx),
                    preset: args.preset,
                    max_runs: args.max_runs,
                    safety: SafetyLimits {
                        min_vram_free_mib: args.min_vram_free_mib,
                        max_swap_delta_mib: args.max_swap_delta_mib,
                    },
                    port_start: args.port_start,
                    gpu_index: args.gpu_index,
                    plan_only,
                },
            )
            .await?;
            if !plan_only {
                println!("{}", report::markdown_for_recommendations(&recs));
            }
        }
        Commands::Fullctx(args) => {
            let result = runner::run_fullctx(
                &args.path,
                FullCtxOptions {
                    profile: args.profile,
                    target_tokens: args.target_tokens,
                    ctx_cap: Some(args.ctx),
                    safety: SafetyLimits {
                        min_vram_free_mib: args.min_vram_free_mib,
                        max_swap_delta_mib: args.max_swap_delta_mib,
                    },
                    port_start: args.port_start,
                    gpu_index: args.gpu_index,
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
