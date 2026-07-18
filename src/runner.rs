use crate::environment::{EnvironmentSnapshot, capture_environment, compare_environment};
use crate::gguf::{GgufMetadata, ModelIdentity, read_metadata, resolve_model_path};
use crate::profile::{
    ArtifactPaths, CandidateConfig, Manifest, Outcome, Preset, ProbeSummary, ProfileResult,
    RealisticValidation, Recommendation, RecommendationFile, SCHEMA_VERSION, SafetyLimits,
    SearchCoverage, TelemetryStatus, ValidationLevel, WorkloadGoal,
    build_recommendations_for_model, command_display, generate_candidates,
    generate_candidates_for_environment,
};
use crate::report;
use crate::telemetry::{TelemetrySampler, TelemetrySummary};
use anyhow::{Context, Result, anyhow, bail};
use chrono::{DateTime, Utc};
use clap::ValueEnum;
use futures_util::StreamExt;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::ffi::OsStr;
use std::fs;
use std::io::IsTerminal;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};
use tokio::io::{AsyncRead, AsyncWriteExt};
use tokio::process::{Child, Command};

const STARTUP_TIMEOUT: Duration = Duration::from_secs(180);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);
const TELEMETRY_INTERVAL: Duration = Duration::from_millis(500);
const THINKING_BUDGET_MESSAGE: &str = "I should stop thinking and answer now.";
const THREAD_REFINEMENT_MIN_IMPROVEMENT: f64 = 0.03;
const REALISTIC_OUTPUT_TOKENS: u64 = 1024;
const REALISTIC_MIN_RETAINED_RATIO: f64 = 0.25;
const REALISTIC_TIMEOUT_MAX: Duration = Duration::from_secs(2 * 60 * 60);

#[derive(Debug, Clone)]
pub struct TuneOptions {
    pub ctx_cap: Option<u64>,
    pub preset: Preset,
    pub max_runs: Option<usize>,
    pub safety: SafetyLimits,
    pub port_start: u16,
    pub gpu_index: Option<u32>,
    pub n_cpu_moe_values: Vec<u64>,
    pub plan_only: bool,
    pub near_full_ingest: bool,
    pub near_full_target_tokens: Option<u64>,
    pub validate_best: bool,
    pub confirm_best: bool,
    pub goal: WorkloadGoal,
    pub probe_mode: ProbeMode,
}

#[derive(Debug, Clone)]
pub struct FullCtxOptions {
    pub profile: String,
    pub target_tokens: Option<u64>,
    pub ctx_cap: Option<u64>,
    pub safety: SafetyLimits,
    pub port_start: u16,
    pub gpu_index: Option<u32>,
    pub probe_mode: ProbeMode,
}

#[derive(Debug, Clone)]
pub struct ServeOptions {
    pub profile: String,
    pub port: u16,
    pub print_only: bool,
    pub allow_stale: bool,
}

#[derive(Debug, Clone)]
pub struct RecommendOptions {
    pub ctx_cap: Option<u64>,
    pub preset: Preset,
    pub max_runs: Option<usize>,
    pub profile: Option<String>,
    pub goal: WorkloadGoal,
    pub port: u16,
    pub safety: SafetyLimits,
    pub port_start: u16,
    pub gpu_index: Option<u32>,
    pub n_cpu_moe_values: Vec<u64>,
    pub near_full_ingest: bool,
    pub near_full_target_tokens: Option<u64>,
    pub validate_best: bool,
    pub confirm_best: bool,
    pub agent: bool,
    pub probe_mode: ProbeMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendOutput {
    pub agent_schema_version: u32,
    pub schema_version: u32,
    pub model_path: PathBuf,
    pub model_identity: Option<ModelIdentity>,
    pub environment_valid: bool,
    pub telemetry_status: TelemetryStatus,
    pub profile_id: String,
    pub profile_key: String,
    pub goal: WorkloadGoal,
    pub confidence: String,
    pub measurement_count: usize,
    pub coverage: Option<SearchCoverage>,
    pub command: String,
    pub exact_command: String,
    pub output_toks_per_s: Option<f64>,
    pub prompt_toks_per_s: Option<f64>,
    pub ttft_ms: Option<u64>,
    pub requested_context: u64,
    pub validated_prompt_tokens: Option<u64>,
    pub validation_level: ValidationLevel,
    pub realistic_validation: Option<RealisticValidation>,
    pub risk: String,
    pub failures: Vec<crate::profile::RejectedRun>,
    pub stale: Vec<crate::profile::StaleRun>,
    pub next_suggested_test: Option<String>,
}

struct RecommendValidation {
    model_identity: Option<ModelIdentity>,
    environment_valid: bool,
    failures: Vec<crate::profile::RejectedRun>,
    stale: Vec<crate::profile::StaleRun>,
}

#[derive(Debug, Clone, Default)]
enum TelemetrySource {
    #[default]
    Live,
    #[cfg(test)]
    Fixed(TelemetrySummary),
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum ProbeMode {
    #[default]
    Thinking,
    Generic,
}

#[derive(Debug, Clone)]
struct ServerCapabilities {
    executable: String,
    help: String,
}

impl ServerCapabilities {
    fn detect() -> Result<Self> {
        let executable =
            std::env::var("LLAMA_SERVER").unwrap_or_else(|_| "llama-server".to_string());
        let output = std::process::Command::new(&executable)
            .arg("--help")
            .output()
            .with_context(|| {
                format!("run `{executable} --help`; set LLAMA_SERVER if llama-server is elsewhere")
            })?;
        if !output.status.success() {
            bail!("`{executable} --help` exited with {}", output.status);
        }
        let help = format!(
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        Ok(Self { executable, help })
    }

    fn supports(&self, flag: &str) -> bool {
        self.help.contains(flag)
    }

    fn environment(&self) -> EnvironmentSnapshot {
        capture_environment(&self.executable, Some(&self.help))
    }
}

#[derive(Debug, Clone)]
struct RunRequest {
    test_kind: String,
    candidate: CandidateConfig,
    port: u16,
    prompt_plan: PromptPlan,
    probe_mode: ProbeMode,
    telemetry_source: TelemetrySource,
}

#[derive(Debug, Clone)]
enum PromptPlan {
    Tune {
        ingest_target_tokens: u64,
        near_full_ingest_tokens: Option<u64>,
    },
    FullCtx {
        target_tokens: u64,
    },
    Realistic {
        target_tokens: u64,
        output_tokens: u64,
        timeout: Duration,
    },
}

pub async fn run_tune(path: &Path, options: TuneOptions) -> Result<RecommendationFile> {
    run_tune_with_telemetry(path, options, TelemetrySource::Live).await
}

async fn run_tune_with_telemetry(
    path: &Path,
    options: TuneOptions,
    telemetry_source: TelemetrySource,
) -> Result<RecommendationFile> {
    let model_path = resolve_model_path(path)?;
    let metadata = read_metadata(&model_path)?;
    let requested_context = metadata.context_or(options.ctx_cap);
    let capabilities = ServerCapabilities::detect()?;
    let environment = capabilities.environment();
    let mut all_candidates = generate_candidates_for_environment(
        &metadata,
        options.preset,
        requested_context,
        None,
        Some(&environment),
    );
    prepend_explicit_n_cpu_moe_candidates(
        &mut all_candidates,
        &metadata,
        requested_context,
        &options.n_cpu_moe_values,
    );
    if all_candidates.is_empty() {
        bail!("no candidates generated for {}", metadata.path.display());
    }
    let run_budget = options
        .max_runs
        .unwrap_or_else(|| options.preset.max_candidates())
        .min(all_candidates.len());
    if options.plan_only {
        let candidates = all_candidates.iter().take(run_budget).cloned().collect();
        let plan = crate::profile::CandidatePlan {
            schema_version: SCHEMA_VERSION,
            generated_at: Utc::now(),
            model_path: metadata.path.clone(),
            model_identity: metadata.model_identity(),
            requested_context,
            preset: options.preset,
            environment,
            candidates,
        };
        println!("{}", serde_json::to_string_pretty(&plan)?);
        return Ok(RecommendationFile {
            schema_version: SCHEMA_VERSION,
            generated_at: Utc::now(),
            model_path: metadata.path.clone(),
            model_identity: Some(metadata.model_identity()),
            profiles: Vec::new(),
            rejected: Vec::new(),
            stale: Vec::new(),
            environment: None,
            environment_valid: false,
            coverage: None,
            next_suggested_test: None,
        });
    }

    let profiler_dir = metadata.profiler_dir();
    fs::create_dir_all(profiler_dir.join("runs"))?;
    fs::create_dir_all(profiler_dir.join("reports"))?;

    let mut queue = VecDeque::from(all_candidates.clone());
    let mut seen = BTreeSet::new();
    let mut completed = 0usize;
    let mut current_results = Vec::new();
    while completed < run_budget {
        let Some(candidate) = queue.pop_front() else {
            break;
        };
        if !seen.insert(candidate.id.clone()) {
            continue;
        }
        let port = find_open_port(port_with_offset(options.port_start, completed)?)?;
        eprintln!(
            "[{}/{}] {} on port {}",
            completed + 1,
            run_budget,
            candidate.id,
            port
        );
        let request = RunRequest {
            test_kind: "tune".to_string(),
            candidate,
            port,
            prompt_plan: PromptPlan::Tune {
                ingest_target_tokens: options.preset.ingest_target_tokens(),
                near_full_ingest_tokens: options.near_full_ingest.then(|| {
                    options
                        .near_full_target_tokens
                        .unwrap_or_else(|| near_full_ingest_target(requested_context))
                }),
            },
            probe_mode: options.probe_mode,
            telemetry_source: telemetry_source.clone(),
        };
        let validation_level = if options.preset == Preset::Quick {
            ValidationLevel::Smoke
        } else {
            ValidationLevel::StandardIngest
        };
        let result = run_candidate(
            &metadata,
            &capabilities,
            request,
            &options.safety,
            options.gpu_index,
            environment.clone(),
            validation_level,
        )
        .await?;
        eprintln!(
            "  -> {:?}, output {:?} tok/s, prompt {:?} tok/s, headroom {:?} MiB",
            result.outcome,
            result.metrics.server_generation_toks_per_s,
            result.metrics.server_prompt_eval_toks_per_s,
            result.metrics.min_free_vram_mib
        );
        completed += 1;
        promote_adaptive_candidates(&mut queue, &seen, &result);
        current_results.push(result);
    }

    let refined_winner = run_thread_refinement(
        &metadata,
        &capabilities,
        &options,
        &environment,
        &current_results,
        completed,
        &telemetry_source,
    )
    .await?;

    let confirmation_results = if options.confirm_best {
        run_confirmation_runs(
            &metadata,
            &capabilities,
            &options,
            &environment,
            &current_results,
            completed.saturating_add(5),
            &telemetry_source,
        )
        .await?
    } else {
        Vec::new()
    };
    let confirmation_runs = confirmation_results.len();

    if options.preset != Preset::Quick || options.validate_best {
        let mut validation_candidates = current_results
            .iter()
            .filter(|result| {
                result.outcome.is_usable() && !violates_safety(&result.metrics, &options.safety)
            })
            .cloned()
            .collect::<Vec<_>>();
        if let Some(refined) = refined_winner.as_ref() {
            validation_candidates.push(refined.clone());
        }
        validation_candidates.extend(confirmation_results.into_iter().filter(|result| {
            result.outcome.is_usable() && !violates_safety(&result.metrics, &options.safety)
        }));
        let validation_candidates =
            crate::profile::median_confirmed_measurements(validation_candidates.iter().collect());
        run_realistic_validation(
            &metadata,
            &capabilities,
            &options,
            &environment,
            validation_candidates,
            completed.saturating_add(5),
            &telemetry_source,
        )
        .await?;
    }

    write_manifest(&metadata)?;
    let all_results = load_prior_results(&metadata.profiler_dir())?;
    let model_identity = metadata.model_identity();
    let mut recommendations = build_recommendations_for_model(
        metadata.path.clone(),
        Some(&model_identity),
        &all_results,
        &options.safety,
        Some(&environment),
    );
    recommendations.coverage = Some(search_coverage(
        options.preset,
        run_budget,
        completed,
        &all_candidates,
        options.near_full_ingest,
        refined_winner.is_some(),
        confirmation_runs,
    ));
    apply_confidence_labels(&mut recommendations);
    prioritize_goal(&mut recommendations, options.goal);
    write_json(
        metadata.profiler_dir().join("recommendations.json"),
        &recommendations,
    )?;
    report::write_latest_markdown(&metadata.profiler_dir(), &recommendations)?;
    Ok(recommendations)
}

async fn run_confirmation_runs(
    metadata: &GgufMetadata,
    capabilities: &ServerCapabilities,
    options: &TuneOptions,
    environment: &EnvironmentSnapshot,
    primary_results: &[ProfileResult],
    port_offset: usize,
    telemetry_source: &TelemetrySource,
) -> Result<Vec<ProfileResult>> {
    let mut promising = primary_results
        .iter()
        .filter(|result| {
            result.outcome.is_usable() && !violates_safety(&result.metrics, &options.safety)
        })
        .collect::<Vec<_>>();
    promising.sort_by(|left, right| {
        balanced_throughput(right)
            .partial_cmp(&balanced_throughput(left))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    promising.truncate(3);
    if promising.is_empty() {
        eprintln!("confirmation skipped: no safe candidate was observed");
        return Ok(Vec::new());
    }
    eprintln!(
        "confirmation: rerunning {} promising candidates",
        promising.len()
    );
    let confirmation_count = promising.len();
    let mut results = Vec::with_capacity(confirmation_count);
    for (index, baseline) in promising.into_iter().enumerate() {
        let offset = port_offset.saturating_add(index);
        let port = find_open_port(port_with_offset(options.port_start, offset)?)?;
        let result = run_candidate(
            metadata,
            capabilities,
            RunRequest {
                test_kind: "confirmation".to_string(),
                candidate: baseline.candidate.clone(),
                port,
                prompt_plan: PromptPlan::Tune {
                    ingest_target_tokens: options.preset.ingest_target_tokens(),
                    near_full_ingest_tokens: None,
                },
                probe_mode: options.probe_mode,
                telemetry_source: telemetry_source.clone(),
            },
            &options.safety,
            options.gpu_index,
            environment.clone(),
            if options.preset == Preset::Quick {
                ValidationLevel::Smoke
            } else {
                ValidationLevel::StandardIngest
            },
        )
        .await?;
        eprintln!(
            "  confirmed {}: {:?}, output {:?} tok/s, prompt {:?} tok/s",
            baseline.candidate.id,
            result.outcome,
            result.metrics.server_generation_toks_per_s,
            result.metrics.server_prompt_eval_toks_per_s,
        );
        results.push(result);
    }
    Ok(results)
}

fn search_coverage(
    preset: Preset,
    planned_candidates: usize,
    tested_candidates: usize,
    candidates: &[CandidateConfig],
    near_full_ingest: bool,
    thread_refined: bool,
    confirmation_runs: usize,
) -> SearchCoverage {
    let mut searched_dimensions = vec!["context", "batch", "microbatch", "KV cache"]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    if candidates.iter().any(|candidate| {
        candidate.cpu_moe || candidate.n_cpu_moe.is_some() || candidate.gpu_layers.is_some()
    }) {
        searched_dimensions.push("model placement".to_string());
    }
    if thread_refined {
        searched_dimensions.push("CPU threads".to_string());
    }
    let mut not_searched_dimensions = Vec::new();
    if !near_full_ingest {
        not_searched_dimensions.push("near-full context".to_string());
    }
    if !thread_refined {
        not_searched_dimensions.push("CPU threads".to_string());
    }
    if confirmation_runs == 0 {
        not_searched_dimensions.push("repeated measurements".to_string());
    }
    SearchCoverage {
        preset,
        planned_candidates,
        tested_candidates,
        searched_dimensions,
        not_searched_dimensions,
        confirmation_runs,
    }
}

fn prioritize_goal(recommendations: &mut RecommendationFile, goal: WorkloadGoal) {
    let profile_id = goal.profile_id();
    if let Some(index) = recommendations
        .profiles
        .iter()
        .position(|profile| profile.id == profile_id)
    {
        recommendations.profiles.swap(0, index);
    }
}

fn apply_confidence_labels(recommendations: &mut RecommendationFile) {
    let broad_search = recommendations.coverage.as_ref().is_some_and(|coverage| {
        coverage.tested_candidates >= 3 || coverage.tested_candidates >= coverage.planned_candidates
    });
    for profile in &mut recommendations.profiles {
        profile.confidence = if profile.validation_level == ValidationLevel::Fullctx {
            "full-context-validated".to_string()
        } else if profile.measurement_count >= 2 {
            "confirmed".to_string()
        } else if broad_search
            && matches!(
                profile.validation_level,
                ValidationLevel::StandardIngest | ValidationLevel::Realistic
            )
        {
            "benchmarked".to_string()
        } else {
            "provisional".to_string()
        };
    }
}

async fn run_thread_refinement(
    metadata: &GgufMetadata,
    capabilities: &ServerCapabilities,
    options: &TuneOptions,
    environment: &EnvironmentSnapshot,
    primary_results: &[ProfileResult],
    completed_primary_runs: usize,
    telemetry_source: &TelemetrySource,
) -> Result<Option<ProfileResult>> {
    if !capabilities.supports("--threads") || !capabilities.supports("--threads-batch") {
        eprintln!("thread refinement skipped: llama-server does not expose both thread flags");
        return Ok(None);
    }

    let Some(winner) = primary_results
        .iter()
        .filter(|result| {
            result.outcome.is_usable() && !violates_safety(&result.metrics, &options.safety)
        })
        .max_by(|left, right| {
            balanced_throughput(left)
                .partial_cmp(&balanced_throughput(right))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    else {
        eprintln!("thread refinement skipped: no usable primary candidate was observed");
        return Ok(None);
    };

    if !has_meaningful_cpu_participation(winner, metadata) {
        eprintln!("thread refinement skipped: winning placement is fully GPU-resident");
        return Ok(None);
    }

    let thread_candidates = thread_refinement_candidates(&winner.candidate, environment);
    if thread_candidates.len() < 2 {
        eprintln!("thread refinement skipped: CPU topology produced no distinct explicit pairs");
        return Ok(None);
    }

    eprintln!(
        "thread refinement: testing {} configurations for {}",
        thread_candidates.len(),
        winner.candidate.id
    );
    let refinement_count = thread_candidates.len();
    let mut refinement_results = Vec::new();
    for (index, candidate) in thread_candidates.into_iter().enumerate() {
        let port_offset = completed_primary_runs.saturating_add(index);
        let port = find_open_port(port_with_offset(options.port_start, port_offset)?)?;
        eprintln!(
            "  [threads {}/{}] {} on port {}",
            index + 1,
            refinement_count,
            candidate.id,
            port
        );
        let result = run_candidate(
            metadata,
            capabilities,
            RunRequest {
                test_kind: "thread-refinement-observation".to_string(),
                candidate,
                port,
                prompt_plan: PromptPlan::Tune {
                    ingest_target_tokens: options.preset.ingest_target_tokens(),
                    near_full_ingest_tokens: None,
                },
                probe_mode: options.probe_mode,
                telemetry_source: telemetry_source.clone(),
            },
            &options.safety,
            options.gpu_index,
            environment.clone(),
            if options.preset == Preset::Quick {
                ValidationLevel::Smoke
            } else {
                ValidationLevel::StandardIngest
            },
        )
        .await?;
        eprintln!(
            "    -> {:?}, output {:?} tok/s, prompt {:?} tok/s",
            result.outcome,
            result.metrics.server_generation_toks_per_s,
            result.metrics.server_prompt_eval_toks_per_s
        );
        refinement_results.push(result);
    }

    let accepted = accept_thread_refinement(&mut refinement_results, &options.safety)?
        .map(|index| refinement_results[index].clone());
    Ok(accepted)
}

async fn run_realistic_validation(
    metadata: &GgufMetadata,
    capabilities: &ServerCapabilities,
    options: &TuneOptions,
    environment: &EnvironmentSnapshot,
    mut candidates: Vec<ProfileResult>,
    port_offset: usize,
    telemetry_source: &TelemetrySource,
) -> Result<()> {
    candidates.sort_by(|left, right| {
        crate::profile::cache_precision(&right.candidate)
            .cmp(&crate::profile::cache_precision(&left.candidate))
            .then_with(|| {
                balanced_throughput(right)
                    .partial_cmp(&balanced_throughput(left))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    let mut seen = BTreeSet::new();
    candidates.retain(|result| seen.insert(result.candidate.id.clone()));
    if candidates.is_empty() {
        eprintln!("realistic validation skipped: no usable candidate");
        return Ok(());
    }

    for (index, baseline) in candidates.into_iter().enumerate() {
        let target_tokens = realistic_validation_prompt_target(baseline.requested_context);
        let timeout = realistic_validation_timeout(&baseline, target_tokens);
        let offset = port_offset.saturating_add(index);
        let port = find_open_port(port_with_offset(options.port_start, offset)?)?;
        let mut candidate = baseline.candidate.clone();
        candidate.id = format!("{}-realistic-validation", candidate.id);
        candidate.note = format!(
            "Final-stage realistic validation of selected observed candidate {} with about {} prompt tokens and up to {} output tokens",
            baseline.candidate.id, target_tokens, REALISTIC_OUTPUT_TOKENS
        );
        eprintln!(
            "realistic validation: {} on port {} (prompt ~{}, output up to {}, timeout {}s)",
            baseline.candidate.id,
            port,
            target_tokens,
            REALISTIC_OUTPUT_TOKENS,
            timeout.as_secs()
        );
        let mut result = run_candidate(
            metadata,
            capabilities,
            RunRequest {
                test_kind: "realistic-validation".to_string(),
                candidate,
                port,
                prompt_plan: PromptPlan::Realistic {
                    target_tokens,
                    output_tokens: REALISTIC_OUTPUT_TOKENS,
                    timeout,
                },
                probe_mode: options.probe_mode,
                telemetry_source: telemetry_source.clone(),
            },
            &options.safety,
            options.gpu_index,
            environment.clone(),
            ValidationLevel::Realistic,
        )
        .await?;
        apply_realistic_validation_assessment(&mut result, &baseline, target_tokens);
        write_json(&result.artifacts.result_json, &result)?;
        eprintln!(
            "  -> {:?}, generated {:?}/{}, prompt retained {}, output retained {}",
            result.outcome,
            result.completion_tokens,
            REALISTIC_OUTPUT_TOKENS,
            format_ratio(
                result
                    .realistic_validation
                    .as_ref()
                    .and_then(|value| value.prompt_retained_ratio)
            ),
            format_ratio(
                result
                    .realistic_validation
                    .as_ref()
                    .and_then(|value| value.generation_retained_ratio)
            ),
        );
        if result.outcome.is_usable() {
            return Ok(());
        }
        eprintln!(
            "realistic validation: trying the next ranked candidate after {} failed",
            baseline.candidate.id
        );
    }
    eprintln!("realistic validation: all ranked candidates failed");
    Ok(())
}

fn realistic_validation_prompt_target(context: u64) -> u64 {
    let target = (context / 4).clamp(16_000, 64_000);
    target.min(context.saturating_sub(REALISTIC_OUTPUT_TOKENS).max(1))
}

fn realistic_validation_timeout(baseline: &ProfileResult, target_prompt_tokens: u64) -> Duration {
    let prompt_seconds = baseline
        .metrics
        .server_prompt_eval_toks_per_s
        .filter(|value| *value > 0.0)
        .map_or(REQUEST_TIMEOUT.as_secs_f64(), |speed| {
            target_prompt_tokens as f64 / speed
        });
    let generation_seconds = baseline
        .metrics
        .server_generation_toks_per_s
        .filter(|value| *value > 0.0)
        .map_or(REQUEST_TIMEOUT.as_secs_f64(), |speed| {
            REALISTIC_OUTPUT_TOKENS as f64 / speed
        });
    let scaled = Duration::from_secs_f64((prompt_seconds + generation_seconds) * 2.0 + 120.0);
    scaled.clamp(REQUEST_TIMEOUT, REALISTIC_TIMEOUT_MAX)
}

fn apply_realistic_validation_assessment(
    result: &mut ProfileResult,
    baseline: &ProfileResult,
    target_prompt_tokens: u64,
) {
    let prompt_retained_ratio = retained_ratio(
        result.metrics.server_prompt_eval_toks_per_s,
        baseline.metrics.server_prompt_eval_toks_per_s,
    );
    let generation_retained_ratio = retained_ratio(
        result.metrics.server_generation_toks_per_s,
        baseline.metrics.server_generation_toks_per_s,
    );
    let incomplete_generation = result
        .completion_tokens
        .is_none_or(|tokens| tokens < REALISTIC_OUTPUT_TOKENS);
    result.realistic_validation = Some(RealisticValidation {
        baseline_run_id: baseline.run_id.clone(),
        target_prompt_tokens,
        requested_output_tokens: REALISTIC_OUTPUT_TOKENS,
        actual_prompt_tokens: result.prompt_tokens,
        actual_output_tokens: result.completion_tokens,
        prompt_retained_ratio,
        generation_retained_ratio,
        incomplete_generation,
    });
    if result.outcome.is_usable()
        && [prompt_retained_ratio, generation_retained_ratio]
            .into_iter()
            .flatten()
            .any(|ratio| ratio < REALISTIC_MIN_RETAINED_RATIO)
    {
        result.outcome = Outcome::PerformanceDegraded;
        result.note = format!(
            "realistic validation retained less than {:.0}% of short-probe throughput",
            REALISTIC_MIN_RETAINED_RATIO * 100.0
        );
    } else if result.outcome.is_usable() && incomplete_generation {
        result.note = format!(
            "realistic validation passed with early EOS after {} of up to {} requested output tokens",
            result.completion_tokens.unwrap_or(0),
            REALISTIC_OUTPUT_TOKENS
        );
    } else if result.outcome.is_usable() {
        result.note = "realistic validation passed".to_string();
    }
}

fn retained_ratio(measured: Option<f64>, baseline: Option<f64>) -> Option<f64> {
    measured.zip(baseline).and_then(|(measured, baseline)| {
        (baseline > 0.0 && measured.is_finite() && baseline.is_finite())
            .then_some(measured / baseline)
    })
}

fn format_ratio(value: Option<f64>) -> String {
    value.map_or_else(
        || "unknown".to_string(),
        |value| format!("{:.0}%", value * 100.0),
    )
}

fn thread_refinement_candidates(
    winner: &CandidateConfig,
    environment: &EnvironmentSnapshot,
) -> Vec<CandidateConfig> {
    let logical = environment
        .cpu_cores
        .or_else(|| std::thread::available_parallelism().ok().map(usize::from))
        .unwrap_or(1)
        .max(1);
    let mut system = sysinfo::System::new_all();
    system.refresh_cpu();
    let physical = system
        .physical_core_count()
        .unwrap_or(logical)
        .clamp(1, logical);
    thread_pairs(physical, logical)
        .into_iter()
        .map(|(threads, threads_batch, label)| {
            let mut candidate = winner.clone();
            candidate.id = format!("{}-threads-{label}", winner.id);
            candidate.threads = threads;
            candidate.threads_batch = threads_batch;
            candidate.note = format!(
                "Thread refinement of {}; generation threads {}, prompt threads {}",
                winner.id,
                threads.map_or_else(
                    || "llama.cpp default".to_string(),
                    |value| value.to_string()
                ),
                threads_batch.map_or_else(
                    || "llama.cpp default".to_string(),
                    |value| value.to_string()
                )
            );
            candidate.planning_note =
                "conditional topology-aware refinement of the winning placement".to_string();
            candidate
        })
        .collect()
}

fn thread_pairs(
    physical: usize,
    logical: usize,
) -> Vec<(Option<usize>, Option<usize>, &'static str)> {
    let logical = logical.max(1);
    let physical = physical.clamp(1, logical);
    let half_physical = (physical / 2).max(1);
    let pairs = [
        (None, None, "default"),
        (Some(half_physical), Some(physical), "halfphys-phys"),
        (Some(physical), Some(physical), "phys-phys"),
        (Some(physical), Some(logical), "phys-logical"),
        (Some(logical), Some(logical), "logical-logical"),
    ];
    let mut seen = BTreeSet::new();
    pairs
        .into_iter()
        .filter(|(threads, threads_batch, _)| seen.insert((*threads, *threads_batch)))
        .collect()
}

fn has_meaningful_cpu_participation(result: &ProfileResult, metadata: &GgufMetadata) -> bool {
    if result.candidate.cpu_moe || result.candidate.n_cpu_moe.is_some_and(|value| value > 0) {
        return true;
    }
    if result
        .candidate
        .gpu_layers
        .zip(metadata.block_count)
        .is_some_and(|(gpu_layers, blocks)| gpu_layers < blocks)
    {
        return true;
    }
    let log = fs::read_to_string(&result.artifacts.server_log).unwrap_or_default();
    log_has_partial_gpu_offload(&log)
}

fn log_has_partial_gpu_offload(log: &str) -> bool {
    let regex = Regex::new(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers").expect("valid regex");
    regex.captures_iter(log).any(|captures| {
        let offloaded = captures
            .get(1)
            .and_then(|value| value.as_str().parse::<u64>().ok());
        let total = captures
            .get(2)
            .and_then(|value| value.as_str().parse::<u64>().ok());
        offloaded
            .zip(total)
            .is_some_and(|(offloaded, total)| offloaded < total)
    })
}

fn balanced_throughput(result: &ProfileResult) -> f64 {
    let generation = result.metrics.server_generation_toks_per_s.unwrap_or(0.0);
    let prompt = result.metrics.server_prompt_eval_toks_per_s.unwrap_or(0.0);
    if generation <= 0.0 || prompt <= 0.0 {
        return 0.0;
    }
    2.0 * generation * prompt / (generation + prompt)
}

fn accept_thread_refinement(
    results: &mut [ProfileResult],
    safety: &SafetyLimits,
) -> Result<Option<usize>> {
    let Some(baseline) = results.iter().find(|result| {
        result.candidate.threads.is_none()
            && result.candidate.threads_batch.is_none()
            && result.outcome.is_usable()
            && !violates_safety(&result.metrics, safety)
    }) else {
        eprintln!("thread refinement: default-thread baseline did not pass");
        return Ok(None);
    };
    let baseline_score = balanced_throughput(baseline);
    let best = results
        .iter()
        .enumerate()
        .filter(|(_, result)| {
            result.candidate.threads.is_some()
                && result.outcome.is_usable()
                && !violates_safety(&result.metrics, safety)
        })
        .max_by(|(_, left), (_, right)| {
            balanced_throughput(left)
                .partial_cmp(&balanced_throughput(right))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    let Some((best_index, best)) = best else {
        eprintln!("thread refinement: no explicit thread configuration passed");
        return Ok(None);
    };
    let best_score = balanced_throughput(best);
    if !thread_improvement_is_significant(baseline_score, best_score) {
        eprintln!(
            "thread refinement: retained llama.cpp defaults (highest observed explicit balanced score {:.2}, baseline {:.2})",
            best_score, baseline_score
        );
        return Ok(None);
    }

    let improvement = (best_score / baseline_score - 1.0) * 100.0;
    let accepted = &mut results[best_index];
    accepted.test_kind = "thread-refinement".to_string();
    accepted.note = format!(
        "{}; accepted with {:.1}% balanced-throughput improvement over the contemporaneous default-thread baseline",
        accepted.note, improvement
    );
    write_json(&accepted.artifacts.result_json, accepted)?;
    eprintln!(
        "thread refinement: accepted {} ({:.1}% balanced improvement)",
        accepted.candidate.id, improvement
    );
    Ok(Some(best_index))
}

fn thread_improvement_is_significant(baseline_score: f64, candidate_score: f64) -> bool {
    baseline_score > 0.0
        && candidate_score >= baseline_score * (1.0 + THREAD_REFINEMENT_MIN_IMPROVEMENT)
}

pub async fn run_recommend(path: &Path, options: RecommendOptions) -> Result<RecommendOutput> {
    let recs = run_tune(
        path,
        TuneOptions {
            ctx_cap: options.ctx_cap,
            preset: options.preset,
            max_runs: options.max_runs,
            safety: options.safety.clone(),
            port_start: options.port_start,
            gpu_index: options.gpu_index,
            n_cpu_moe_values: options.n_cpu_moe_values.clone(),
            plan_only: false,
            near_full_ingest: options.near_full_ingest,
            near_full_target_tokens: options.near_full_target_tokens,
            validate_best: options.validate_best,
            confirm_best: options.confirm_best,
            goal: options.goal,
            probe_mode: options.probe_mode,
        },
    )
    .await?;
    let model_path = recs.model_path.clone();
    let profile = recs
        .profiles
        .iter()
        .find(|profile| {
            options
                .profile
                .as_deref()
                .unwrap_or(options.goal.profile_id())
                == profile.id
        })
        .or_else(|| recs.profiles.first())
        .with_context(|| {
            format!(
                "no usable profiles were produced for {}",
                model_path.display()
            )
        })?;
    let command = command_display(&command_with_port(&profile.command, options.port));
    let current_metadata = read_metadata(&model_path)?;
    let current_identity = current_metadata.model_identity();
    let environment_valid = recs.environment_valid
        && recs.model_identity.as_ref() == Some(&current_identity)
        && compare_environment(
            recs.environment.as_ref(),
            &current_environment_best_effort(),
        )
        .is_current();
    if !options.agent {
        eprintln!(
            "best observed profile {} from {}; output {:?} tok/s, prompt {:?} tok/s",
            profile.id,
            profile.source_candidate_id,
            profile.output_toks_per_s,
            profile.prompt_toks_per_s
        );
    }
    Ok(recommend_output(
        &model_path,
        profile,
        command,
        RecommendValidation {
            model_identity: recs.model_identity.clone(),
            environment_valid,
            failures: recs.rejected.clone(),
            stale: recs.stale.clone(),
        },
        recs.next_suggested_test.clone(),
        options.goal,
        recs.coverage.clone(),
    ))
}

fn promote_adaptive_candidates(
    queue: &mut VecDeque<CandidateConfig>,
    seen: &BTreeSet<String>,
    result: &ProfileResult,
) {
    let want_more_aggressive = result.outcome == Outcome::Pass
        && result
            .metrics
            .min_free_vram_mib
            .is_some_and(|free| free >= 2048);
    let want_safer = matches!(
        result.outcome,
        Outcome::Oom | Outcome::TooTight | Outcome::Timeout | Outcome::ServerCrash
    );
    if !want_more_aggressive && !want_safer {
        return;
    }

    let base_score = candidate_aggressiveness(&result.candidate);
    let mut matches = queue
        .iter()
        .enumerate()
        .filter(|(_, candidate)| !seen.contains(&candidate.id))
        .filter(|(_, candidate)| same_candidate_family(&result.candidate, candidate))
        .filter_map(|(index, candidate)| {
            let score = candidate_aggressiveness(candidate);
            if want_more_aggressive && score > base_score {
                Some((index, score - base_score))
            } else if want_safer && score < base_score {
                Some((index, base_score - score))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    matches.sort_by_key(|(_, delta)| std::cmp::Reverse(*delta));
    let mut promoted = Vec::new();
    for (index, _) in matches.into_iter().take(2) {
        if let Some(candidate) = queue.get(index) {
            promoted.push(candidate.id.clone());
        }
    }
    for id in promoted.into_iter().rev() {
        if let Some(index) = queue.iter().position(|candidate| candidate.id == id)
            && let Some(candidate) = queue.remove(index)
        {
            queue.push_front(candidate);
        }
    }
}

fn prepend_explicit_n_cpu_moe_candidates(
    candidates: &mut Vec<CandidateConfig>,
    metadata: &GgufMetadata,
    requested_context: u64,
    values: &[u64],
) {
    if values.is_empty() || metadata.model_kind != crate::gguf::ModelKind::Moe {
        return;
    }

    let mut explicit = Vec::new();
    let mut seen = BTreeSet::new();
    for value in values {
        if let Some(expert_count) = metadata.expert_count
            && *value > expert_count
        {
            continue;
        }
        if !seen.insert(*value) {
            continue;
        }
        explicit.push(CandidateConfig {
            id: format!("moe-q8_0-ncpumoe{value}-b16384-ub4096-explicit"),
            requested_context,
            batch: Some(16_384),
            ubatch: Some(4_096),
            kv_cache: Some("q8_0".to_string()),
            kv_cache_v: None,
            fit_target: Some(1_536),
            gpu_layers: None,
            cpu_moe: false,
            n_cpu_moe: Some(*value),
            threads: None,
            threads_batch: None,
            expected_risk: crate::profile::CandidateRisk::Medium,
            note: "Explicit MoE expert-placement probe requested on the CLI".to_string(),
            planning_note: "front-of-queue candidate from --n-cpu-moe-values".to_string(),
        });
    }
    for candidate in explicit.into_iter().rev() {
        if let Some(index) = candidates
            .iter()
            .position(|existing| same_candidate_shape(existing, &candidate))
        {
            candidates.remove(index);
        }
        candidates.insert(0, candidate);
    }
}

fn same_candidate_shape(left: &CandidateConfig, right: &CandidateConfig) -> bool {
    left.requested_context == right.requested_context
        && left.batch == right.batch
        && left.ubatch == right.ubatch
        && left.kv_cache == right.kv_cache
        && left.kv_cache_v == right.kv_cache_v
        && left.fit_target == right.fit_target
        && left.gpu_layers == right.gpu_layers
        && left.cpu_moe == right.cpu_moe
        && left.n_cpu_moe == right.n_cpu_moe
}

fn same_candidate_family(left: &CandidateConfig, right: &CandidateConfig) -> bool {
    left.id.split('-').next() == right.id.split('-').next()
}

fn candidate_aggressiveness(candidate: &CandidateConfig) -> i64 {
    let context = candidate.requested_context as i64 / 4096;
    let batch = candidate.batch.unwrap_or(0) as i64 / 1024;
    let ubatch = candidate.ubatch.unwrap_or(0) as i64 / 512;
    let fit = candidate
        .fit_target
        .map_or(0, |value| (2048_i64 - value.min(2048) as i64).max(0) / 128);
    let kv = match (
        candidate.kv_cache.as_deref(),
        candidate
            .kv_cache_v
            .as_deref()
            .or(candidate.kv_cache.as_deref()),
    ) {
        (Some("q8_0"), Some("q8_0")) => 3,
        (Some("q8_0"), Some("q4_0")) => 2,
        (Some("q4_0"), Some("q4_0")) => 1,
        _ => 0,
    };
    let moe = if candidate.cpu_moe {
        -64
    } else {
        candidate
            .n_cpu_moe
            .map(|value| 64_i64.saturating_sub(value as i64))
            .unwrap_or(0)
    };
    context + batch + ubatch + fit + kv + moe
}

pub async fn run_fullctx(path: &Path, options: FullCtxOptions) -> Result<ProfileResult> {
    let model_path = resolve_model_path(path)?;
    let metadata = read_metadata(&model_path)?;
    let capabilities = ServerCapabilities::detect()?;
    let environment = capabilities.environment();
    let candidate = candidate_for_saved_profile(&metadata, &options.profile, &environment)?
        .unwrap_or_else(|| {
            let requested_context = metadata.context_or(options.ctx_cap);
            generate_candidates(&metadata, Preset::Quick, requested_context, Some(1))
                .into_iter()
                .next()
                .unwrap_or(CandidateConfig {
                    id: "fullctx-default".to_string(),
                    requested_context,
                    batch: Some(512),
                    ubatch: Some(128),
                    kv_cache: Some("q4_0".to_string()),
                    kv_cache_v: None,
                    fit_target: Some(1536),
                    gpu_layers: None,
                    cpu_moe: false,
                    n_cpu_moe: None,
                    threads: None,
                    threads_batch: None,
                    expected_risk: crate::profile::CandidateRisk::Medium,
                    note: "fallback full-context candidate".to_string(),
                    planning_note: "fallback when no saved profile is available".to_string(),
                })
        });

    fs::create_dir_all(metadata.profiler_dir().join("runs"))?;
    fs::create_dir_all(metadata.profiler_dir().join("reports"))?;
    let target_tokens = fullctx_prompt_target(candidate.requested_context, options.target_tokens);
    let request = RunRequest {
        test_kind: "fullctx".to_string(),
        candidate,
        port: find_open_port(options.port_start)?,
        prompt_plan: PromptPlan::FullCtx { target_tokens },
        probe_mode: options.probe_mode,
        telemetry_source: TelemetrySource::Live,
    };
    let result = run_candidate(
        &metadata,
        &capabilities,
        request,
        &options.safety,
        options.gpu_index,
        environment.clone(),
        ValidationLevel::Fullctx,
    )
    .await?;
    write_manifest(&metadata)?;
    let results = load_prior_results(&metadata.profiler_dir())?;
    let model_identity = metadata.model_identity();
    let recommendations = build_recommendations_for_model(
        metadata.path.clone(),
        Some(&model_identity),
        &results,
        &options.safety,
        Some(&environment),
    );
    write_json(
        metadata.profiler_dir().join("recommendations.json"),
        &recommendations,
    )?;
    report::write_latest_markdown(&metadata.profiler_dir(), &recommendations)?;
    Ok(result)
}

pub async fn run_serve(path: &Path, options: ServeOptions) -> Result<()> {
    let model_path = resolve_model_path(path)?;
    let metadata = read_metadata(&model_path)?;
    let recs = load_recommendations(&metadata.profiler_dir())?;
    let profile = recs
        .profiles
        .iter()
        .find(|profile| profile.id == options.profile)
        .with_context(|| format!("profile {:?} not found", options.profile))?;
    let current_environment = current_environment_best_effort();
    let recommendation_compatibility =
        compare_environment(recs.environment.as_ref(), &current_environment);
    let current_identity = metadata.model_identity();
    let model_identity_valid = recs.model_identity.as_ref() == Some(&current_identity)
        && profile.model_identity.as_ref() == Some(&current_identity);
    if !options.allow_stale
        && (!model_identity_valid
            || !profile.compatibility.is_current()
            || !recommendation_compatibility.is_current())
    {
        bail!(
            "profile {:?} is stale: {}; rerun tune or pass --allow-stale",
            options.profile,
            if !model_identity_valid {
                if recs.model_identity.is_none() || profile.model_identity.is_none() {
                    "legacy state is missing model identity"
                } else {
                    "model at this path changed"
                }
            } else if !recommendation_compatibility.is_current() {
                recommendation_compatibility.reason()
            } else {
                profile
                    .stale_reason
                    .as_deref()
                    .unwrap_or_else(|| profile.compatibility.reason())
            }
        );
    }
    let command = command_with_port(&profile.command, options.port);
    let display = command_display(&command);

    if options.print_only {
        println!("{display}");
        return Ok(());
    }

    let Some((program, args)) = command.split_first() else {
        bail!("saved command is empty");
    };
    eprintln!("running {display}");
    let status = std::process::Command::new(program)
        .args(args)
        .status()
        .with_context(|| format!("run {program}"))?;
    if !status.success() {
        bail!("serve command exited with {status}");
    }
    Ok(())
}

fn recommend_output(
    model_path: &Path,
    profile: &Recommendation,
    command: String,
    validation: RecommendValidation,
    next_suggested_test: Option<String>,
    goal: WorkloadGoal,
    coverage: Option<SearchCoverage>,
) -> RecommendOutput {
    RecommendOutput {
        agent_schema_version: crate::profile::AGENT_SCHEMA_VERSION,
        schema_version: SCHEMA_VERSION,
        model_path: model_path.to_path_buf(),
        model_identity: validation.model_identity,
        environment_valid: validation.environment_valid,
        telemetry_status: profile.telemetry_status,
        profile_id: profile.id.clone(),
        profile_key: profile_key(model_path, profile),
        goal,
        confidence: profile.confidence.clone(),
        measurement_count: profile.measurement_count,
        coverage,
        exact_command: command.clone(),
        command,
        output_toks_per_s: profile.output_toks_per_s,
        prompt_toks_per_s: profile.prompt_toks_per_s,
        ttft_ms: profile.ttft_ms,
        requested_context: profile.requested_context,
        validated_prompt_tokens: profile.validated_prompt_tokens,
        validation_level: profile.validation_level,
        realistic_validation: profile.realistic_validation.clone(),
        risk: profile.risk.clone(),
        failures: validation.failures,
        stale: validation.stale,
        next_suggested_test,
    }
}

pub fn profile_key(model_path: &Path, profile: &Recommendation) -> String {
    format!("{}#{}", model_path.display(), profile.id)
}

pub fn current_environment_best_effort() -> EnvironmentSnapshot {
    let executable = std::env::var("LLAMA_SERVER").unwrap_or_else(|_| "llama-server".to_string());
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
    capture_environment(&executable, help.as_deref())
}

pub fn validate_recommendation_file(
    metadata: &GgufMetadata,
    recommendations: &RecommendationFile,
    current_environment: &EnvironmentSnapshot,
) -> RecommendationFile {
    let mut validated = recommendations.clone();
    let current_identity = metadata.model_identity();
    let identity_compatibility = match validated.model_identity.as_ref() {
        None => Some(crate::environment::Compatibility::LegacyMissingIdentity),
        Some(identity) if identity != &current_identity => {
            Some(crate::environment::Compatibility::ModelChanged)
        }
        Some(_) => None,
    };
    let environment_compatibility =
        compare_environment(validated.environment.as_ref(), current_environment);
    validated.environment_valid = environment_compatibility.is_current();
    let mut stale_profiles = Vec::new();
    let profiles = std::mem::take(&mut validated.profiles);
    for profile in profiles {
        let profile_identity_matches = profile.model_identity.as_ref() == Some(&current_identity);
        let profile_environment_current = profile.compatibility.is_current();
        let compatibility = identity_compatibility
            .or_else(|| {
                (!environment_compatibility.is_current()).then_some(environment_compatibility)
            })
            .or_else(|| {
                (!profile_identity_matches).then_some(if profile.model_identity.is_none() {
                    crate::environment::Compatibility::LegacyMissingIdentity
                } else {
                    crate::environment::Compatibility::ModelChanged
                })
            })
            .or_else(|| (!profile_environment_current).then_some(profile.compatibility));
        if let Some(compatibility) = compatibility {
            stale_profiles.push(crate::profile::StaleRun {
                run_id: profile.source_run_id,
                candidate_id: profile.source_candidate_id,
                compatibility,
                reason: compatibility.reason().to_string(),
            });
        } else {
            validated.profiles.push(profile);
        }
    }
    validated.stale.extend(stale_profiles);
    validated
}

async fn run_candidate(
    metadata: &GgufMetadata,
    capabilities: &ServerCapabilities,
    request: RunRequest,
    safety: &SafetyLimits,
    gpu_index: Option<u32>,
    environment: EnvironmentSnapshot,
    validation_level: ValidationLevel,
) -> Result<ProfileResult> {
    let started_at = Utc::now();
    let run_id = run_id(&request.candidate.id);
    let run_dir = metadata.profiler_dir().join("runs").join(&run_id);
    fs::create_dir_all(&run_dir)?;
    let artifacts = ArtifactPaths {
        command: run_dir.join("command.sh"),
        server_log: run_dir.join("server.log"),
        telemetry_jsonl: run_dir.join("telemetry.jsonl"),
        request_json: run_dir.join("request.json"),
        response_json: run_dir.join("response.json"),
        result_json: run_dir.join("result.json"),
    };
    let command = build_command(
        capabilities,
        metadata,
        &request.candidate,
        request.port,
        request.probe_mode,
    );
    let command_display = command_display(&command);
    write_shell_command(&artifacts.command, &command_display)?;

    let mut server = match LlamaServer::spawn(&command, &artifacts.server_log).await {
        Ok(server) => server,
        Err(err) => {
            ensure_failure_artifacts(&artifacts)?;
            let ended_at = Utc::now();
            let result = empty_result(
                metadata,
                FailedRun {
                    run_id,
                    request,
                    artifacts,
                    command,
                    command_display,
                    started_at,
                    ended_at,
                    outcome: Outcome::ServerCrash,
                    note: format!("failed to start llama-server: {err:#}"),
                    environment,
                    validation_level,
                },
            );
            write_json(&result.artifacts.result_json, &result)?;
            return Ok(result);
        }
    };

    let sampler = match &request.telemetry_source {
        TelemetrySource::Live => Some(TelemetrySampler::start(
            server.pid,
            &artifacts.telemetry_jsonl,
            gpu_index,
            TELEMETRY_INTERVAL,
        )),
        #[cfg(test)]
        TelemetrySource::Fixed(_) => {
            fs::write(&artifacts.telemetry_jsonl, "")?;
            None
        }
    };
    let base_url = format!("http://127.0.0.1:{}/v1", request.port);
    let run_started = Instant::now();
    let drive_result = tokio::select! {
        result = drive_probes_with_server(&base_url, metadata, &request, &mut server.child) => Ok(result),
        _ = tokio::signal::ctrl_c() => Err(anyhow!("interrupted by Ctrl-C")),
    };

    let mut note = String::new();
    let mut outcome = Outcome::Pass;
    let mut probes = BTreeMap::new();
    let mut request_artifact = json!({});
    let mut response_artifact = json!({});

    match drive_result {
        Ok(probe_output) => {
            probes = probe_output.probes;
            request_artifact = json!({ "probes": probe_output.request_artifact });
            response_artifact = json!({ "probes": probe_output.response_artifact });
            if let Some(error) = probe_output.error {
                note = error;
                outcome = classify_probe_error(&note);
            }
        }
        Err(err) => {
            note = err.to_string();
            outcome = if note.contains("interrupted") {
                Outcome::Interrupted
            } else if note.contains("timed out") {
                Outcome::Timeout
            } else {
                Outcome::ServerCrash
            };
        }
    }

    server.terminate().await;
    let telemetry = match (sampler, &request.telemetry_source) {
        (Some(sampler), _) => sampler.stop().await,
        #[cfg(test)]
        (None, TelemetrySource::Fixed(summary)) => summary.clone(),
        (None, TelemetrySource::Live) => TelemetrySummary::default(),
    };
    let ended_at = Utc::now();
    let log_text = fs::read_to_string(&artifacts.server_log).unwrap_or_default();
    if !matches!(outcome, Outcome::Interrupted) && log_indicates_oom(&log_text) {
        outcome = Outcome::Oom;
        note = "server log contains OOM/CUDA allocation failure".to_string();
    }

    let timing = parse_llama_timings(&log_text);
    let mut metrics = crate::profile::RunMetrics::from(telemetry);
    let telemetry_status = if metrics.min_free_vram_mib.is_some() {
        TelemetryStatus::Measured
    } else {
        TelemetryStatus::Unknown
    };
    metrics.server_prompt_eval_toks_per_s = timing.best_prompt_toks_per_s();
    metrics.server_generation_toks_per_s = timing.best_generation_toks_per_s();
    metrics.client_ttft_ms = probes
        .get("output")
        .or_else(|| probes.get("realistic"))
        .or_else(|| probes.get("fullctx"))
        .or_else(|| probes.get("sanity"))
        .and_then(|probe| probe.ttft_ms);
    metrics.total_wall_ms = Some(run_started.elapsed().as_millis() as u64);

    if outcome == Outcome::Pass
        && (metrics.server_prompt_eval_toks_per_s.is_none()
            || metrics.server_generation_toks_per_s.is_none())
    {
        outcome = Outcome::ParsePartial;
        note = "request completed but llama.cpp timing lines were incomplete".to_string();
    }
    if outcome.is_usable() && violates_safety(&metrics, safety) {
        outcome = Outcome::TooTight;
        note = "run passed but violated VRAM or swap safety limits".to_string();
    }

    let prompt_tokens = timing.max_prompt_tokens().or_else(|| {
        probes
            .get("ingest")
            .or_else(|| probes.get("realistic"))
            .or_else(|| probes.get("fullctx"))
            .and_then(|probe| probe.prompt_tokens)
    });
    let completion_tokens = timing.max_generation_tokens().or_else(|| {
        probes
            .get("realistic")
            .or_else(|| probes.get("output"))
            .or_else(|| probes.get("sanity"))
            .and_then(|probe| probe.completion_tokens)
    });

    write_json(&artifacts.request_json, &request_artifact)?;
    write_json(&artifacts.response_json, &response_artifact)?;

    let requested_context = request.candidate.requested_context;
    let result = ProfileResult {
        schema_version: SCHEMA_VERSION,
        run_id,
        started_at,
        ended_at,
        model_path: metadata.path.clone(),
        model_size_bytes: metadata.file_size_bytes,
        model_identity: Some(metadata.model_identity()),
        gguf: metadata.clone(),
        quant: metadata.quant.clone(),
        command,
        command_display,
        candidate: request.candidate,
        test_kind: request.test_kind,
        requested_context,
        prompt_tokens,
        completion_tokens,
        metrics,
        probes,
        outcome,
        artifacts,
        note,
        environment: Some(environment.clone()),
        validation_level,
        compatibility: compare_environment(Some(&environment), &environment),
        telemetry_status,
        realistic_validation: None,
    };
    write_json(&result.artifacts.result_json, &result)?;
    Ok(result)
}

async fn drive_probes_with_server(
    base_url: &str,
    metadata: &GgufMetadata,
    request: &RunRequest,
    child: &mut Child,
) -> ProbeOutput {
    let mut output = ProbeOutput::default();
    if let Err(err) = wait_for_server(base_url, child).await {
        output.fail("startup", err);
        return output;
    }

    if let PromptPlan::Realistic {
        target_tokens,
        output_tokens,
        timeout,
    } = &request.prompt_plan
    {
        let (prompt, estimate) = repeated_realistic_prompt(*target_tokens);
        let realistic = post_chat_completion_with_mode(
            base_url,
            &metadata.display_name(),
            &prompt,
            *output_tokens,
            *timeout,
            request.probe_mode,
        )
        .await;
        let realistic = match realistic {
            Ok(probe) => probe,
            Err(err) => {
                output.fail("realistic", err);
                return output;
            }
        };
        let mut summary = realistic.summary;
        summary.prompt_tokens = Some(estimate);
        output.record_summary(
            "realistic",
            summary,
            realistic.request_json,
            realistic.response_json,
        );
        return output;
    }

    let sanity_prompt = "Reply with exactly the single character: K";
    let sanity = post_chat_completion_with_mode(
        base_url,
        &metadata.display_name(),
        sanity_prompt,
        1,
        REQUEST_TIMEOUT,
        request.probe_mode,
    )
    .await;
    let sanity = match sanity {
        Ok(probe) => probe,
        Err(err) => {
            output.fail("sanity", err);
            return output;
        }
    };
    output.record("sanity", sanity);

    match &request.prompt_plan {
        PromptPlan::Tune {
            ingest_target_tokens,
            near_full_ingest_tokens,
        } => {
            let output_prompt =
                "Write a concise checklist for safely profiling a local llama.cpp model.";
            let output_probe = post_chat_completion_with_mode(
                base_url,
                &metadata.display_name(),
                output_prompt,
                128,
                REQUEST_TIMEOUT,
                request.probe_mode,
            )
            .await;
            let output_probe = match output_probe {
                Ok(probe) => probe,
                Err(err) => {
                    output.fail("output", err);
                    return output;
                }
            };
            output.record("output", output_probe);

            let (ingest_prompt, estimate) = repeated_license_prompt(*ingest_target_tokens);
            let ingest = post_chat_completion_with_mode(
                base_url,
                &metadata.display_name(),
                &ingest_prompt,
                1,
                REQUEST_TIMEOUT,
                request.probe_mode,
            )
            .await;
            let ingest = match ingest {
                Ok(probe) => probe,
                Err(err) => {
                    output.fail("ingest", err);
                    return output;
                }
            };
            let mut summary = ingest.summary;
            summary.prompt_tokens = Some(estimate);
            output.record_summary("ingest", summary, ingest.request_json, ingest.response_json);

            if let Some(target_tokens) = near_full_ingest_tokens {
                let (prompt, estimate) = repeated_license_prompt(*target_tokens);
                let near_full = post_chat_completion_with_mode(
                    base_url,
                    &metadata.display_name(),
                    &prompt,
                    1,
                    REQUEST_TIMEOUT,
                    request.probe_mode,
                )
                .await;
                let near_full = match near_full {
                    Ok(probe) => probe,
                    Err(err) => {
                        output.fail("near_full_ingest", err);
                        return output;
                    }
                };
                let mut summary = near_full.summary;
                summary.prompt_tokens = Some(estimate);
                output.record_summary(
                    "near_full_ingest",
                    summary,
                    near_full.request_json,
                    near_full.response_json,
                );
            }
        }
        PromptPlan::FullCtx { target_tokens } => {
            let (full_prompt, estimate) = repeated_license_prompt(*target_tokens);
            let fullctx = post_chat_completion_with_mode(
                base_url,
                &metadata.display_name(),
                &full_prompt,
                1,
                REQUEST_TIMEOUT,
                request.probe_mode,
            )
            .await;
            let fullctx = match fullctx {
                Ok(probe) => probe,
                Err(err) => {
                    output.fail("fullctx", err);
                    return output;
                }
            };
            let mut summary = fullctx.summary;
            summary.prompt_tokens = Some(estimate);
            output.record_summary(
                "fullctx",
                summary,
                fullctx.request_json,
                fullctx.response_json,
            );
        }
        PromptPlan::Realistic { .. } => unreachable!("handled before sanity probe"),
    }

    output
}

#[derive(Debug, Default)]
struct ProbeOutput {
    probes: BTreeMap<String, ProbeSummary>,
    request_artifact: serde_json::Value,
    response_artifact: serde_json::Value,
    error: Option<String>,
}

impl ProbeOutput {
    fn record(&mut self, name: &str, probe: ChatProbe) {
        self.record_summary(name, probe.summary, probe.request_json, probe.response_json);
    }

    fn record_summary(
        &mut self,
        name: &str,
        summary: ProbeSummary,
        request: serde_json::Value,
        response: serde_json::Value,
    ) {
        self.probes.insert(name.to_string(), summary);
        self.request_values_mut().push(request);
        self.response_values_mut().push(response);
    }

    fn fail(&mut self, name: &str, error: anyhow::Error) {
        let message = format!("{name} probe: {error:#}");
        self.error = Some(message.clone());
        self.request_values_mut()
            .push(json!({"probe": name, "error": &message}));
        self.response_values_mut()
            .push(json!({"probe": name, "error": message}));
    }

    fn request_values_mut(&mut self) -> &mut Vec<serde_json::Value> {
        ensure_array(&mut self.request_artifact)
    }

    fn response_values_mut(&mut self) -> &mut Vec<serde_json::Value> {
        ensure_array(&mut self.response_artifact)
    }
}

fn ensure_array(value: &mut serde_json::Value) -> &mut Vec<serde_json::Value> {
    if !value.is_array() {
        *value = json!([]);
    }
    value.as_array_mut().expect("value was set to an array")
}

#[derive(Debug)]
pub struct ChatProbe {
    pub summary: ProbeSummary,
    pub request_json: serde_json::Value,
    pub response_json: serde_json::Value,
}

pub async fn post_chat_completion(
    base_url: &str,
    model: &str,
    prompt: &str,
    max_tokens: u64,
    timeout: Duration,
) -> Result<ChatProbe> {
    post_chat_completion_with_mode(
        base_url,
        model,
        prompt,
        max_tokens,
        timeout,
        ProbeMode::Thinking,
    )
    .await
}

pub async fn post_chat_completion_with_mode(
    base_url: &str,
    model: &str,
    prompt: &str,
    max_tokens: u64,
    timeout: Duration,
    probe_mode: ProbeMode,
) -> Result<ChatProbe> {
    tokio::time::timeout(
        timeout,
        post_chat_completion_inner(base_url, model, prompt, max_tokens, probe_mode),
    )
    .await
    .context("chat request timed out")?
}

async fn post_chat_completion_inner(
    base_url: &str,
    model: &str,
    prompt: &str,
    max_tokens: u64,
    probe_mode: ProbeMode,
) -> Result<ChatProbe> {
    let client = reqwest::Client::new();
    let url = format!("{base_url}/chat/completions");
    let mut payload = json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": true},
        "stream": true
    });
    if probe_mode == ProbeMode::Generic {
        payload
            .as_object_mut()
            .expect("chat payload is an object")
            .remove("chat_template_kwargs");
    }
    let started = Instant::now();
    let response = client
        .post(&url)
        .json(&payload)
        .send()
        .await?
        .error_for_status()
        .context("chat request failed")?;

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut text = String::new();
    let mut first_token_at = None;
    let mut completion_events = 0u64;
    let mut raw_chunks = Vec::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let chunk_text = String::from_utf8_lossy(&chunk);
        raw_chunks.push(chunk_text.to_string());
        buffer.push_str(&chunk_text);
        for event in drain_sse_events(&mut buffer) {
            if event == "[DONE]" {
                continue;
            }
            let Ok(value) = serde_json::from_str::<serde_json::Value>(&event) else {
                continue;
            };
            let content = value
                .pointer("/choices/0/delta/content")
                .or_else(|| value.pointer("/choices/0/message/content"))
                .and_then(|value| value.as_str())
                .unwrap_or_default();
            if !content.is_empty() {
                if first_token_at.is_none() {
                    first_token_at = Some(started.elapsed());
                }
                completion_events += 1;
                text.push_str(content);
            }
        }
    }

    let wall = started.elapsed();
    let excerpt = if text.chars().count() > 500 {
        Some(text.chars().take(500).collect::<String>())
    } else if text.is_empty() {
        None
    } else {
        Some(text.clone())
    };
    Ok(ChatProbe {
        summary: ProbeSummary {
            prompt_tokens: Some(estimate_tokens(prompt)),
            completion_tokens: Some(completion_events.max(max_tokens.min(1))),
            ttft_ms: first_token_at.map(|duration| duration.as_millis() as u64),
            wall_ms: Some(wall.as_millis() as u64),
            response_excerpt: excerpt,
        },
        request_json: json!({
            "url": url,
            "max_tokens": max_tokens,
            "prompt_chars": prompt.chars().count(),
            "estimated_prompt_tokens": estimate_tokens(prompt),
            "payload": payload
        }),
        response_json: json!({
            "wall_ms": wall.as_millis() as u64,
            "ttft_ms": first_token_at.map(|duration| duration.as_millis() as u64),
            "completion_events": completion_events,
            "text_excerpt": text.chars().take(1000).collect::<String>(),
            "raw_stream_excerpt": raw_chunks.join("").chars().take(4000).collect::<String>()
        }),
    })
}

pub fn drain_sse_events(buffer: &mut String) -> Vec<String> {
    let mut events = Vec::new();
    while let Some(index) = buffer.find('\n') {
        let mut line = buffer.drain(..=index).collect::<String>();
        line.truncate(line.trim_end_matches(['\r', '\n']).len());
        let line = line.trim();
        if let Some(data) = line.strip_prefix("data:") {
            events.push(data.trim().to_string());
        }
    }
    events
}

async fn wait_for_server(base_url: &str, child: &mut Child) -> Result<()> {
    let health = base_url.trim_end_matches("/v1").to_string();
    let client = reqwest::Client::new();
    let started = Instant::now();
    while started.elapsed() < STARTUP_TIMEOUT {
        if let Some(status) = child
            .try_wait()
            .context("check llama-server startup status")?
        {
            bail!("llama-server exited before becoming healthy ({status})");
        }
        let health_ok = client
            .get(format!("{health}/health"))
            .send()
            .await
            .map(|response| response.status().is_success())
            .unwrap_or(false);
        if health_ok {
            return Ok(());
        }
        let models_ok = client
            .get(format!("{base_url}/models"))
            .send()
            .await
            .map(|response| response.status().is_success())
            .unwrap_or(false);
        if models_ok {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    bail!("server startup timed out after {:?}", STARTUP_TIMEOUT);
}

fn classify_probe_error(note: &str) -> Outcome {
    let lower = note.to_ascii_lowercase();
    if lower.contains("interrupted") {
        Outcome::Interrupted
    } else if lower.contains("timed out") || lower.contains("timeout") {
        Outcome::Timeout
    } else if lower.contains("out of memory") || lower.contains("oom") {
        Outcome::Oom
    } else {
        Outcome::ServerCrash
    }
}

struct LlamaServer {
    child: Child,
    pid: u32,
    log_tasks: Vec<tokio::task::JoinHandle<()>>,
}

impl LlamaServer {
    async fn spawn(command: &[String], log_path: &Path) -> Result<Self> {
        let Some((program, args)) = command.split_first() else {
            bail!("empty server command");
        };
        let mut child_command = Command::new(program);
        child_command
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        unsafe {
            child_command.pre_exec(|| {
                libc::setpgid(0, 0);
                Ok(())
            });
        }
        let mut child = child_command
            .spawn()
            .with_context(|| format!("spawn {program}"))?;
        let pid = child.id().context("child process missing pid")?;
        let mut log_tasks = Vec::new();
        if let Some(stdout) = child.stdout.take() {
            log_tasks.push(spawn_log_copy(stdout, log_path.to_path_buf()));
        }
        if let Some(stderr) = child.stderr.take() {
            log_tasks.push(spawn_log_copy(stderr, log_path.to_path_buf()));
        }
        Ok(Self {
            child,
            pid,
            log_tasks,
        })
    }

    async fn terminate(&mut self) {
        #[cfg(unix)]
        unsafe {
            let pgid = -(self.pid as i32);
            libc::kill(pgid, libc::SIGTERM);
        }
        #[cfg(not(unix))]
        {
            let _ = self.child.start_kill();
        }

        let wait = tokio::time::timeout(Duration::from_secs(10), self.child.wait()).await;
        if wait.is_err() {
            #[cfg(unix)]
            unsafe {
                let pgid = -(self.pid as i32);
                libc::kill(pgid, libc::SIGKILL);
            }
            #[cfg(not(unix))]
            {
                let _ = self.child.start_kill();
            }
            let _ = self.child.wait().await;
        }
        for task in self.log_tasks.drain(..) {
            let _ = task.await;
        }
    }
}

fn spawn_log_copy<R>(mut reader: R, log_path: PathBuf) -> tokio::task::JoinHandle<()>
where
    R: AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        if let Ok(mut file) = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
            .await
        {
            let _ = tokio::io::copy(&mut reader, &mut file).await;
            let _ = file.flush().await;
        }
    })
}

fn build_command(
    capabilities: &ServerCapabilities,
    metadata: &GgufMetadata,
    candidate: &CandidateConfig,
    port: u16,
    probe_mode: ProbeMode,
) -> Vec<String> {
    let mut args = vec![capabilities.executable.clone()];
    if capabilities.supports("--model") {
        args.extend(["--model".to_string(), metadata.path.display().to_string()]);
    } else {
        args.extend(["-m".to_string(), metadata.path.display().to_string()]);
    }
    args.extend([
        "--host".to_string(),
        "127.0.0.1".to_string(),
        "--port".to_string(),
        port.to_string(),
        "-c".to_string(),
        candidate.requested_context.to_string(),
    ]);

    if capabilities.supports("--jinja") {
        args.push("--jinja".to_string());
    }
    if capabilities.supports("-np") {
        args.extend(["-np".to_string(), "1".to_string()]);
    }
    if capabilities.supports("--fit") {
        args.extend(["--fit".to_string(), "on".to_string()]);
    }
    if let Some(fit_target) = candidate.fit_target
        && capabilities.supports("--fit-target")
    {
        args.extend(["--fit-target".to_string(), fit_target.to_string()]);
    }
    if capabilities.supports("-fa") {
        args.extend(["-fa".to_string(), "on".to_string()]);
    }
    if probe_mode == ProbeMode::Thinking && capabilities.supports("--reasoning") {
        args.extend(["--reasoning".to_string(), "on".to_string()]);
    }
    if probe_mode == ProbeMode::Thinking && capabilities.supports("--reasoning-budget") {
        args.extend(["--reasoning-budget".to_string(), "4096".to_string()]);
    }
    if probe_mode == ProbeMode::Thinking && capabilities.supports("--chat-template-kwargs") {
        args.extend([
            "--chat-template-kwargs".to_string(),
            "{\"enable_thinking\":true}".to_string(),
        ]);
    }
    if capabilities.supports("--temp") {
        args.extend(["--temp".to_string(), "0.6".to_string()]);
    }
    if capabilities.supports("--top-p") {
        args.extend(["--top-p".to_string(), "0.95".to_string()]);
    }
    if capabilities.supports("--top-k") {
        args.extend(["--top-k".to_string(), "20".to_string()]);
    }
    if capabilities.supports("--min-p") {
        args.extend(["--min-p".to_string(), "0.0".to_string()]);
    }
    if capabilities.supports("--presence-penalty") {
        args.extend(["--presence-penalty".to_string(), "0.0".to_string()]);
    }
    if capabilities.supports("--repeat-penalty") {
        args.extend(["--repeat-penalty".to_string(), "1.0".to_string()]);
    }
    if probe_mode == ProbeMode::Thinking && capabilities.supports("--reasoning-budget-message") {
        args.extend([
            "--reasoning-budget-message".to_string(),
            THINKING_BUDGET_MESSAGE.to_string(),
        ]);
    }
    if let Some(batch) = candidate.batch
        && capabilities.supports("-b")
    {
        args.extend(["-b".to_string(), batch.to_string()]);
    }
    if let Some(ubatch) = candidate.ubatch
        && capabilities.supports("-ub")
    {
        args.extend(["-ub".to_string(), ubatch.to_string()]);
    }
    if let Some(kv_cache) = &candidate.kv_cache {
        if capabilities.supports("-ctk") {
            args.extend(["-ctk".to_string(), kv_cache.clone()]);
        } else if capabilities.supports("--cache-type-k") {
            args.extend(["--cache-type-k".to_string(), kv_cache.clone()]);
        }
        let value_cache = candidate.kv_cache_v.as_ref().unwrap_or(kv_cache);
        if capabilities.supports("-ctv") {
            args.extend(["-ctv".to_string(), value_cache.clone()]);
        } else if capabilities.supports("--cache-type-v") {
            args.extend(["--cache-type-v".to_string(), value_cache.clone()]);
        }
    }
    if let Some(gpu_layers) = candidate.gpu_layers {
        if capabilities.supports("-ngl") {
            args.extend(["-ngl".to_string(), gpu_layers.to_string()]);
        } else if capabilities.supports("--n-gpu-layers") {
            args.extend(["--n-gpu-layers".to_string(), gpu_layers.to_string()]);
        }
    }
    if candidate.cpu_moe && capabilities.supports("--cpu-moe") {
        args.push("--cpu-moe".to_string());
    }
    if let Some(n_cpu_moe) = candidate.n_cpu_moe
        && capabilities.supports("--n-cpu-moe")
    {
        args.extend(["--n-cpu-moe".to_string(), n_cpu_moe.to_string()]);
    }
    if let Some(threads) = candidate.threads
        && capabilities.supports("--threads")
    {
        args.extend(["--threads".to_string(), threads.to_string()]);
    }
    if let Some(threads_batch) = candidate.threads_batch
        && capabilities.supports("--threads-batch")
    {
        args.extend(["--threads-batch".to_string(), threads_batch.to_string()]);
    }
    args
}

struct FailedRun {
    run_id: String,
    request: RunRequest,
    artifacts: ArtifactPaths,
    command: Vec<String>,
    command_display: String,
    started_at: DateTime<Utc>,
    ended_at: DateTime<Utc>,
    outcome: Outcome,
    note: String,
    environment: EnvironmentSnapshot,
    validation_level: ValidationLevel,
}

fn empty_result(metadata: &GgufMetadata, failed: FailedRun) -> ProfileResult {
    let requested_context = failed.request.candidate.requested_context;
    ProfileResult {
        schema_version: SCHEMA_VERSION,
        run_id: failed.run_id,
        started_at: failed.started_at,
        ended_at: failed.ended_at,
        model_path: metadata.path.clone(),
        model_size_bytes: metadata.file_size_bytes,
        model_identity: Some(metadata.model_identity()),
        gguf: metadata.clone(),
        quant: metadata.quant.clone(),
        command: failed.command,
        command_display: failed.command_display,
        candidate: failed.request.candidate,
        test_kind: failed.request.test_kind,
        requested_context,
        prompt_tokens: None,
        completion_tokens: None,
        metrics: crate::profile::RunMetrics::default(),
        probes: BTreeMap::new(),
        outcome: failed.outcome,
        artifacts: failed.artifacts,
        note: failed.note,
        environment: Some(failed.environment.clone()),
        validation_level: failed.validation_level,
        compatibility: compare_environment(Some(&failed.environment), &failed.environment),
        telemetry_status: TelemetryStatus::Unknown,
        realistic_validation: None,
    }
}

fn violates_safety(metrics: &crate::profile::RunMetrics, safety: &SafetyLimits) -> bool {
    metrics
        .min_free_vram_mib
        .is_some_and(|free| free < safety.min_vram_free_mib)
        || metrics
            .swap_delta_mib
            .is_some_and(|delta| delta > safety.max_swap_delta_mib as i64)
}

fn port_with_offset(start: u16, offset: usize) -> Result<u16> {
    if start == 0 {
        bail!("--port-start must be between 1 and 65535");
    }
    let offset = u16::try_from(offset).context("port offset exceeds u16")?;
    start
        .checked_add(offset)
        .context("port range exceeds 65535")
}

fn find_open_port(start: u16) -> Result<u16> {
    if start == 0 {
        bail!("--port-start must be between 1 and 65535");
    }
    for port in start..=start.saturating_add(999) {
        if TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return Ok(port);
        }
    }
    bail!("no open localhost port found from {start}");
}

fn write_shell_command(path: &Path, command: &str) -> Result<()> {
    fs::write(
        path,
        format!("#!/usr/bin/env bash\nset -euo pipefail\nexec {command}\n"),
    )?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(path)?.permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions)?;
    }
    Ok(())
}

fn write_json(path: impl AsRef<Path>, value: &impl Serialize) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(value)?)?;
    Ok(())
}

fn ensure_failure_artifacts(artifacts: &ArtifactPaths) -> Result<()> {
    for path in [
        &artifacts.server_log,
        &artifacts.telemetry_jsonl,
        &artifacts.request_json,
        &artifacts.response_json,
    ] {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        if !path.exists() {
            fs::File::create(path)?;
        }
    }
    Ok(())
}

fn write_manifest(metadata: &GgufMetadata) -> Result<()> {
    let profiler_dir = metadata.profiler_dir();
    let runs = load_prior_results(&profiler_dir)?
        .into_iter()
        .map(|result| result.artifacts.result_json)
        .collect();
    let manifest = Manifest {
        schema_version: SCHEMA_VERSION,
        generated_at: Utc::now(),
        model_path: metadata.path.clone(),
        model_identity: metadata.model_identity(),
        gguf: metadata.clone(),
        runs,
    };
    write_json(profiler_dir.join("manifest.json"), &manifest)
}

fn load_prior_results(profiler_dir: &Path) -> Result<Vec<ProfileResult>> {
    let runs_dir = profiler_dir.join("runs");
    if !runs_dir.exists() {
        return Ok(Vec::new());
    }
    let mut results = Vec::new();
    for entry in fs::read_dir(runs_dir)? {
        let entry = entry?;
        let path = entry.path().join("result.json");
        if path.exists() {
            let data = fs::read_to_string(&path)?;
            match serde_json::from_str::<ProfileResult>(&data) {
                Ok(result) => results.push(result),
                Err(err) => eprintln!("skipping malformed {}: {err:#}", path.display()),
            }
        }
    }
    results.sort_by_key(|result| result.started_at);
    Ok(results)
}

fn load_recommendations(profiler_dir: &Path) -> Result<RecommendationFile> {
    let path = profiler_dir.join("recommendations.json");
    let data = fs::read_to_string(&path)
        .with_context(|| format!("read {}; run tune first", path.display()))?;
    serde_json::from_str(&data).with_context(|| format!("parse {}", path.display()))
}

fn candidate_for_saved_profile(
    metadata: &GgufMetadata,
    profile_id: &str,
    current_environment: &EnvironmentSnapshot,
) -> Result<Option<CandidateConfig>> {
    let recs = match load_recommendations(&metadata.profiler_dir()) {
        Ok(recs) => recs,
        Err(_) => return Ok(None),
    };
    if recs.model_identity.as_ref() != Some(&metadata.model_identity())
        || !compare_environment(recs.environment.as_ref(), current_environment).is_current()
    {
        return Ok(None);
    }
    let Some(profile) = recs
        .profiles
        .iter()
        .find(|profile| profile.id == profile_id)
    else {
        return Ok(None);
    };
    let result_path = metadata
        .profiler_dir()
        .join("runs")
        .join(&profile.source_run_id)
        .join("result.json");
    let data = fs::read_to_string(&result_path)?;
    let result: ProfileResult = serde_json::from_str(&data)?;
    Ok(Some(result.candidate))
}

fn command_with_port(command: &[String], port: u16) -> Vec<String> {
    let mut command = command.to_vec();
    for index in 0..command.len() {
        if command[index] == "--port" && index + 1 < command.len() {
            command[index + 1] = port.to_string();
            return command;
        }
    }
    command.extend(["--port".to_string(), port.to_string()]);
    command
}

fn run_id(candidate_id: &str) -> String {
    let sanitized = candidate_id
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>();
    format!("{}-{sanitized}", Utc::now().format("%Y%m%dT%H%M%SZ"))
}

fn repeated_license_prompt(target_tokens: u64) -> (String, u64) {
    let source = fs::read_to_string("/usr/share/licenses/spdx/MIT.txt")
        .unwrap_or_else(|_| include_str!("../LICENSE-MIT").to_string());
    let source_tokens = estimate_tokens(&source).max(1);
    let repeats = (target_tokens / source_tokens).max(1) + 1;
    let mut prompt = String::with_capacity(source.len() * repeats as usize);
    prompt.push_str("Summarize the following repeated license text in one short sentence.\n\n");
    for _ in 0..repeats {
        prompt.push_str(&source);
        prompt.push_str("\n\n");
    }
    let estimate = estimate_tokens(&prompt);
    (prompt, estimate)
}

fn repeated_realistic_prompt(target_tokens: u64) -> (String, u64) {
    let source = fs::read_to_string("/usr/share/licenses/spdx/MIT.txt")
        .unwrap_or_else(|_| include_str!("../LICENSE-MIT").to_string());
    let target_chars = usize::try_from(target_tokens.saturating_mul(4)).unwrap_or(usize::MAX);
    let instruction = "Analyze the following repeated license text. Produce a detailed, structured review of its obligations, permissions, risks, and practical compliance steps. Continue until the analysis is comprehensive.\n\n";
    let mut prompt = instruction.chars().take(target_chars).collect::<String>();
    let remaining = target_chars.saturating_sub(prompt.chars().count());
    prompt.extend(
        source
            .chars()
            .chain(std::iter::once('\n'))
            .cycle()
            .take(remaining),
    );
    let estimate = estimate_tokens(&prompt);
    (prompt, estimate)
}

fn near_full_ingest_target(requested_context: u64) -> u64 {
    if requested_context <= 2_048 {
        return requested_context.saturating_mul(3) / 4;
    }
    let ratio_target = requested_context.saturating_mul(94) / 100;
    ratio_target
        .min(requested_context.saturating_sub(1024))
        .max(1)
}

fn fullctx_prompt_target(requested_context: u64, override_tokens: Option<u64>) -> u64 {
    let maximum = requested_context.saturating_sub(1).max(1);
    override_tokens
        .unwrap_or_else(|| requested_context.saturating_mul(80) / 100)
        .clamp(1, maximum)
}

fn estimate_tokens(text: &str) -> u64 {
    let chars = text.chars().count() as u64;
    (chars / 4).max(1)
}

fn log_indicates_oom(log_text: &str) -> bool {
    let lower = log_text.to_ascii_lowercase();
    lower.contains("out of memory")
        || lower.contains("cuda error")
        || lower.contains("cudamalloc")
        || lower.contains("failed to allocate")
        || lower.contains("oom")
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct TimingSummary {
    pub prompt_evals: Vec<Timing>,
    pub evals: Vec<Timing>,
}

impl TimingSummary {
    fn best_prompt_toks_per_s(&self) -> Option<f64> {
        self.prompt_evals
            .iter()
            .max_by_key(|timing| timing.tokens)
            .map(|timing| timing.toks_per_s)
    }

    fn best_generation_toks_per_s(&self) -> Option<f64> {
        self.evals
            .iter()
            .max_by_key(|timing| timing.tokens)
            .map(|timing| timing.toks_per_s)
    }

    fn max_prompt_tokens(&self) -> Option<u64> {
        self.prompt_evals.iter().map(|timing| timing.tokens).max()
    }

    fn max_generation_tokens(&self) -> Option<u64> {
        self.evals.iter().map(|timing| timing.tokens).max()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Timing {
    pub millis: f64,
    pub tokens: u64,
    pub toks_per_s: f64,
}

pub fn parse_llama_timings(log_text: &str) -> TimingSummary {
    let prompt_re = Regex::new(
        r"prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens?.*?([0-9.]+)\s*tokens per second",
    )
    .expect("valid prompt timing regex");
    let eval_re = Regex::new(
        r"(?m)(?:^|:|\|)\s*eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*(?:runs?|tokens?).*?([0-9.]+)\s*tokens per second",
    )
    .expect("valid eval timing regex");

    let prompt_evals = prompt_re
        .captures_iter(log_text)
        .filter_map(|captures| timing_from_captures(&captures))
        .collect();
    let evals = eval_re
        .captures_iter(log_text)
        .filter_map(|captures| timing_from_captures(&captures))
        .collect();
    TimingSummary {
        prompt_evals,
        evals,
    }
}

fn timing_from_captures(captures: &regex::Captures<'_>) -> Option<Timing> {
    Some(Timing {
        millis: captures.get(1)?.as_str().parse().ok()?,
        tokens: captures.get(2)?.as_str().parse().ok()?,
        toks_per_s: captures.get(3)?.as_str().parse().ok()?,
    })
}

pub fn is_interactive_stdout() -> bool {
    std::io::stdout().is_terminal()
}

pub fn model_label_for_opencode(metadata: &GgufMetadata, profile_id: &str) -> String {
    let mut label = metadata.display_name();
    if let Some(quant) = &metadata.quant
        && !label.to_ascii_uppercase().contains(quant)
    {
        label.push(' ');
        label.push_str(quant);
    }
    format!("{label} ({profile_id})")
}

pub fn model_key_for_opencode(metadata: &GgufMetadata, profile_id: &str) -> String {
    let base = metadata
        .display_name()
        .to_ascii_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|part| !part.is_empty())
        .take(6)
        .collect::<Vec<_>>()
        .join("-");
    format!("{base}-{profile_id}")
}

pub fn collect_profiled_models(root: &Path) -> Result<Vec<(GgufMetadata, RecommendationFile)>> {
    let mut found = Vec::new();
    if root.is_file() {
        let metadata = read_metadata(root)?;
        for directory in [metadata.profiler_dir(), metadata.legacy_profiler_dir()] {
            let recommendation = directory.join("recommendations.json");
            if recommendation.exists() {
                found.push(recommendation);
            }
        }
    } else {
        collect_recommendation_dirs(root, &mut found)?;
    }
    found.sort_by_key(|path| {
        if path
            .components()
            .any(|component| component.as_os_str() == OsStr::new("models"))
        {
            0
        } else {
            1
        }
    });
    let mut output = Vec::new();
    let mut seen = BTreeSet::new();
    let current_environment = current_environment_best_effort();
    for rec_path in found {
        let data = fs::read_to_string(&rec_path)?;
        let recs: RecommendationFile = serde_json::from_str(&data)?;
        let model_key =
            fs::canonicalize(&recs.model_path).unwrap_or_else(|_| recs.model_path.clone());
        if !seen.insert(model_key) {
            continue;
        }
        if recs.model_path.exists() {
            let metadata = read_metadata(&recs.model_path)?;
            output.push((
                metadata.clone(),
                validate_recommendation_file(&metadata, &recs, &current_environment),
            ));
        }
    }
    Ok(output)
}

fn collect_recommendation_dirs(root: &Path, found: &mut Vec<PathBuf>) -> Result<()> {
    if root.is_file() {
        return Ok(());
    }
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            if path.file_name() == Some(OsStr::new(".llama-cpp-profiler")) {
                collect_recommendation_files(&path, found)?;
                continue;
            }
            collect_recommendation_dirs(&path, found)?;
        }
    }
    Ok(())
}

fn collect_recommendation_files(root: &Path, found: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_recommendation_files(&path, found)?;
        } else if path.file_name() == Some(OsStr::new("recommendations.json")) {
            found.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    #[test]
    fn parses_llama_cpp_timing_lines() {
        let log = r#"
llama_perf_context_print:        prompt eval time =    123.45 ms /    16 tokens (    7.72 ms per token,   129.61 tokens per second)
llama_perf_context_print:               eval time =    200.00 ms /   128 runs   (    1.56 ms per token,   640.00 tokens per second)
llama_perf_context_print:        prompt eval time =   1000.00 ms / 16000 tokens (    0.06 ms per token, 16000.00 tokens per second)
"#;
        let timing = parse_llama_timings(log);
        assert_eq!(timing.prompt_evals.len(), 2);
        assert_eq!(timing.evals.len(), 1);
        assert_eq!(timing.best_prompt_toks_per_s(), Some(16000.0));
        assert_eq!(timing.best_generation_toks_per_s(), Some(640.0));
    }

    #[test]
    fn parses_llama_server_slot_timing_lines() {
        let log = r#"
0.04.579.842 I slot print_timing: id  0 | task 3 | prompt eval time =     204.20 ms /    25 tokens (    8.17 ms per token,   122.43 tokens per second)
0.04.579.845 I slot print_timing: id  0 | task 3 |        eval time =    2120.60 ms /   128 tokens (   16.57 ms per token,    60.36 tokens per second)
0.10.835.450 I slot print_timing: id  0 | task 133 | prompt eval time =    6188.84 ms / 14221 tokens (    0.44 ms per token,  2297.85 tokens per second)
0.10.835.453 I slot print_timing: id  0 | task 133 |        eval time =       0.00 ms /     1 tokens (    0.00 ms per token, 1000000.00 tokens per second)
"#;
        let timing = parse_llama_timings(log);
        assert_eq!(timing.prompt_evals.len(), 2);
        assert_eq!(timing.evals.len(), 2);
        assert_eq!(timing.best_prompt_toks_per_s(), Some(2297.85));
        assert_eq!(timing.best_generation_toks_per_s(), Some(60.36));
    }

    #[test]
    fn drains_sse_data_lines() {
        let mut buffer = "data: {\"a\":1}\n\ndata: [DONE]\n".to_string();
        assert_eq!(
            drain_sse_events(&mut buffer),
            vec!["{\"a\":1}".to_string(), "[DONE]".to_string()]
        );
    }

    #[test]
    fn replaces_saved_command_port() {
        let command = vec![
            "llama-server".to_string(),
            "--port".to_string(),
            "18180".to_string(),
        ];
        let updated = command_with_port(&command, 18080);
        assert_eq!(updated[2], "18080");
    }

    #[test]
    fn generic_command_omits_thinking_arguments() {
        let capabilities = ServerCapabilities {
            executable: "llama-server".to_string(),
            help: "--model --host --port -c --reasoning --reasoning-budget --chat-template-kwargs --reasoning-budget-message --temp --top-p".to_string(),
        };
        let metadata = GgufMetadata {
            path: PathBuf::from("/models/test.gguf"),
            file_name: "test.gguf".to_string(),
            file_size_bytes: 1,
            gguf_version: 3,
            tensor_count: 0,
            metadata_kv_count: 0,
            name: None,
            architecture: Some("llama".to_string()),
            size_label: None,
            native_context: Some(4096),
            block_count: Some(1),
            expert_count: None,
            expert_used_count: None,
            tokenizer_has_chat_template: true,
            quant: None,
            file_type: None,
            model_kind: crate::gguf::ModelKind::Dense,
            metadata: BTreeMap::new(),
        };
        let candidate = test_candidate("dense", 1024, 256, 1536);
        let generic = build_command(
            &capabilities,
            &metadata,
            &candidate,
            18080,
            ProbeMode::Generic,
        );
        let thinking = build_command(
            &capabilities,
            &metadata,
            &candidate,
            18080,
            ProbeMode::Thinking,
        );
        assert!(!generic.iter().any(|part| part == "--reasoning"));
        assert!(!generic.iter().any(|part| part == "--chat-template-kwargs"));
        assert!(thinking.iter().any(|part| part == "--reasoning"));
    }

    #[tokio::test]
    async fn startup_exit_is_reported_without_waiting_for_startup_timeout() {
        let mut child = Command::new("sh").args(["-c", "exit 7"]).spawn().unwrap();
        let error = wait_for_server("http://127.0.0.1:1/v1", &mut child)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("exited before becoming healthy"));
    }

    #[tokio::test]
    async fn streaming_body_stall_is_covered_by_probe_timeout() {
        use std::io::{Read, Write};
        use std::thread;

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0; 4096];
            let _ = stream.read(&mut request);
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 100\r\nConnection: keep-alive\r\n\r\n")
                .unwrap();
            thread::sleep(Duration::from_millis(200));
        });
        let error = post_chat_completion(
            &format!("http://127.0.0.1:{port}/v1"),
            "fake",
            "stall",
            1,
            Duration::from_millis(50),
        )
        .await
        .unwrap_err();
        handle.join().unwrap();
        assert!(error.to_string().contains("timed out"));
    }

    #[test]
    fn adaptive_promotes_more_aggressive_candidate_after_headroom() {
        let base = test_candidate("dense-q8_0-fit1536-b1024-ub256", 1024, 256, 1536);
        let aggressive = test_candidate("dense-q8_0-fit512-b16384-ub4096", 16384, 4096, 512);
        let unrelated = test_candidate("moe-q8_0-cpu-moe-b1024-ub256", 1024, 256, 1536);
        let mut queue = VecDeque::from(vec![unrelated, aggressive.clone()]);
        let mut seen = BTreeSet::new();
        seen.insert(base.id.clone());
        let result = test_result(base, Outcome::Pass, Some(4096));

        promote_adaptive_candidates(&mut queue, &seen, &result);

        assert_eq!(
            queue.front().map(|candidate| candidate.id.as_str()),
            Some(aggressive.id.as_str())
        );
    }

    #[test]
    fn thread_pairs_follow_topology_and_deduplicate_small_systems() {
        assert_eq!(
            thread_pairs(16, 32),
            vec![
                (None, None, "default"),
                (Some(8), Some(16), "halfphys-phys"),
                (Some(16), Some(16), "phys-phys"),
                (Some(16), Some(32), "phys-logical"),
                (Some(32), Some(32), "logical-logical"),
            ]
        );
        assert_eq!(
            thread_pairs(1, 1),
            vec![(None, None, "default"), (Some(1), Some(1), "halfphys-phys")]
        );
    }

    #[test]
    fn thread_refinement_requires_three_percent_balanced_improvement() {
        assert!(!thread_improvement_is_significant(100.0, 102.99));
        assert!(thread_improvement_is_significant(100.0, 103.0));
        assert!(!thread_improvement_is_significant(0.0, 100.0));
    }

    #[test]
    fn realistic_validation_target_is_bounded_and_reserves_output_context() {
        assert_eq!(realistic_validation_prompt_target(131_072), 32_768);
        assert_eq!(realistic_validation_prompt_target(1_000_000), 64_000);
        assert_eq!(realistic_validation_prompt_target(65_536), 16_384);
        assert_eq!(realistic_validation_prompt_target(8_192), 7_168);
        let (_, estimate) = repeated_realistic_prompt(32_768);
        assert_eq!(estimate, 32_768);
    }

    #[test]
    fn fullctx_defaults_to_eighty_percent_and_caps_explicit_targets() {
        assert_eq!(fullctx_prompt_target(262_144, None), 209_715);
        assert_eq!(fullctx_prompt_target(8_192, None), 6_553);
        assert_eq!(fullctx_prompt_target(8_192, Some(4_096)), 4_096);
        assert_eq!(fullctx_prompt_target(8_192, Some(250_000)), 8_191);
    }

    #[test]
    fn port_offsets_reject_zero_and_overflow() {
        assert!(port_with_offset(0, 0).is_err());
        assert_eq!(port_with_offset(65_535, 0).unwrap(), 65_535);
        assert!(port_with_offset(65_535, 1).is_err());
        assert_eq!(port_with_offset(18_180, 5).unwrap(), 18_185);
    }

    #[test]
    fn realistic_validation_timeout_scales_and_is_bounded() {
        let mut baseline = test_result(
            test_candidate("baseline", 1024, 256, 1536),
            Outcome::Pass,
            Some(4096),
        );
        baseline.metrics.server_prompt_eval_toks_per_s = Some(100.0);
        baseline.metrics.server_generation_toks_per_s = Some(10.0);
        let scaled = realistic_validation_timeout(&baseline, 32_000);
        assert!(scaled > REQUEST_TIMEOUT);
        assert!(scaled < REALISTIC_TIMEOUT_MAX);

        baseline.metrics.server_prompt_eval_toks_per_s = Some(1.0);
        baseline.metrics.server_generation_toks_per_s = Some(1.0);
        assert_eq!(
            realistic_validation_timeout(&baseline, 64_000),
            REALISTIC_TIMEOUT_MAX
        );
    }

    #[test]
    fn realistic_validation_accepts_early_eos_but_rejects_severe_degradation() {
        let candidate = test_candidate("baseline", 1024, 256, 1536);
        let mut baseline = test_result(candidate.clone(), Outcome::Pass, Some(4096));
        baseline.run_id = "baseline-run".to_string();
        baseline.metrics.server_prompt_eval_toks_per_s = Some(100.0);
        baseline.metrics.server_generation_toks_per_s = Some(20.0);

        let mut early_eos = test_result(candidate.clone(), Outcome::Pass, Some(4096));
        early_eos.metrics.server_prompt_eval_toks_per_s = Some(80.0);
        early_eos.metrics.server_generation_toks_per_s = Some(15.0);
        early_eos.prompt_tokens = Some(32_000);
        early_eos.completion_tokens = Some(400);
        apply_realistic_validation_assessment(&mut early_eos, &baseline, 32_000);
        assert_eq!(early_eos.outcome, Outcome::Pass);
        assert!(
            early_eos
                .realistic_validation
                .as_ref()
                .unwrap()
                .incomplete_generation
        );

        let mut degraded = test_result(candidate, Outcome::Pass, Some(4096));
        degraded.metrics.server_prompt_eval_toks_per_s = Some(20.0);
        degraded.metrics.server_generation_toks_per_s = Some(15.0);
        degraded.completion_tokens = Some(1024);
        apply_realistic_validation_assessment(&mut degraded, &baseline, 32_000);
        assert_eq!(degraded.outcome, Outcome::PerformanceDegraded);
    }

    #[test]
    fn detects_explicit_and_logged_cpu_participation() {
        let mut result = test_result(
            test_candidate("moe-q8_0-cpu-moe-b1024-ub256", 1024, 256, 1536),
            Outcome::Pass,
            Some(4096),
        );
        assert!(has_meaningful_cpu_participation(&result, &result.gguf));

        result.candidate.cpu_moe = false;
        result.candidate.gpu_layers = Some(1);
        result.gguf.block_count = Some(2);
        assert!(has_meaningful_cpu_participation(&result, &result.gguf));
        assert!(log_has_partial_gpu_offload(
            "load_tensors: offloaded 40/49 layers to GPU"
        ));
        assert!(!log_has_partial_gpu_offload(
            "load_tensors: offloaded 49/49 layers to GPU"
        ));
    }

    #[test]
    fn build_command_includes_explicit_thread_pair() {
        let capabilities = ServerCapabilities {
            executable: "llama-server".to_string(),
            help: "--model --host --port -c --threads --threads-batch".to_string(),
        };
        let mut candidate = test_candidate("moe-cpu-moe", 1024, 256, 1536);
        candidate.threads = Some(16);
        candidate.threads_batch = Some(32);
        let metadata = test_result(candidate.clone(), Outcome::Pass, Some(4096)).gguf;
        let command = build_command(
            &capabilities,
            &metadata,
            &candidate,
            18080,
            ProbeMode::Generic,
        );
        assert!(command.windows(2).any(|pair| pair == ["--threads", "16"]));
        assert!(
            command
                .windows(2)
                .any(|pair| pair == ["--threads-batch", "32"])
        );
    }

    #[test]
    fn build_command_supports_mixed_k_and_v_cache_precision() {
        let capabilities = ServerCapabilities {
            executable: "llama-server".to_string(),
            help: "--model --host --port -c -ctk -ctv".to_string(),
        };
        let mut candidate = test_candidate("mixed-cache", 1024, 256, 1536);
        candidate.kv_cache_v = Some("q4_0".to_string());
        let metadata = test_result(candidate.clone(), Outcome::Pass, Some(4096)).gguf;
        let command = build_command(
            &capabilities,
            &metadata,
            &candidate,
            18080,
            ProbeMode::Generic,
        );
        assert!(command.windows(2).any(|pair| pair == ["-ctk", "q8_0"]));
        assert!(command.windows(2).any(|pair| pair == ["-ctv", "q4_0"]));
    }

    #[test]
    fn explicit_n_cpu_moe_candidates_are_prepended() {
        let metadata = GgufMetadata {
            path: PathBuf::from("/models/test.gguf"),
            file_name: "moe.gguf".to_string(),
            file_size_bytes: 1,
            gguf_version: 3,
            tensor_count: 0,
            metadata_kv_count: 0,
            name: Some("test".to_string()),
            architecture: Some("qwen35moe".to_string()),
            size_label: None,
            native_context: Some(262_144),
            block_count: Some(1),
            expert_count: Some(256),
            expert_used_count: Some(8),
            tokenizer_has_chat_template: true,
            quant: Some("Q4_K_M".to_string()),
            file_type: None,
            model_kind: crate::gguf::ModelKind::Moe,
            metadata: BTreeMap::new(),
        };
        let mut candidates = vec![test_candidate(
            "moe-q8_0-ncpumoe31-b16384-ub4096",
            16_384,
            4_096,
            1_536,
        )];

        prepend_explicit_n_cpu_moe_candidates(&mut candidates, &metadata, 262_144, &[32, 31, 30]);

        let ids = candidates
            .iter()
            .take(3)
            .map(|candidate| candidate.id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            ids,
            vec![
                "moe-q8_0-ncpumoe32-b16384-ub4096-explicit",
                "moe-q8_0-ncpumoe31-b16384-ub4096-explicit",
                "moe-q8_0-ncpumoe30-b16384-ub4096-explicit"
            ]
        );
    }

    #[tokio::test]
    async fn parses_streaming_chat_completion_from_fake_server() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0; 4096];
            let _ = stream.read(&mut request).unwrap();
            let body = "data: {\"choices\":[{\"delta\":{\"content\":\"K\"}}]}\n\ndata: [DONE]\n\n";
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).unwrap();
        });

        let probe = post_chat_completion(
            &format!("http://127.0.0.1:{port}/v1"),
            "fake",
            "say K",
            1,
            Duration::from_secs(5),
        )
        .await
        .unwrap();
        handle.join().unwrap();

        assert_eq!(probe.summary.response_excerpt.as_deref(), Some("K"));
        assert_eq!(probe.summary.completion_tokens, Some(1));
        assert!(probe.summary.ttft_ms.is_some());
    }

    #[tokio::test]
    #[allow(clippy::await_holding_lock)]
    async fn e2e_tune_recommendations_and_serve_print_with_fake_server() {
        let _guard = env_lock().lock().unwrap();
        let temp = tempfile::tempdir().unwrap();
        let server_path = temp.path().join("fake-llama-server.py");
        write_fake_llama_server(&server_path);
        let model_path = temp.path().join("tiny.gguf");
        write_test_gguf(&model_path, 8_192);

        unsafe {
            std::env::set_var("LLAMA_SERVER", &server_path);
        }
        let recs = run_tune_with_telemetry(
            &model_path,
            TuneOptions {
                ctx_cap: None,
                preset: Preset::Quick,
                max_runs: Some(1),
                safety: SafetyLimits::default(),
                port_start: 28_180,
                gpu_index: None,
                n_cpu_moe_values: Vec::new(),
                plan_only: false,
                near_full_ingest: false,
                near_full_target_tokens: None,
                validate_best: true,
                confirm_best: false,
                goal: WorkloadGoal::Balanced,
                probe_mode: ProbeMode::Thinking,
            },
            TelemetrySource::Fixed(TelemetrySummary {
                peak_vram_mib: Some(4_096),
                min_free_vram_mib: Some(4_096),
                ram_available_min_mib: Some(16_384),
                swap_start_mib: Some(0),
                swap_end_mib: Some(0),
                swap_delta_mib: Some(0),
                sample_count: 1,
                ..TelemetrySummary::default()
            }),
        )
        .await
        .unwrap();
        unsafe {
            std::env::remove_var("LLAMA_SERVER");
        }

        assert!(
            recs.profiles
                .iter()
                .any(|profile| profile.id == "interactive-fast")
        );
        let result_path = read_metadata(&model_path)
            .unwrap()
            .profiler_dir()
            .join("runs")
            .join(&recs.profiles[0].source_run_id)
            .join("result.json");
        let result: ProfileResult =
            serde_json::from_str(&fs::read_to_string(result_path).unwrap()).unwrap();
        assert_eq!(result.test_kind, "realistic-validation");
        assert_eq!(result.validation_level, ValidationLevel::Realistic);
        assert!(
            result
                .realistic_validation
                .as_ref()
                .is_some_and(|validation| validation.incomplete_generation)
        );
        assert_eq!(result.requested_context, 8_192);
        assert_command_contains_context(&result.command, 8_192);
        assert_command_contains_high_thinking_defaults(&result.command);

        unsafe {
            std::env::set_var("LLAMA_SERVER", &server_path);
        }
        run_serve(
            &model_path,
            ServeOptions {
                profile: "interactive-fast".to_string(),
                port: 28_080,
                print_only: true,
                allow_stale: false,
            },
        )
        .await
        .unwrap();

        use std::io::Write;
        fs::OpenOptions::new()
            .append(true)
            .open(&model_path)
            .unwrap()
            .write_all(b"changed")
            .unwrap();
        let stale_error = run_serve(
            &model_path,
            ServeOptions {
                profile: "interactive-fast".to_string(),
                port: 28_080,
                print_only: true,
                allow_stale: false,
            },
        )
        .await
        .unwrap_err();
        assert!(
            stale_error
                .to_string()
                .contains("model at this path changed")
        );
        unsafe {
            std::env::remove_var("LLAMA_SERVER");
        }
    }

    fn test_candidate(id: &str, batch: u64, ubatch: u64, fit_target: u64) -> CandidateConfig {
        CandidateConfig {
            id: id.to_string(),
            requested_context: 4096,
            batch: Some(batch),
            ubatch: Some(ubatch),
            kv_cache: Some("q8_0".to_string()),
            kv_cache_v: None,
            fit_target: Some(fit_target),
            gpu_layers: None,
            cpu_moe: id.contains("cpu-moe"),
            n_cpu_moe: None,
            threads: None,
            threads_batch: None,
            expected_risk: crate::profile::CandidateRisk::Medium,
            note: String::new(),
            planning_note: String::new(),
        }
    }

    fn test_result(
        candidate: CandidateConfig,
        outcome: Outcome,
        free_vram: Option<u64>,
    ) -> ProfileResult {
        let gguf = GgufMetadata {
            path: PathBuf::from("/models/test.gguf"),
            file_name: "test.gguf".to_string(),
            file_size_bytes: 1,
            gguf_version: 3,
            tensor_count: 0,
            metadata_kv_count: 0,
            name: Some("test".to_string()),
            architecture: Some("llama".to_string()),
            size_label: None,
            native_context: Some(4096),
            block_count: Some(1),
            expert_count: None,
            expert_used_count: None,
            tokenizer_has_chat_template: false,
            quant: Some("Q4_K_M".to_string()),
            file_type: None,
            model_kind: crate::gguf::ModelKind::Dense,
            metadata: BTreeMap::new(),
        };
        ProfileResult {
            schema_version: SCHEMA_VERSION,
            run_id: "test".to_string(),
            started_at: Utc::now(),
            ended_at: Utc::now(),
            model_path: gguf.path.clone(),
            model_size_bytes: gguf.file_size_bytes,
            model_identity: Some(gguf.model_identity()),
            gguf: gguf.clone(),
            quant: gguf.quant.clone(),
            command: vec!["llama-server".to_string()],
            command_display: "llama-server".to_string(),
            candidate,
            test_kind: "tune".to_string(),
            requested_context: 4096,
            prompt_tokens: None,
            completion_tokens: None,
            metrics: crate::profile::RunMetrics {
                min_free_vram_mib: free_vram,
                ..crate::profile::RunMetrics::default()
            },
            probes: BTreeMap::new(),
            outcome,
            artifacts: ArtifactPaths {
                command: PathBuf::from("command.sh"),
                server_log: PathBuf::from("server.log"),
                telemetry_jsonl: PathBuf::from("telemetry.jsonl"),
                request_json: PathBuf::from("request.json"),
                response_json: PathBuf::from("response.json"),
                result_json: PathBuf::from("result.json"),
            },
            note: String::new(),
            environment: None,
            validation_level: ValidationLevel::Smoke,
            compatibility: crate::environment::Compatibility::Current,
            telemetry_status: TelemetryStatus::Measured,
            realistic_validation: None,
        }
    }

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn write_fake_llama_server(path: &Path) {
        let script = r#"#!/usr/bin/env python3
import argparse
import signal
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

if "--help" in sys.argv:
    print("--model -m --host --port -c --jinja -np --fit --fit-target -fa --reasoning --reasoning-budget --reasoning-budget-message --chat-template-kwargs --temp --top-p --top-k --min-p --presence-penalty --repeat-penalty -b -ub -ctk -ctv --cpu-moe --n-cpu-moe")
    raise SystemExit(0)
if "--version" in sys.argv:
    print("fake llama-server 1")
    raise SystemExit(0)

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--port", type=int, default=18180)
args, _ = parser.parse_known_args()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        if self.path in ("/health", "/v1/models"):
            body = b"{\"ok\":true}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        if length:
            self.rfile.read(length)
        body = b"data: {\"choices\":[{\"delta\":{\"content\":\"K\"}}]}\n\ndata: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        print("llama_perf_context_print:        prompt eval time =    100.00 ms /  4096 tokens (    0.02 ms per token, 40960.00 tokens per second)", file=sys.stderr, flush=True)
        print("llama_perf_context_print:               eval time =    100.00 ms /   128 runs   (    0.78 ms per token, 1280.00 tokens per second)", file=sys.stderr, flush=True)

server = HTTPServer(("127.0.0.1", args.port), Handler)
signal.signal(signal.SIGTERM, lambda *_: server.shutdown())
server.serve_forever()
"#;
        fs::write(path, script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut permissions = fs::metadata(path).unwrap().permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(path, permissions).unwrap();
        }
    }

    fn write_test_gguf(path: &Path, native_context: u32) {
        use std::io::Write;
        let mut file = fs::File::create(path).unwrap();
        file.write_all(b"GGUF").unwrap();
        file.write_all(&3u32.to_le_bytes()).unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap();
        file.write_all(&4u64.to_le_bytes()).unwrap();
        write_gguf_string(&mut file, "general.architecture", "llama");
        write_gguf_string(&mut file, "general.name", "Tiny Fake");
        write_gguf_u32(&mut file, "general.file_type", 15);
        write_gguf_u32(&mut file, "llama.context_length", native_context);
    }

    fn write_gguf_key(file: &mut fs::File, key: &str, kind: u32) {
        use std::io::Write;
        file.write_all(&(key.len() as u64).to_le_bytes()).unwrap();
        file.write_all(key.as_bytes()).unwrap();
        file.write_all(&kind.to_le_bytes()).unwrap();
    }

    fn write_gguf_string(file: &mut fs::File, key: &str, value: &str) {
        use std::io::Write;
        write_gguf_key(file, key, 8);
        file.write_all(&(value.len() as u64).to_le_bytes()).unwrap();
        file.write_all(value.as_bytes()).unwrap();
    }

    fn write_gguf_u32(file: &mut fs::File, key: &str, value: u32) {
        use std::io::Write;
        write_gguf_key(file, key, 4);
        file.write_all(&value.to_le_bytes()).unwrap();
    }

    fn assert_command_contains_context(command: &[String], context: u64) {
        let index = command
            .iter()
            .position(|part| part == "-c")
            .expect("command contains -c");
        assert_eq!(command[index + 1], context.to_string());
    }

    fn assert_command_contains_high_thinking_defaults(command: &[String]) {
        for (flag, value) in [
            ("--reasoning", "on"),
            ("--reasoning-budget", "4096"),
            ("--chat-template-kwargs", "{\"enable_thinking\":true}"),
            ("--temp", "0.6"),
            ("--top-p", "0.95"),
            ("--top-k", "20"),
            ("--min-p", "0.0"),
            ("--presence-penalty", "0.0"),
            ("--repeat-penalty", "1.0"),
            ("--reasoning-budget-message", THINKING_BUDGET_MESSAGE),
        ] {
            let has_pair = command
                .windows(2)
                .any(|window| window[0] == flag && window[1] == value);
            assert!(has_pair, "command missing {flag} {value}: {command:?}");
        }
    }
}
