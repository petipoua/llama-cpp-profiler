use crate::environment::{Compatibility, EnvironmentSnapshot, compare_environment};
use crate::gguf::{GgufMetadata, ModelKind};
use anyhow::{Result, bail};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::str::FromStr;

pub const SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Preset {
    Quick,
    Standard,
    Thorough,
}

impl Preset {
    pub fn max_candidates(self) -> usize {
        match self {
            Self::Quick => 6,
            Self::Standard => 16,
            Self::Thorough => 48,
        }
    }

    pub fn ingest_target_tokens(self) -> u64 {
        match self {
            Self::Quick => 16_000,
            Self::Standard | Self::Thorough => 64_000,
        }
    }
}

impl FromStr for Preset {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "quick" => Ok(Self::Quick),
            "standard" => Ok(Self::Standard),
            "thorough" => Ok(Self::Thorough),
            other => bail!("unknown preset {other:?}; expected quick, standard, or thorough"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyLimits {
    pub min_vram_free_mib: u64,
    pub max_swap_delta_mib: u64,
}

impl Default for SafetyLimits {
    fn default() -> Self {
        Self {
            min_vram_free_mib: 512,
            max_swap_delta_mib: 1024,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CandidateConfig {
    pub id: String,
    pub requested_context: u64,
    pub batch: Option<u64>,
    pub ubatch: Option<u64>,
    pub kv_cache: Option<String>,
    pub fit_target: Option<u64>,
    pub gpu_layers: Option<u64>,
    pub cpu_moe: bool,
    pub n_cpu_moe: Option<u64>,
    #[serde(default)]
    pub expected_risk: CandidateRisk,
    pub note: String,
    #[serde(default)]
    pub planning_note: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum CandidateRisk {
    Low,
    #[default]
    Medium,
    High,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ValidationLevel {
    #[default]
    Smoke,
    StandardIngest,
    Fullctx,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Pass,
    Oom,
    Timeout,
    ServerCrash,
    TooTight,
    ParsePartial,
    Interrupted,
}

impl Outcome {
    pub fn is_usable(&self) -> bool {
        matches!(self, Self::Pass | Self::ParsePartial)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct RunMetrics {
    pub server_prompt_eval_toks_per_s: Option<f64>,
    pub server_generation_toks_per_s: Option<f64>,
    pub client_ttft_ms: Option<u64>,
    pub total_wall_ms: Option<u64>,
    pub peak_vram_mib: Option<u64>,
    pub min_free_vram_mib: Option<u64>,
    pub gpu_util_avg_pct: Option<f64>,
    pub gpu_util_max_pct: Option<u64>,
    pub ram_available_min_mib: Option<u64>,
    pub swap_delta_mib: Option<i64>,
    pub process_rss_peak_mib: Option<u64>,
    pub cpu_util_avg_pct: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ProbeSummary {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub ttft_ms: Option<u64>,
    pub wall_ms: Option<u64>,
    pub response_excerpt: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ArtifactPaths {
    pub command: PathBuf,
    pub server_log: PathBuf,
    pub telemetry_jsonl: PathBuf,
    pub request_json: PathBuf,
    pub response_json: PathBuf,
    pub result_json: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProfileResult {
    pub schema_version: u32,
    pub run_id: String,
    pub started_at: DateTime<Utc>,
    pub ended_at: DateTime<Utc>,
    pub model_path: PathBuf,
    pub model_size_bytes: u64,
    pub gguf: GgufMetadata,
    pub quant: Option<String>,
    pub command: Vec<String>,
    pub command_display: String,
    pub candidate: CandidateConfig,
    pub test_kind: String,
    pub requested_context: u64,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub metrics: RunMetrics,
    pub probes: BTreeMap<String, ProbeSummary>,
    pub outcome: Outcome,
    pub artifacts: ArtifactPaths,
    pub note: String,
    #[serde(default)]
    pub environment: Option<EnvironmentSnapshot>,
    #[serde(default)]
    pub validation_level: ValidationLevel,
    #[serde(default)]
    pub compatibility: Compatibility,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Manifest {
    pub schema_version: u32,
    pub generated_at: DateTime<Utc>,
    pub model_path: PathBuf,
    pub gguf: GgufMetadata,
    pub runs: Vec<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RecommendationFile {
    pub schema_version: u32,
    pub generated_at: DateTime<Utc>,
    pub model_path: PathBuf,
    pub profiles: Vec<Recommendation>,
    pub rejected: Vec<RejectedRun>,
    #[serde(default)]
    pub stale: Vec<StaleRun>,
    #[serde(default)]
    pub environment: Option<EnvironmentSnapshot>,
    pub next_suggested_test: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Recommendation {
    pub id: String,
    pub role: String,
    pub source_run_id: String,
    #[serde(default)]
    pub source_candidate_id: String,
    #[serde(default)]
    pub source_test_kind: String,
    #[serde(default)]
    pub requested_context: u64,
    #[serde(default)]
    pub validated_prompt_tokens: Option<u64>,
    #[serde(default)]
    pub validation_level: ValidationLevel,
    #[serde(default)]
    pub compatibility: Compatibility,
    #[serde(default)]
    pub stale_reason: Option<String>,
    pub command: Vec<String>,
    pub command_display: String,
    pub output_toks_per_s: Option<f64>,
    pub prompt_toks_per_s: Option<f64>,
    pub ttft_ms: Option<u64>,
    pub peak_vram_mib: Option<u64>,
    pub headroom_mib: Option<u64>,
    pub risk: String,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RejectedRun {
    pub run_id: String,
    pub candidate_id: String,
    pub outcome: Outcome,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StaleRun {
    pub run_id: String,
    pub candidate_id: String,
    pub compatibility: Compatibility,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CandidatePlan {
    pub schema_version: u32,
    pub generated_at: DateTime<Utc>,
    pub model_path: PathBuf,
    pub requested_context: u64,
    pub preset: Preset,
    pub environment: EnvironmentSnapshot,
    pub candidates: Vec<CandidateConfig>,
}

pub fn generate_candidates(
    metadata: &GgufMetadata,
    preset: Preset,
    requested_context: u64,
    max_runs: Option<usize>,
) -> Vec<CandidateConfig> {
    generate_candidates_for_environment(metadata, preset, requested_context, max_runs, None)
}

pub fn generate_candidates_for_environment(
    metadata: &GgufMetadata,
    preset: Preset,
    requested_context: u64,
    max_runs: Option<usize>,
    environment: Option<&EnvironmentSnapshot>,
) -> Vec<CandidateConfig> {
    let cap = max_runs.unwrap_or_else(|| preset.max_candidates());
    let mut candidates = match metadata.model_kind {
        ModelKind::Moe => moe_candidates(metadata, requested_context),
        ModelKind::Dense | ModelKind::Unknown => dense_candidates(requested_context),
    };
    annotate_and_order_candidates(&mut candidates, metadata, environment);
    candidates.truncate(cap);
    candidates
}

pub fn build_recommendations(
    model_path: PathBuf,
    results: &[ProfileResult],
    safety: &SafetyLimits,
    current_environment: Option<&EnvironmentSnapshot>,
) -> RecommendationFile {
    let mut profiles = Vec::new();
    let usable: Vec<&ProfileResult> = results
        .iter()
        .filter(|result| result.outcome.is_usable())
        .filter(|result| passes_safety(result, safety))
        .filter(|result| run_compatibility(result, current_environment).is_current())
        .collect();

    if let Some(result) = best_by(&usable, generation_score) {
        profiles.push(to_recommendation(
            "interactive-fast",
            "Fastest observed generation throughput within safety limits",
            result,
            safety,
        ));
    }

    let safe: Vec<&ProfileResult> = usable
        .iter()
        .copied()
        .filter(|result| result.metrics.min_free_vram_mib.unwrap_or(0) >= 1024)
        .collect();
    if let Some(result) = best_by(&safe, generation_score) {
        profiles.push(to_recommendation(
            "interactive-safe",
            "Fastest observed generation throughput with at least 1 GiB free VRAM",
            result,
            safety,
        ));
    }

    if let Some(result) = best_by(&usable, prompt_score) {
        profiles.push(to_recommendation(
            "prompt-replay",
            "Fastest observed prompt ingest throughput within safety limits",
            result,
            safety,
        ));
    }

    if let Some(result) = best_by(&usable, balanced_score) {
        profiles.push(to_recommendation(
            "balanced",
            "Best observed balance of prompt and output throughput",
            result,
            safety,
        ));
    }

    if let Some(result) = best_by(&usable, quality_score) {
        profiles.push(to_recommendation(
            "quality-night",
            "Highest observed rough quant tier that starts and passes sanity",
            result,
            safety,
        ));
    }

    dedupe_profiles(&mut profiles);
    let rejected = results
        .iter()
        .filter(|result| {
            run_compatibility(result, current_environment).is_current()
                && (!result.outcome.is_usable()
                    || !passes_safety(result, safety)
                    || matches!(result.outcome, Outcome::TooTight))
        })
        .map(|result| RejectedRun {
            run_id: result.run_id.clone(),
            candidate_id: result.candidate.id.clone(),
            outcome: result.outcome.clone(),
            reason: rejection_reason(result, safety),
        })
        .collect();
    let stale = results
        .iter()
        .filter_map(|result| {
            let compatibility = run_compatibility(result, current_environment);
            (!compatibility.is_current()).then(|| StaleRun {
                run_id: result.run_id.clone(),
                candidate_id: result.candidate.id.clone(),
                compatibility,
                reason: compatibility.reason().to_string(),
            })
        })
        .collect();

    let next_suggested_test = if profiles.is_empty() {
        Some("Run `tune --preset quick` with a lower --ctx or safer MoE CPU offload.".to_string())
    } else if !results.iter().any(|result| result.test_kind == "fullctx") {
        Some(
            "If near-native context matters, run explicit `fullctx` for the chosen profile."
                .to_string(),
        )
    } else {
        None
    };

    RecommendationFile {
        schema_version: SCHEMA_VERSION,
        generated_at: Utc::now(),
        model_path,
        profiles,
        rejected,
        stale,
        environment: current_environment.cloned(),
        next_suggested_test,
    }
}

pub fn command_display(command: &[String]) -> String {
    command
        .iter()
        .map(|part| shell_escape(part))
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn quant_quality_score(quant: Option<&str>) -> f64 {
    let Some(quant) = quant else {
        return 0.0;
    };
    let quant = quant.to_ascii_uppercase();
    if quant.contains("F32") {
        100.0
    } else if quant.contains("BF16") || quant.contains("F16") {
        95.0
    } else if quant.contains("Q8") {
        88.0
    } else if quant.contains("Q6") {
        76.0
    } else if quant.contains("Q5") {
        66.0
    } else if quant.contains("IQ4") {
        58.0
    } else if quant.contains("Q4") {
        54.0
    } else if quant.contains("IQ3") {
        46.0
    } else if quant.contains("Q3") {
        42.0
    } else if quant.contains("IQ2") {
        34.0
    } else if quant.contains("Q2") {
        30.0
    } else {
        10.0
    }
}

fn dense_candidates(requested_context: u64) -> Vec<CandidateConfig> {
    let mut candidates = Vec::new();
    let priority = [
        ("q8_0", 1024, 256, 1536),
        ("q8_0", 4096, 1024, 1536),
        ("q8_0", 8192, 2048, 1536),
        ("q8_0", 16384, 4096, 1536),
        ("q4_0", 8192, 2048, 768),
        ("q4_0", 16384, 4096, 512),
        ("q4_0", 16384, 4096, 256),
        ("q4_0", 32768, 4096, 512),
    ];
    for (kv_cache, batch, ubatch, fit_target) in priority {
        candidates.push(CandidateConfig {
            id: format!("dense-{kv_cache}-fit{fit_target}-b{batch}-ub{ubatch}"),
            requested_context,
            batch: Some(batch),
            ubatch: Some(ubatch),
            kv_cache: Some(kv_cache.to_string()),
            fit_target: Some(fit_target),
            gpu_layers: None,
            cpu_moe: false,
            n_cpu_moe: None,
            expected_risk: CandidateRisk::Medium,
            note: "dense sweep over batch, ubatch, KV cache, and llama.cpp fit target".to_string(),
            planning_note: String::new(),
        });
    }

    let kv_types = ["q8_0", "q4_0"];
    let batches = [
        (1024, 256),
        (2048, 512),
        (4096, 1024),
        (8192, 2048),
        (16384, 4096),
    ];
    let fit_targets = [1536, 768, 512, 256];

    for kv_cache in kv_types {
        for (batch, ubatch) in batches {
            for fit_target in fit_targets {
                let id = format!("dense-{kv_cache}-fit{fit_target}-b{batch}-ub{ubatch}");
                if candidates.iter().any(|candidate| candidate.id == id) {
                    continue;
                }
                candidates.push(CandidateConfig {
                    id,
                    requested_context,
                    batch: Some(batch),
                    ubatch: Some(ubatch),
                    kv_cache: Some(kv_cache.to_string()),
                    fit_target: Some(fit_target),
                    gpu_layers: None,
                    cpu_moe: false,
                    n_cpu_moe: None,
                    expected_risk: CandidateRisk::Medium,
                    note: "dense sweep over batch, ubatch, KV cache, and llama.cpp fit target"
                        .to_string(),
                    planning_note: String::new(),
                });
            }
        }
    }
    candidates
}

fn moe_candidates(metadata: &GgufMetadata, requested_context: u64) -> Vec<CandidateConfig> {
    let mut candidates = Vec::new();
    for kv_cache in ["q8_0", "q4_0"] {
        for (batch, ubatch) in [(1024, 256), (8192, 2048), (16384, 4096)] {
            candidates.push(CandidateConfig {
                id: format!("moe-{kv_cache}-cpu-moe-b{batch}-ub{ubatch}"),
                requested_context,
                batch: Some(batch),
                ubatch: Some(ubatch),
                kv_cache: Some(kv_cache.to_string()),
                fit_target: Some(1536),
                gpu_layers: None,
                cpu_moe: true,
                n_cpu_moe: None,
                expected_risk: CandidateRisk::Low,
                note: "MoE baseline with CPU expert offload enabled".to_string(),
                planning_note: String::new(),
            });
        }
    }

    let mut n_cpu_moe_values = vec![44, 40, 36, 34, 33, 32, 31, 30, 28, 24];
    if let Some(expert_count) = metadata.expert_count {
        n_cpu_moe_values.extend([
            expert_count,
            expert_count.saturating_mul(3) / 4,
            expert_count / 2,
            expert_count / 4,
            0,
        ]);
    }
    if let Some(expert_count) = metadata.expert_count {
        n_cpu_moe_values.retain(|value| *value <= expert_count);
    }
    n_cpu_moe_values.sort_unstable_by(|left, right| right.cmp(left));
    n_cpu_moe_values.dedup();

    for kv_cache in ["q8_0", "q4_0"] {
        for (batch, ubatch) in [(16384, 4096), (32768, 4096), (16384, 2048), (8192, 2048)] {
            for n_cpu_moe in &n_cpu_moe_values {
                let id = format!("moe-{kv_cache}-ncpumoe{n_cpu_moe}-b{batch}-ub{ubatch}");
                if candidates.iter().any(|candidate| candidate.id == id) {
                    continue;
                }
                candidates.push(CandidateConfig {
                    id,
                    requested_context,
                    batch: Some(batch),
                    ubatch: Some(ubatch),
                    kv_cache: Some(kv_cache.to_string()),
                    fit_target: Some(1536),
                    gpu_layers: None,
                    cpu_moe: false,
                    n_cpu_moe: Some(*n_cpu_moe),
                    expected_risk: CandidateRisk::Medium,
                    note: "MoE sweep from safer CPU-heavy expert placement toward GPU residency"
                        .to_string(),
                    planning_note: String::new(),
                });
            }
        }
    }

    if candidates.is_empty() {
        candidates.push(CandidateConfig {
            id: "moe-cpu-moe".to_string(),
            requested_context,
            batch: Some(1024),
            ubatch: Some(256),
            kv_cache: Some("q8_0".to_string()),
            fit_target: Some(1536),
            gpu_layers: None,
            cpu_moe: true,
            n_cpu_moe: None,
            expected_risk: CandidateRisk::Low,
            note: "MoE baseline with CPU expert offload enabled".to_string(),
            planning_note: String::new(),
        });
    }
    candidates
}

fn annotate_and_order_candidates(
    candidates: &mut [CandidateConfig],
    metadata: &GgufMetadata,
    environment: Option<&EnvironmentSnapshot>,
) {
    let total_vram = environment.and_then(total_vram_mib);
    let model_mib = metadata.file_size_bytes / 1024 / 1024;
    for candidate in candidates.iter_mut() {
        let risk = candidate_risk(candidate, model_mib, total_vram);
        candidate.expected_risk = risk;
        candidate.planning_note = match (total_vram, risk) {
            (Some(vram), CandidateRisk::Low) => {
                format!(
                    "model is {model_mib} MiB against {vram} MiB total VRAM; safe-first candidate"
                )
            }
            (Some(vram), CandidateRisk::Medium) => {
                format!(
                    "model is {model_mib} MiB against {vram} MiB total VRAM; normal tuning candidate"
                )
            }
            (Some(vram), CandidateRisk::High) => {
                format!(
                    "model is {model_mib} MiB against {vram} MiB total VRAM; aggressive candidate"
                )
            }
            (None, _) => {
                "hardware VRAM unavailable; preserving conservative default order".to_string()
            }
        };
    }
    candidates.sort_by_key(|candidate| match candidate.expected_risk {
        CandidateRisk::Low => 0,
        CandidateRisk::Medium => 1,
        CandidateRisk::High => 2,
    });
}

fn candidate_risk(
    candidate: &CandidateConfig,
    model_mib: u64,
    total_vram_mib: Option<u64>,
) -> CandidateRisk {
    if candidate.cpu_moe {
        return CandidateRisk::Low;
    }
    let Some(total_vram_mib) = total_vram_mib else {
        return CandidateRisk::Medium;
    };
    let kv_risk_mib = match candidate.kv_cache.as_deref() {
        Some("q8_0") => candidate.requested_context / 128,
        Some("q4_0") => candidate.requested_context / 256,
        _ => candidate.requested_context / 192,
    };
    let working_set = model_mib.saturating_add(kv_risk_mib);
    if working_set.saturating_mul(100) < total_vram_mib.saturating_mul(70) {
        CandidateRisk::Low
    } else if working_set.saturating_mul(100) < total_vram_mib.saturating_mul(90) {
        CandidateRisk::Medium
    } else {
        CandidateRisk::High
    }
}

fn total_vram_mib(environment: &EnvironmentSnapshot) -> Option<u64> {
    let total = environment
        .gpus
        .iter()
        .filter_map(|gpu| gpu.total_vram_mib)
        .sum::<u64>();
    (total > 0).then_some(total)
}

fn passes_safety(result: &ProfileResult, safety: &SafetyLimits) -> bool {
    if matches!(result.outcome, Outcome::TooTight) {
        return false;
    }
    if let Some(free) = result.metrics.min_free_vram_mib
        && free < safety.min_vram_free_mib
    {
        return false;
    }
    if let Some(delta) = result.metrics.swap_delta_mib
        && delta > safety.max_swap_delta_mib as i64
    {
        return false;
    }
    true
}

fn best_by<'a>(
    results: &'a [&'a ProfileResult],
    score: impl Fn(&ProfileResult) -> f64,
) -> Option<&'a ProfileResult> {
    results.iter().copied().max_by(|left, right| {
        score(left)
            .partial_cmp(&score(right))
            .unwrap_or(Ordering::Equal)
    })
}

fn generation_score(result: &ProfileResult) -> f64 {
    result.metrics.server_generation_toks_per_s.unwrap_or(0.0)
}

fn prompt_score(result: &ProfileResult) -> f64 {
    result.metrics.server_prompt_eval_toks_per_s.unwrap_or(0.0)
}

fn balanced_score(result: &ProfileResult) -> f64 {
    let generation = generation_score(result);
    let prompt = prompt_score(result);
    if generation <= 0.0 || prompt <= 0.0 {
        return generation.max(prompt) * 0.5;
    }
    2.0 * generation * prompt / (generation + prompt)
}

fn quality_score(result: &ProfileResult) -> f64 {
    quant_quality_score(result.quant.as_deref()) + generation_score(result).ln_1p()
}

fn to_recommendation(
    id: &str,
    role: &str,
    result: &ProfileResult,
    safety: &SafetyLimits,
) -> Recommendation {
    Recommendation {
        id: id.to_string(),
        role: role.to_string(),
        source_run_id: result.run_id.clone(),
        source_candidate_id: result.candidate.id.clone(),
        source_test_kind: result.test_kind.clone(),
        requested_context: result.requested_context,
        validated_prompt_tokens: result.prompt_tokens,
        validation_level: result.validation_level,
        compatibility: result.compatibility,
        stale_reason: None,
        command: result.command.clone(),
        command_display: result.command_display.clone(),
        output_toks_per_s: result.metrics.server_generation_toks_per_s,
        prompt_toks_per_s: result.metrics.server_prompt_eval_toks_per_s,
        ttft_ms: result.metrics.client_ttft_ms,
        peak_vram_mib: result.metrics.peak_vram_mib,
        headroom_mib: result.metrics.min_free_vram_mib,
        risk: risk_label(result, safety),
        note: result.note.clone(),
    }
}

fn run_compatibility(
    result: &ProfileResult,
    current_environment: Option<&EnvironmentSnapshot>,
) -> Compatibility {
    current_environment.map_or(result.compatibility, |current| {
        compare_environment(result.environment.as_ref(), current)
    })
}

fn dedupe_profiles(profiles: &mut Vec<Recommendation>) {
    let mut seen = std::collections::BTreeSet::new();
    profiles.retain(|profile| {
        let key = (profile.id.clone(), profile.source_run_id.clone());
        seen.insert(key)
    });
}

fn risk_label(result: &ProfileResult, safety: &SafetyLimits) -> String {
    let free = result.metrics.min_free_vram_mib.unwrap_or(u64::MAX);
    let swap = result.metrics.swap_delta_mib.unwrap_or(0);
    if free < safety.min_vram_free_mib || swap > safety.max_swap_delta_mib as i64 {
        "high".to_string()
    } else if free < 1024 || swap > 0 {
        "medium".to_string()
    } else {
        "low".to_string()
    }
}

fn rejection_reason(result: &ProfileResult, safety: &SafetyLimits) -> String {
    if !result.outcome.is_usable() {
        if result.note.trim().is_empty() {
            return format!("{:?}", result.outcome);
        }
        return format!("{:?}: {}", result.outcome, compact_note(&result.note));
    }
    if let Some(free) = result.metrics.min_free_vram_mib
        && free < safety.min_vram_free_mib
    {
        return format!(
            "VRAM headroom {free} MiB is below {} MiB",
            safety.min_vram_free_mib
        );
    }
    if let Some(delta) = result.metrics.swap_delta_mib
        && delta > safety.max_swap_delta_mib as i64
    {
        return format!(
            "swap delta {delta} MiB exceeds {} MiB",
            safety.max_swap_delta_mib
        );
    }
    "too tight or rejected by safety policy".to_string()
}

fn compact_note(value: &str) -> String {
    let value = value.lines().next().unwrap_or(value).trim();
    if value.chars().count() > 120 {
        format!("{}...", value.chars().take(117).collect::<String>())
    } else {
        value.to_string()
    }
}

fn shell_escape(value: &str) -> String {
    if value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || "-_./:=,@".contains(ch))
    {
        value.to_string()
    } else {
        format!("'{}'", value.replace('\'', r"'\''"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufValue;

    #[test]
    fn scoring_selects_expected_profiles() {
        let gguf = fake_metadata(Some("Q4_K_M"));
        let results = vec![
            fake_result("slow-safe", &gguf, 15.0, 900.0, Some(2048), Outcome::Pass),
            fake_result("fast-tight", &gguf, 40.0, 400.0, Some(128), Outcome::Pass),
            fake_result("oom", &gguf, 0.0, 0.0, None, Outcome::Oom),
        ];
        let recs = build_recommendations(
            PathBuf::from("/models/test.gguf"),
            &results,
            &SafetyLimits::default(),
            Some(&fake_environment()),
        );

        let fast = recs
            .profiles
            .iter()
            .find(|profile| profile.id == "interactive-fast")
            .unwrap();
        assert_eq!(fast.source_run_id, "slow-safe");
        assert!(recs.rejected.iter().any(|run| run.run_id == "fast-tight"));
        assert!(recs.rejected.iter().any(|run| run.run_id == "oom"));
    }

    #[test]
    fn generates_cpu_heavy_moe_candidates_first() {
        let mut metadata = fake_metadata(Some("Q4_K_M"));
        metadata.model_kind = ModelKind::Moe;
        metadata.expert_count = Some(16);
        let candidates = generate_candidates(&metadata, Preset::Standard, 262_144, Some(8));
        assert!(candidates[0].cpu_moe);
        assert!(candidates.iter().any(|candidate| candidate.cpu_moe));
        assert!(
            candidates
                .iter()
                .any(|candidate| candidate.n_cpu_moe == Some(16))
        );
    }

    fn fake_metadata(quant: Option<&str>) -> GgufMetadata {
        GgufMetadata {
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
            quant: quant.map(str::to_string),
            file_type: None,
            model_kind: ModelKind::Dense,
            metadata: BTreeMap::<String, GgufValue>::new(),
        }
    }

    fn fake_result(
        run_id: &str,
        gguf: &GgufMetadata,
        generation: f64,
        prompt: f64,
        free_vram: Option<u64>,
        outcome: Outcome,
    ) -> ProfileResult {
        ProfileResult {
            schema_version: SCHEMA_VERSION,
            run_id: run_id.to_string(),
            started_at: Utc::now(),
            ended_at: Utc::now(),
            model_path: gguf.path.clone(),
            model_size_bytes: gguf.file_size_bytes,
            gguf: gguf.clone(),
            quant: gguf.quant.clone(),
            command: vec!["llama-server".to_string()],
            command_display: "llama-server".to_string(),
            candidate: CandidateConfig {
                id: run_id.to_string(),
                requested_context: 4096,
                batch: None,
                ubatch: None,
                kv_cache: None,
                fit_target: None,
                gpu_layers: None,
                cpu_moe: false,
                n_cpu_moe: None,
                expected_risk: CandidateRisk::Medium,
                note: String::new(),
                planning_note: String::new(),
            },
            test_kind: "tune".to_string(),
            requested_context: 4096,
            prompt_tokens: Some(100),
            completion_tokens: Some(100),
            metrics: RunMetrics {
                server_generation_toks_per_s: Some(generation),
                server_prompt_eval_toks_per_s: Some(prompt),
                min_free_vram_mib: free_vram,
                ..RunMetrics::default()
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
            environment: Some(fake_environment()),
            validation_level: ValidationLevel::StandardIngest,
            compatibility: crate::environment::Compatibility::Current,
        }
    }

    #[test]
    fn stale_runs_are_excluded_from_recommendations() {
        let gguf = fake_metadata(Some("Q4_K_M"));
        let current = fake_environment();
        let mut stale_environment = current.clone();
        stale_environment.llama_server.help_hash = Some("changed".to_string());
        let mut stale = fake_result("stale-fast", &gguf, 100.0, 100.0, Some(4096), Outcome::Pass);
        stale.environment = Some(stale_environment);
        let current_run = fake_result(
            "current-slow",
            &gguf,
            10.0,
            100.0,
            Some(4096),
            Outcome::Pass,
        );
        let recs = build_recommendations(
            PathBuf::from("/models/test.gguf"),
            &[stale, current_run],
            &SafetyLimits::default(),
            Some(&current),
        );
        let fast = recs
            .profiles
            .iter()
            .find(|profile| profile.id == "interactive-fast")
            .unwrap();
        assert_eq!(fast.source_run_id, "current-slow");
        assert_eq!(recs.stale.len(), 1);
        assert_eq!(recs.stale[0].run_id, "stale-fast");
    }

    #[test]
    fn v1_result_deserializes_as_legacy_stale() {
        let json = serde_json::json!({
            "schema_version": 1,
            "run_id": "legacy",
            "started_at": Utc::now(),
            "ended_at": Utc::now(),
            "model_path": "/models/test.gguf",
            "model_size_bytes": 1,
            "gguf": fake_metadata(Some("Q4_K_M")),
            "quant": "Q4_K_M",
            "command": ["llama-server"],
            "command_display": "llama-server",
            "candidate": {
                "id": "legacy",
                "requested_context": 4096,
                "batch": null,
                "ubatch": null,
                "kv_cache": null,
                "fit_target": null,
                "gpu_layers": null,
                "cpu_moe": false,
                "n_cpu_moe": null,
                "note": ""
            },
            "test_kind": "tune",
            "requested_context": 4096,
            "prompt_tokens": 100,
            "completion_tokens": 100,
            "metrics": {},
            "probes": {},
            "outcome": "pass",
            "artifacts": {
                "command": "command.sh",
                "server_log": "server.log",
                "telemetry_jsonl": "telemetry.jsonl",
                "request_json": "request.json",
                "response_json": "response.json",
                "result_json": "result.json"
            },
            "note": ""
        });
        let result: ProfileResult = serde_json::from_value(json).unwrap();
        assert_eq!(
            result.compatibility,
            crate::environment::Compatibility::LegacyMissingSnapshot
        );
        assert_eq!(result.validation_level, ValidationLevel::Smoke);
        assert!(result.environment.is_none());
    }

    fn fake_environment() -> EnvironmentSnapshot {
        EnvironmentSnapshot {
            llama_server: crate::environment::ServerInfo {
                executable: "llama-server".to_string(),
                resolved_path: Some(PathBuf::from("/usr/bin/llama-server")),
                version: Some("test".to_string()),
                help_hash: Some("help".to_string()),
                usable: true,
                error: None,
            },
            ..EnvironmentSnapshot::default()
        }
    }
}
