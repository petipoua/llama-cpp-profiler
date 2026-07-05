use crate::gguf::{GgufMetadata, discover_models, format_bytes};
use crate::profile::{Recommendation, RecommendationFile};
use crate::runner::{collect_profiled_models, model_key_for_opencode, model_label_for_opencode};
use anyhow::{Context, Result, bail};
use comfy_table::{Cell, Table, presets::UTF8_FULL_CONDENSED};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::fs;
use std::path::{Path, PathBuf};

const MANAGED_START: &str = "<!-- llama-cpp-profiler:start -->";
const MANAGED_END: &str = "<!-- llama-cpp-profiler:end -->";

#[derive(Debug, Clone)]
pub struct ReportOptions {
    pub agent: bool,
    pub include_stale: bool,
}

#[derive(Debug, Clone)]
pub struct ExportOptions {
    pub markdown: bool,
    pub opencode: Option<PathBuf>,
    pub dry_run: bool,
    pub write: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AgentReport {
    schema_version: u32,
    best_profile_ids: Vec<String>,
    exact_command: Option<String>,
    confidence: Option<String>,
    key_metrics: Vec<AgentMetric>,
    failures: Vec<String>,
    stale_profiles: Vec<String>,
    next_suggested_test: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AgentMetric {
    model_path: PathBuf,
    model_kind: crate::gguf::ModelKind,
    quant: Option<String>,
    native_context: Option<u64>,
    profile_id: String,
    source_run_id: String,
    source_candidate_id: String,
    source_test_kind: String,
    requested_context: u64,
    validated_prompt_tokens: Option<u64>,
    validation_level: crate::profile::ValidationLevel,
    output_toks_per_s: Option<f64>,
    prompt_toks_per_s: Option<f64>,
    ttft_ms: Option<u64>,
    headroom_mib: Option<u64>,
    risk: String,
    confidence: String,
    why: String,
    compatibility: crate::environment::Compatibility,
    exact_command: String,
}

pub fn print_report(root: &Path, options: ReportOptions) -> Result<()> {
    let profiled = collect_profiled_models(root)?;
    if options.agent {
        print_agent_report(&profiled)?;
        return Ok(());
    }

    if profiled.is_empty() {
        println!("No profiler recommendations found under {}", root.display());
        return Ok(());
    }

    let mut table = comparison_table();
    for (metadata, recs) in &profiled {
        for profile in &recs.profiles {
            table.add_row(profile_row(metadata, profile));
        }
        if options.include_stale {
            for stale in &recs.stale {
                table.add_row(vec![
                    Cell::new(&metadata.file_name),
                    Cell::new(format!("stale:{}", stale.candidate_id)),
                    Cell::new(stale.reason.clone()),
                    Cell::new("-"),
                    Cell::new("-"),
                    Cell::new("-"),
                    Cell::new("-"),
                    Cell::new("-"),
                    Cell::new(format!("{:?}", stale.compatibility).to_ascii_lowercase()),
                    Cell::new("stale"),
                    Cell::new("-"),
                ]);
            }
        }
    }
    println!("{table}");
    Ok(())
}

pub fn write_latest_markdown(
    profiler_dir: &Path,
    recommendations: &RecommendationFile,
) -> Result<PathBuf> {
    let reports_dir = profiler_dir.join("reports");
    fs::create_dir_all(&reports_dir)?;
    let path = reports_dir.join("latest.md");
    fs::write(&path, markdown_for_recommendations(recommendations))?;
    Ok(path)
}

pub fn export(root: &Path, options: ExportOptions) -> Result<()> {
    let profiled = collect_profiled_models(root)?;
    if profiled.is_empty() {
        bail!("no profiler recommendations found under {}", root.display());
    }

    if options.markdown {
        export_markdown(root, &profiled, options.dry_run || !options.write)?;
    }
    if let Some(path) = &options.opencode {
        export_opencode(path, &profiled, options.dry_run || !options.write)?;
    }
    if !options.markdown && options.opencode.is_none() {
        println!("{}", markdown_block(&profiled));
    }
    Ok(())
}

pub fn inspect_model(path: &Path) -> Result<Value> {
    let model_path = crate::gguf::resolve_model_path(path)?;
    let metadata = crate::gguf::read_metadata(&model_path)?;
    let rec_path = metadata.profiler_dir().join("recommendations.json");
    let prior_runs = metadata.profiler_dir().join("runs");
    let run_count = if prior_runs.exists() {
        fs::read_dir(prior_runs)?
            .filter_map(|entry| entry.ok())
            .count()
    } else {
        0
    };
    let recommendations = if rec_path.exists() {
        let data = fs::read_to_string(&rec_path)?;
        Some(serde_json::from_str::<RecommendationFile>(&data)?)
    } else {
        None
    };
    Ok(json!({
        "path": metadata.path,
        "file_name": metadata.file_name,
        "file_size_bytes": metadata.file_size_bytes,
        "file_size": format_bytes(metadata.file_size_bytes),
        "name": metadata.name,
        "architecture": metadata.architecture,
        "kind": metadata.model_kind,
        "quant": metadata.quant,
        "native_context": metadata.native_context,
        "block_count": metadata.block_count,
        "expert_count": metadata.expert_count,
        "expert_used_count": metadata.expert_used_count,
        "tokenizer_has_chat_template": metadata.tokenizer_has_chat_template,
        "prior_run_count": run_count,
        "recommendations": recommendations,
    }))
}

pub fn print_scan_table(root: &Path) -> Result<()> {
    let models = discover_models(root)?;
    if models.is_empty() {
        println!("No model GGUF files found under {}", root.display());
        return Ok(());
    }
    let mut table = Table::new();
    table.load_preset(UTF8_FULL_CONDENSED);
    table.set_header(["Model", "Size", "Arch", "Kind", "Quant", "Native ctx"]);
    for model in models {
        table.add_row([
            Cell::new(model.path.display()),
            Cell::new(format_bytes(model.file_size_bytes)),
            Cell::new(model.architecture.unwrap_or_else(|| "-".to_string())),
            Cell::new(format!("{:?}", model.model_kind).to_ascii_lowercase()),
            Cell::new(model.quant.unwrap_or_else(|| "-".to_string())),
            Cell::new(
                model
                    .native_context
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_string()),
            ),
        ]);
    }
    println!("{table}");
    Ok(())
}

pub fn update_managed_block(existing: &str, block_body: &str) -> String {
    let block = format!("{MANAGED_START}\n{block_body}\n{MANAGED_END}");
    match (existing.find(MANAGED_START), existing.find(MANAGED_END)) {
        (Some(start), Some(end)) if start <= end => {
            let end = end + MANAGED_END.len();
            let mut output = String::new();
            output.push_str(existing[..start].trim_end());
            output.push_str("\n\n");
            output.push_str(&block);
            output.push_str("\n\n");
            output.push_str(existing[end..].trim_start());
            output
        }
        _ if existing.trim().is_empty() => format!("{block}\n"),
        _ => format!("{}\n\n{block}\n", existing.trim_end()),
    }
}

pub fn update_opencode_json(existing: &str, entries: &[(String, String, u64)]) -> Result<String> {
    let mut value: Value = serde_json::from_str(existing)?;
    if !value.get("provider").is_some_and(Value::is_object) {
        value["provider"] = json!({});
    }
    if !value["provider"]
        .get("llamacpp")
        .is_some_and(Value::is_object)
    {
        value["provider"]["llamacpp"] = json!({
            "npm": "@ai-sdk/openai-compatible",
            "name": "llama.cpp local",
            "options": {
                "baseURL": "http://127.0.0.1:18080/v1",
                "apiKey": "local"
            },
            "models": {}
        });
    }
    if !value["provider"]["llamacpp"]
        .get("models")
        .is_some_and(Value::is_object)
    {
        value["provider"]["llamacpp"]["models"] = json!({});
    }

    let models = value["provider"]["llamacpp"]["models"]
        .as_object_mut()
        .context("provider.llamacpp.models is not an object")?;
    for (key, label, context) in entries {
        models.insert(
            key.clone(),
            json!({
                "name": label,
                "limit": {
                    "context": context,
                    "output": 8192
                }
            }),
        );
    }
    Ok(format!("{}\n", serde_json::to_string_pretty(&value)?))
}

pub fn markdown_for_recommendations(recommendations: &RecommendationFile) -> String {
    let mut lines = Vec::new();
    lines.push("# llama.cpp profiler recommendations".to_string());
    lines.push(String::new());
    lines.push(format!("Model: `{}`", recommendations.model_path.display()));
    lines.push(String::new());
    lines.push(comparison_markdown_table(
        recommendations.profiles.iter().map(|profile| {
            (
                profile.id.as_str(),
                profile.role.as_str(),
                profile.output_toks_per_s,
                profile.prompt_toks_per_s,
                profile.ttft_ms,
                profile.peak_vram_mib,
                profile.headroom_mib,
                profile.risk.as_str(),
                validation_label(profile.validation_level),
                profile.command_display.as_str(),
            )
        }),
    ));
    if !recommendations.rejected.is_empty() {
        lines.push(String::new());
        lines.push("## Rejected".to_string());
        lines.push(String::new());
        for rejected in &recommendations.rejected {
            lines.push(format!(
                "- `{}` / `{}`: {:?} - {}",
                rejected.run_id, rejected.candidate_id, rejected.outcome, rejected.reason
            ));
        }
    }
    if !recommendations.stale.is_empty() {
        lines.push(String::new());
        lines.push("## Stale".to_string());
        lines.push(String::new());
        for stale in &recommendations.stale {
            lines.push(format!(
                "- `{}` / `{}`: {:?} - {}",
                stale.run_id, stale.candidate_id, stale.compatibility, stale.reason
            ));
        }
    }
    if let Some(next) = &recommendations.next_suggested_test {
        lines.push(String::new());
        lines.push(format!("Next suggested test: {next}"));
    }
    lines.push(String::new());
    lines.join("\n")
}

fn print_agent_report(profiled: &[(GgufMetadata, RecommendationFile)]) -> Result<()> {
    let mut all_profiles = Vec::new();
    let mut failures = Vec::new();
    let mut stale_profiles = Vec::new();
    let mut next = None;
    for (metadata, recs) in profiled {
        if next.is_none() {
            next = recs.next_suggested_test.clone();
        }
        for rejected in &recs.rejected {
            failures.push(format!(
                "{} {} {:?}: {}",
                metadata.file_name, rejected.candidate_id, rejected.outcome, rejected.reason
            ));
        }
        for stale in &recs.stale {
            stale_profiles.push(format!(
                "{} {} {:?}: {}",
                metadata.file_name, stale.candidate_id, stale.compatibility, stale.reason
            ));
        }
        for profile in &recs.profiles {
            all_profiles.push((metadata, profile));
        }
    }

    all_profiles.sort_by(|(_, left), (_, right)| {
        right
            .output_toks_per_s
            .unwrap_or(0.0)
            .partial_cmp(&left.output_toks_per_s.unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let best_profile_ids = all_profiles
        .iter()
        .take(5)
        .map(|(_, profile)| profile.id.clone())
        .collect::<Vec<_>>();
    let exact_command = all_profiles
        .first()
        .map(|(_, profile)| profile.command_display.clone());
    let confidence = all_profiles
        .first()
        .map(|(_, profile)| confidence_label(profile).to_string());
    let key_metrics = all_profiles
        .iter()
        .take(8)
        .map(|(metadata, profile)| AgentMetric {
            model_path: metadata.path.clone(),
            model_kind: metadata.model_kind.clone(),
            quant: metadata.quant.clone(),
            native_context: metadata.native_context,
            profile_id: profile.id.clone(),
            source_run_id: profile.source_run_id.clone(),
            source_candidate_id: profile.source_candidate_id.clone(),
            source_test_kind: profile.source_test_kind.clone(),
            requested_context: profile.requested_context,
            validated_prompt_tokens: profile.validated_prompt_tokens,
            validation_level: profile.validation_level,
            output_toks_per_s: profile.output_toks_per_s,
            prompt_toks_per_s: profile.prompt_toks_per_s,
            ttft_ms: profile.ttft_ms,
            headroom_mib: profile.headroom_mib,
            risk: profile.risk.clone(),
            confidence: confidence_label(profile).to_string(),
            why: agent_why(profile),
            compatibility: profile.compatibility,
            exact_command: profile.command_display.clone(),
        })
        .collect();
    let report = AgentReport {
        schema_version: crate::profile::SCHEMA_VERSION,
        best_profile_ids,
        exact_command,
        confidence,
        key_metrics,
        failures: failures.into_iter().take(12).collect(),
        stale_profiles: stale_profiles.into_iter().take(12).collect(),
        next_suggested_test: next,
    };
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}

fn confidence_label(profile: &Recommendation) -> &'static str {
    match (profile.validation_level, profile.risk.as_str()) {
        (crate::profile::ValidationLevel::Fullctx, "low" | "medium") => "high",
        (crate::profile::ValidationLevel::StandardIngest, "low" | "medium") => "medium",
        (crate::profile::ValidationLevel::Smoke, "low" | "medium") => "low",
        _ => "low",
    }
}

fn agent_why(profile: &Recommendation) -> String {
    format!(
        "{}; out {}; prompt {}; headroom {}",
        profile.id,
        fmt_f64(profile.output_toks_per_s),
        fmt_f64(profile.prompt_toks_per_s),
        fmt_mib(profile.headroom_mib)
    )
}

fn export_markdown(
    root: &Path,
    profiled: &[(GgufMetadata, RecommendationFile)],
    dry_run: bool,
) -> Result<()> {
    let path = root.join("RUNNING_NOTES.md");
    let existing = fs::read_to_string(&path).unwrap_or_default();
    let updated = update_managed_block(&existing, &markdown_block(profiled));
    if dry_run {
        println!("{updated}");
    } else {
        fs::write(&path, updated)?;
        println!("updated {}", path.display());
    }
    Ok(())
}

fn export_opencode(
    path: &Path,
    profiled: &[(GgufMetadata, RecommendationFile)],
    dry_run: bool,
) -> Result<()> {
    let mut entries = Vec::new();
    for (metadata, recs) in profiled {
        for profile in &recs.profiles {
            entries.push((
                model_key_for_opencode(metadata, &profile.id),
                model_label_for_opencode(metadata, &profile.id),
                metadata.native_context.unwrap_or(262_144),
            ));
        }
    }
    let existing = fs::read_to_string(path).unwrap_or_else(|_| "{}".to_string());
    let updated = update_opencode_json(&existing, &entries)?;
    if dry_run {
        println!("{updated}");
    } else {
        fs::write(path, updated)?;
        println!("updated {}", path.display());
    }
    Ok(())
}

fn markdown_block(profiled: &[(GgufMetadata, RecommendationFile)]) -> String {
    let mut rows = Vec::new();
    for (metadata, recs) in profiled {
        for profile in &recs.profiles {
            rows.push((
                profile.id.as_str(),
                profile.role.as_str(),
                profile.output_toks_per_s,
                profile.prompt_toks_per_s,
                profile.ttft_ms,
                profile.peak_vram_mib,
                profile.headroom_mib,
                profile.risk.as_str(),
                validation_label(profile.validation_level),
                profile.command_display.as_str(),
                metadata.file_name.as_str(),
            ));
        }
    }
    rows.sort_by(|left, right| {
        right
            .2
            .unwrap_or(0.0)
            .partial_cmp(&left.2.unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut lines = vec![
        "## llama.cpp profiler".to_string(),
        String::new(),
        "| Model | Profile | Role | Output tok/s | Prompt tok/s | TTFT | Peak VRAM | Headroom | Risk | Validation | Command |".to_string(),
        "|---|---|---|---:|---:|---:|---:|---:|---|---|---|".to_string(),
    ];
    for row in rows {
        lines.push(format!(
            "| `{}` | `{}` | {} | {} | {} | {} | {} | {} | {} | {} | `{}` |",
            row.10,
            row.0,
            escape_md(row.1),
            fmt_f64(row.2),
            fmt_f64(row.3),
            fmt_ms(row.4),
            fmt_mib(row.5),
            fmt_mib(row.6),
            row.7,
            row.8,
            row.9.replace('|', "\\|"),
        ));
    }
    lines.join("\n")
}

fn comparison_markdown_table<'a>(
    rows: impl Iterator<
        Item = (
            &'a str,
            &'a str,
            Option<f64>,
            Option<f64>,
            Option<u64>,
            Option<u64>,
            Option<u64>,
            &'a str,
            &'a str,
            &'a str,
        ),
    >,
) -> String {
    let mut lines = Vec::new();
    lines.push("| Profile | Role | Output tok/s | Prompt tok/s | TTFT | Peak VRAM | Headroom | Risk | Validation | Command |".to_string());
    lines.push("|---|---|---:|---:|---:|---:|---:|---|---|---|".to_string());
    for (id, role, output, prompt, ttft, peak, headroom, risk, validation, command) in rows {
        lines.push(format!(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | `{}` |",
            id,
            escape_md(role),
            fmt_f64(output),
            fmt_f64(prompt),
            fmt_ms(ttft),
            fmt_mib(peak),
            fmt_mib(headroom),
            risk,
            validation,
            command.replace('|', "\\|"),
        ));
    }
    lines.join("\n")
}

fn comparison_table() -> Table {
    let mut table = Table::new();
    table.load_preset(UTF8_FULL_CONDENSED);
    table.set_header([
        "Model",
        "Profile",
        "Role",
        "Out tok/s",
        "Prompt tok/s",
        "TTFT",
        "Peak VRAM",
        "Headroom",
        "Risk",
        "Validation",
        "Command",
    ]);
    table
}

fn profile_row(metadata: &GgufMetadata, profile: &Recommendation) -> Vec<Cell> {
    vec![
        Cell::new(&metadata.file_name),
        Cell::new(&profile.id),
        Cell::new(&profile.role),
        Cell::new(fmt_f64(profile.output_toks_per_s)),
        Cell::new(fmt_f64(profile.prompt_toks_per_s)),
        Cell::new(fmt_ms(profile.ttft_ms)),
        Cell::new(fmt_mib(profile.peak_vram_mib)),
        Cell::new(fmt_mib(profile.headroom_mib)),
        Cell::new(&profile.risk),
        Cell::new(validation_label(profile.validation_level)),
        Cell::new(&profile.command_display),
    ]
}

fn validation_label(value: crate::profile::ValidationLevel) -> &'static str {
    match value {
        crate::profile::ValidationLevel::Smoke => "smoke",
        crate::profile::ValidationLevel::StandardIngest => "standard-ingest",
        crate::profile::ValidationLevel::Fullctx => "fullctx",
    }
}

fn fmt_f64(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| "-".to_string())
}

fn fmt_ms(value: Option<u64>) -> String {
    value
        .map(|value| {
            if value >= 60_000 {
                format!("{:.2}m", value as f64 / 60_000.0)
            } else {
                format!("{value} ms")
            }
        })
        .unwrap_or_else(|| "-".to_string())
}

fn fmt_mib(value: Option<u64>) -> String {
    value
        .map(|value| format!("{value} MiB"))
        .unwrap_or_else(|| "-".to_string())
}

fn escape_md(value: &str) -> String {
    value.replace('|', "\\|")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn managed_block_update_preserves_surrounding_text() {
        let existing = "before\n\n<!-- llama-cpp-profiler:start -->\nold\n<!-- llama-cpp-profiler:end -->\n\nafter\n";
        let updated = update_managed_block(existing, "new");
        assert!(updated.contains("before"));
        assert!(updated.contains("new"));
        assert!(!updated.contains("\nold\n"));
        assert!(updated.contains("after"));
    }

    #[test]
    fn opencode_update_preserves_provider_config() {
        let existing = r#"{
  "provider": {
    "llamacpp": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llama.cpp local",
      "options": {"baseURL": "http://127.0.0.1:18080/v1", "apiKey": "local"},
      "models": {"existing": {"name": "Existing"}}
    }
  },
  "permission": {"*": "allow"}
}"#;
        let updated = update_opencode_json(
            existing,
            &[("new-model".into(), "New Model".into(), 262144)],
        )
        .unwrap();
        let value: Value = serde_json::from_str(&updated).unwrap();
        assert_eq!(
            value
                .pointer("/provider/llamacpp/models/existing/name")
                .and_then(Value::as_str),
            Some("Existing")
        );
        assert_eq!(
            value
                .pointer("/provider/llamacpp/models/new-model/limit/context")
                .and_then(Value::as_u64),
            Some(262144)
        );
        assert_eq!(
            value.pointer("/permission/*").and_then(Value::as_str),
            Some("allow")
        );
    }
}
