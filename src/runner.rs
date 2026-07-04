use crate::gguf::{GgufMetadata, read_metadata, resolve_model_path};
use crate::profile::{
    ArtifactPaths, CandidateConfig, Manifest, Outcome, Preset, ProbeSummary, ProfileResult,
    RecommendationFile, SCHEMA_VERSION, SafetyLimits, build_recommendations, command_display,
    generate_candidates,
};
use crate::report;
use crate::telemetry::TelemetrySampler;
use anyhow::{Context, Result, anyhow, bail};
use chrono::{DateTime, Utc};
use futures_util::StreamExt;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
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

#[derive(Debug, Clone)]
pub struct TuneOptions {
    pub ctx_cap: Option<u64>,
    pub preset: Preset,
    pub max_runs: Option<usize>,
    pub safety: SafetyLimits,
    pub port_start: u16,
    pub gpu_index: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct FullCtxOptions {
    pub profile: String,
    pub target_tokens: u64,
    pub ctx_cap: Option<u64>,
    pub safety: SafetyLimits,
    pub port_start: u16,
    pub gpu_index: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct ServeOptions {
    pub profile: String,
    pub port: u16,
    pub print_only: bool,
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
}

#[derive(Debug, Clone)]
struct RunRequest {
    test_kind: String,
    candidate: CandidateConfig,
    port: u16,
    prompt_plan: PromptPlan,
}

#[derive(Debug, Clone)]
enum PromptPlan {
    Tune { ingest_target_tokens: u64 },
    FullCtx { target_tokens: u64 },
}

pub async fn run_tune(path: &Path, options: TuneOptions) -> Result<RecommendationFile> {
    let model_path = resolve_model_path(path)?;
    let metadata = read_metadata(&model_path)?;
    let requested_context = metadata.context_or(options.ctx_cap);
    let capabilities = ServerCapabilities::detect()?;
    let candidates = generate_candidates(
        &metadata,
        options.preset,
        requested_context,
        options.max_runs,
    );
    if candidates.is_empty() {
        bail!("no candidates generated for {}", metadata.path.display());
    }

    let profiler_dir = metadata.profiler_dir();
    fs::create_dir_all(profiler_dir.join("runs"))?;
    fs::create_dir_all(profiler_dir.join("reports"))?;

    let total_candidates = candidates.len();
    for (index, candidate) in candidates.into_iter().enumerate() {
        let port = find_open_port(options.port_start + index as u16)?;
        eprintln!(
            "[{}/{}] {} on port {}",
            index + 1,
            total_candidates,
            candidate.id,
            port
        );
        let request = RunRequest {
            test_kind: "tune".to_string(),
            candidate,
            port,
            prompt_plan: PromptPlan::Tune {
                ingest_target_tokens: options.preset.ingest_target_tokens(),
            },
        };
        let result = run_candidate(
            &metadata,
            &capabilities,
            request,
            &options.safety,
            options.gpu_index,
        )
        .await?;
        eprintln!(
            "  -> {:?}, output {:?} tok/s, prompt {:?} tok/s, headroom {:?} MiB",
            result.outcome,
            result.metrics.server_generation_toks_per_s,
            result.metrics.server_prompt_eval_toks_per_s,
            result.metrics.min_free_vram_mib
        );
    }

    write_manifest(&metadata)?;
    let all_results = load_prior_results(&metadata.profiler_dir())?;
    let recommendations =
        build_recommendations(metadata.path.clone(), &all_results, &options.safety);
    write_json(
        metadata.profiler_dir().join("recommendations.json"),
        &recommendations,
    )?;
    report::write_latest_markdown(&metadata.profiler_dir(), &recommendations)?;
    Ok(recommendations)
}

pub async fn run_fullctx(path: &Path, options: FullCtxOptions) -> Result<ProfileResult> {
    let model_path = resolve_model_path(path)?;
    let metadata = read_metadata(&model_path)?;
    let capabilities = ServerCapabilities::detect()?;
    let candidate =
        candidate_for_saved_profile(&metadata, &options.profile)?.unwrap_or_else(|| {
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
                    fit_target: Some(1536),
                    gpu_layers: None,
                    cpu_moe: false,
                    n_cpu_moe: None,
                    note: "fallback full-context candidate".to_string(),
                })
        });

    fs::create_dir_all(metadata.profiler_dir().join("runs"))?;
    fs::create_dir_all(metadata.profiler_dir().join("reports"))?;
    let request = RunRequest {
        test_kind: "fullctx".to_string(),
        candidate,
        port: find_open_port(options.port_start)?,
        prompt_plan: PromptPlan::FullCtx {
            target_tokens: options.target_tokens,
        },
    };
    let result = run_candidate(
        &metadata,
        &capabilities,
        request,
        &options.safety,
        options.gpu_index,
    )
    .await?;
    write_manifest(&metadata)?;
    let results = load_prior_results(&metadata.profiler_dir())?;
    let recommendations = build_recommendations(metadata.path.clone(), &results, &options.safety);
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

async fn run_candidate(
    metadata: &GgufMetadata,
    capabilities: &ServerCapabilities,
    request: RunRequest,
    safety: &SafetyLimits,
    gpu_index: Option<u32>,
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
    let command = build_command(capabilities, metadata, &request.candidate, request.port);
    let command_display = command_display(&command);
    write_shell_command(&artifacts.command, &command_display)?;

    let mut server = match LlamaServer::spawn(&command, &artifacts.server_log).await {
        Ok(server) => server,
        Err(err) => {
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
                },
            );
            write_json(&result.artifacts.result_json, &result)?;
            return Ok(result);
        }
    };

    let sampler = TelemetrySampler::start(
        server.pid,
        &artifacts.telemetry_jsonl,
        gpu_index,
        TELEMETRY_INTERVAL,
    );
    let base_url = format!("http://127.0.0.1:{}/v1", request.port);
    let run_started = Instant::now();
    let drive_result = tokio::select! {
        result = drive_probes(&base_url, metadata, &request) => result,
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
            request_artifact = probe_output.request_artifact;
            response_artifact = probe_output.response_artifact;
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
    let telemetry = sampler.stop().await;
    let ended_at = Utc::now();
    let log_text = fs::read_to_string(&artifacts.server_log).unwrap_or_default();
    if outcome == Outcome::Pass && log_indicates_oom(&log_text) {
        outcome = Outcome::Oom;
        note = "server log contains OOM/CUDA allocation failure".to_string();
    }

    let timing = parse_llama_timings(&log_text);
    let mut metrics = crate::profile::RunMetrics::from(telemetry);
    metrics.server_prompt_eval_toks_per_s = timing.best_prompt_toks_per_s();
    metrics.server_generation_toks_per_s = timing.best_generation_toks_per_s();
    metrics.client_ttft_ms = probes
        .get("output")
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
            .or_else(|| probes.get("fullctx"))
            .and_then(|probe| probe.prompt_tokens)
    });
    let completion_tokens = probes
        .get("output")
        .or_else(|| probes.get("sanity"))
        .and_then(|probe| probe.completion_tokens);

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
    };
    write_json(&result.artifacts.result_json, &result)?;
    Ok(result)
}

async fn drive_probes(
    base_url: &str,
    metadata: &GgufMetadata,
    request: &RunRequest,
) -> Result<ProbeOutput> {
    wait_for_server(base_url).await?;
    let mut probes = BTreeMap::new();
    let mut request_artifact = Vec::new();
    let mut response_artifact = Vec::new();

    let sanity_prompt = "Reply with exactly the single character: K";
    let sanity = post_chat_completion(
        base_url,
        &metadata.display_name(),
        sanity_prompt,
        1,
        REQUEST_TIMEOUT,
    )
    .await
    .context("sanity probe")?;
    request_artifact.push(sanity.request_json.clone());
    response_artifact.push(sanity.response_json.clone());
    probes.insert("sanity".to_string(), sanity.summary);

    match request.prompt_plan {
        PromptPlan::Tune {
            ingest_target_tokens,
        } => {
            let output_prompt =
                "Write a concise checklist for safely profiling a local llama.cpp model.";
            let output = post_chat_completion(
                base_url,
                &metadata.display_name(),
                output_prompt,
                128,
                REQUEST_TIMEOUT,
            )
            .await
            .context("output probe")?;
            request_artifact.push(output.request_json.clone());
            response_artifact.push(output.response_json.clone());
            probes.insert("output".to_string(), output.summary);

            let (ingest_prompt, estimate) = repeated_license_prompt(ingest_target_tokens);
            let ingest = post_chat_completion(
                base_url,
                &metadata.display_name(),
                &ingest_prompt,
                1,
                REQUEST_TIMEOUT,
            )
            .await
            .context("ingest probe")?;
            let mut summary = ingest.summary;
            summary.prompt_tokens = Some(estimate);
            request_artifact.push(ingest.request_json.clone());
            response_artifact.push(ingest.response_json.clone());
            probes.insert("ingest".to_string(), summary);
        }
        PromptPlan::FullCtx { target_tokens } => {
            let (full_prompt, estimate) = repeated_license_prompt(target_tokens);
            let fullctx = post_chat_completion(
                base_url,
                &metadata.display_name(),
                &full_prompt,
                1,
                REQUEST_TIMEOUT,
            )
            .await
            .context("fullctx probe")?;
            let mut summary = fullctx.summary;
            summary.prompt_tokens = Some(estimate);
            request_artifact.push(fullctx.request_json.clone());
            response_artifact.push(fullctx.response_json.clone());
            probes.insert("fullctx".to_string(), summary);
        }
    }

    Ok(ProbeOutput {
        probes,
        request_artifact: json!({ "probes": request_artifact }),
        response_artifact: json!({ "probes": response_artifact }),
    })
}

#[derive(Debug)]
struct ProbeOutput {
    probes: BTreeMap<String, ProbeSummary>,
    request_artifact: serde_json::Value,
    response_artifact: serde_json::Value,
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
    let client = reqwest::Client::new();
    let url = format!("{base_url}/chat/completions");
    let payload = json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": true
    });
    let started = Instant::now();
    let response = tokio::time::timeout(timeout, client.post(&url).json(&payload).send())
        .await
        .context("chat request timed out")??
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

async fn wait_for_server(base_url: &str) -> Result<()> {
    let health = base_url.trim_end_matches("/v1").to_string();
    let client = reqwest::Client::new();
    let started = Instant::now();
    while started.elapsed() < STARTUP_TIMEOUT {
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
    if capabilities.supports("--reasoning") {
        args.extend(["--reasoning".to_string(), "off".to_string()]);
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
        if capabilities.supports("-ctv") {
            args.extend(["-ctv".to_string(), kv_cache.clone()]);
        } else if capabilities.supports("--cache-type-v") {
            args.extend(["--cache-type-v".to_string(), kv_cache.clone()]);
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

fn find_open_port(start: u16) -> Result<u16> {
    for port in start..start.saturating_add(1000) {
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
) -> Result<Option<CandidateConfig>> {
    let recs = match load_recommendations(&metadata.profiler_dir()) {
        Ok(recs) => recs,
        Err(_) => return Ok(None),
    };
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
    let source = fs::read_to_string("/usr/share/licenses/spdx/Apache-2.0.txt")
        .unwrap_or_else(|_| include_str!("../fixtures/Apache-2.0.txt").to_string());
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
        r"(?m)(?:^|:)\s*eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*(?:runs?|tokens?).*?([0-9.]+)\s*tokens per second",
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
    collect_recommendation_dirs(root, &mut found)?;
    let mut output = Vec::new();
    let mut seen = BTreeSet::new();
    for rec_path in found {
        let data = fs::read_to_string(&rec_path)?;
        let recs: RecommendationFile = serde_json::from_str(&data)?;
        if !seen.insert(recs.model_path.clone()) {
            continue;
        }
        if recs.model_path.exists() {
            let metadata = read_metadata(&recs.model_path)?;
            output.push((metadata, recs));
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
                let rec = path.join("recommendations.json");
                if rec.exists() {
                    found.push(rec);
                }
                continue;
            }
            collect_recommendation_dirs(&path, found)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
