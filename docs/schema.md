# Data Schemas

The schemas are versioned by `schema_version`. Version `2` adds environment
snapshots, compatibility status, validation levels, and richer agent metadata.
Version `1` result files are read best-effort and treated as legacy stale runs
because they do not contain environment snapshots.

## `result.json`

One file is written for every candidate run.

Important fields:

- `schema_version`: integer schema version.
- `run_id`: timestamp plus candidate id.
- `started_at`, `ended_at`: UTC timestamps.
- `model_path`, `model_size_bytes`: profiled GGUF identity.
- `gguf`: parsed GGUF metadata, including architecture, native context, quant, MoE expert counts, and chat-template presence.
- `command`: argv array used to launch `llama-server`.
- `command_display`: shell-escaped command string.
- `candidate`: generated tuning config: context, batch, microbatch, KV cache type, fit target, GPU layer count, MoE placement flags, expected risk, note, and planning note.
- `candidate.expected_risk`, `candidate.planning_note`: hardware-aware planning annotations.
- `test_kind`: `tune` or `fullctx`.
- `requested_context`: context passed to the server.
- `validation_level`: serialized as `smoke`, `standard_ingest`, or `fullctx`; reports display `standard_ingest` as `standard-ingest`.
- `environment`: profiler, OS/architecture, CPU, memory, GPU, and `llama-server` snapshot.
- `compatibility`: `current`, `legacy_missing_snapshot`, `server_changed`, or `hardware_changed`.
- `prompt_tokens`, `completion_tokens`: best available token counts, preferring server timing lines.
- `metrics.server_prompt_eval_toks_per_s`: parsed from llama.cpp `prompt eval time`.
- `metrics.server_generation_toks_per_s`: parsed from llama.cpp `eval time`.
- `metrics.client_ttft_ms`: client-observed first streamed content token latency.
- `metrics.total_wall_ms`: wall time for the probe sequence.
- `metrics.peak_vram_mib`, `metrics.min_free_vram_mib`: sampled with `nvidia-smi`.
- `metrics.gpu_util_avg_pct`, `metrics.gpu_util_max_pct`: sampled GPU utilization.
- `metrics.ram_available_min_mib`, `metrics.swap_delta_mib`, `metrics.process_rss_peak_mib`, `metrics.cpu_util_avg_pct`: system/process telemetry.
- `probes`: per-probe summaries for `sanity`, `output`, `ingest`, or `fullctx`. `tune` runs `sanity`, `output`, and `ingest`; `fullctx` runs `sanity` and `fullctx`.
- `outcome`: `pass`, `oom`, `timeout`, `server_crash`, `too_tight`, `parse_partial`, or `interrupted`.
- `artifacts`: paths to `command.sh`, `server.log`, `telemetry.jsonl`, `request.json`, `response.json`, and `result.json`. `request.json` and `response.json` wrap all probe payloads in a `probes` array.
- `note`: short human-readable note.

## `recommendations.json`

One file is written per model directory.

Important fields:

- `schema_version`
- `generated_at`
- `model_path`
- `profiles`: ranked profile recommendations.
- `rejected`: failed, OOM, timeout, swap-heavy, or too-tight candidate summaries.
- `stale`: legacy or changed-environment run summaries excluded from ranking.
- `environment`: current environment used when rebuilding recommendations.
- `next_suggested_test`: a compact next action for future agents.

Each profile contains:

- `id`: `interactive-fast`, `interactive-safe`, `prompt-replay`, `balanced`, or `quality-night`.
- `role`: human-readable role.
- `source_run_id`: run that produced the profile.
- `source_candidate_id`, `source_test_kind`, `requested_context`
- `validated_prompt_tokens`, `validation_level`, `compatibility`
- `command`, `command_display`: exact `llama-server` command.
- `output_toks_per_s`, `prompt_toks_per_s`, `ttft_ms`
- `peak_vram_mib`, `headroom_mib`
- `risk`: `low`, `medium`, or `high`.
- `note`

Profile ids may point at the same source run when one candidate is best for
multiple roles. Exact commands are stored as argv plus a shell-escaped display
string; `serve --port` rewrites the saved `--port` value before running or
printing the command.

## Reports

Markdown reports put the comparison table first:

| Profile | Role | Output tok/s | Prompt tok/s | TTFT | Peak VRAM | Headroom | Risk | Validation | Command |
|---|---|---:|---:|---:|---:|---:|---|---|---|

`report --agent` prints one compact JSON object:

- `best_profile_ids`
- `exact_command`
- `confidence`: `low`, `medium`, or `high` for the top command
- `key_metrics`
- `failures`
- `stale_profiles`
- `next_suggested_test`

Each metric includes model path, profile id, source run id, source candidate id,
model kind, quant, native context, test kind, requested context, validated prompt
tokens, validation level, compatibility, metrics, risk, confidence, a short
`why`, and exact command.

This is the main interface for future agents that need a quick, low-token answer.
