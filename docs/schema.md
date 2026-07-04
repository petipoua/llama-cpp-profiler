# Data Schemas

The schemas are versioned by `schema_version`. Version `1` is the initial format.

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
- `candidate`: generated tuning config: context, batch, microbatch, KV cache type, fit target, and MoE placement flags.
- `test_kind`: `tune` or `fullctx`.
- `requested_context`: context passed to the server.
- `prompt_tokens`, `completion_tokens`: best available token counts, preferring server timing lines.
- `metrics.server_prompt_eval_toks_per_s`: parsed from llama.cpp `prompt eval time`.
- `metrics.server_generation_toks_per_s`: parsed from llama.cpp `eval time`.
- `metrics.client_ttft_ms`: client-observed first streamed content token latency.
- `metrics.total_wall_ms`: wall time for the probe sequence.
- `metrics.peak_vram_mib`, `metrics.min_free_vram_mib`: sampled with `nvidia-smi`.
- `metrics.gpu_util_avg_pct`, `metrics.gpu_util_max_pct`: sampled GPU utilization.
- `metrics.ram_available_min_mib`, `metrics.swap_delta_mib`, `metrics.process_rss_peak_mib`, `metrics.cpu_util_avg_pct`: system/process telemetry.
- `probes`: per-probe summaries for `sanity`, `output`, `ingest`, or `fullctx`.
- `outcome`: `pass`, `oom`, `timeout`, `server_crash`, `too_tight`, `parse_partial`, or `interrupted`.
- `artifacts`: paths to `command.sh`, `server.log`, `telemetry.jsonl`, `request.json`, `response.json`, and `result.json`.
- `note`: short human-readable note.

## `recommendations.json`

One file is written per model directory.

Important fields:

- `schema_version`
- `generated_at`
- `model_path`
- `profiles`: ranked profile recommendations.
- `rejected`: failed, OOM, timeout, swap-heavy, or too-tight candidate summaries.
- `next_suggested_test`: a compact next action for future agents.

Each profile contains:

- `id`: `interactive-fast`, `interactive-safe`, `prompt-replay`, `balanced`, or `quality-night`.
- `role`: human-readable role.
- `source_run_id`: run that produced the profile.
- `command`, `command_display`: exact `llama-server` command.
- `output_toks_per_s`, `prompt_toks_per_s`, `ttft_ms`
- `peak_vram_mib`, `headroom_mib`
- `risk`: `low`, `medium`, or `high`.
- `note`

## Reports

Markdown reports put the comparison table first:

| Profile | Role | Output tok/s | Prompt tok/s | TTFT | Peak VRAM | Headroom | Risk | Command |
|---|---|---:|---:|---:|---:|---:|---|---|

`report --agent` prints one compact JSON object:

- `best_profile_ids`
- `exact_command`
- `key_metrics`
- `failures`
- `next_suggested_test`

This is the main interface for future agents that need a quick, low-token answer.
