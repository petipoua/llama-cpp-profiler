# Data Schemas

The schemas are versioned by `schema_version`; the current beta schema version is
`6`. Version `5` added realistic-validation metadata and outcomes. Version `4`
added optional candidate thread counts and thread-refinement run kinds. Version
`3` added versioned model identity and validation fields. Earlier files are read
best-effort; files without model identity are retained as legacy stale evidence.

## `result.json`

One file is written for every candidate run.

Important fields:

- `schema_version`: integer schema version.
- `run_id`: timestamp plus candidate id.
- `started_at`, `ended_at`: UTC timestamps.
- `model_path`, `model_size_bytes`: profiled GGUF identity.
- `model_identity`: version, canonical path, file size, modification time, and a
  stable hash of parsed GGUF metadata. Results without it are legacy/stale.
- `gguf`: parsed GGUF metadata, including architecture, native context, quant, MoE expert counts, and chat-template presence.
- `command`: argv array used to launch `llama-server`.
- `command_display`: shell-escaped command string.
- `candidate`: generated tuning config: context, batch, microbatch, KV cache type,
  fit target, GPU layer count, MoE placement flags, optional `threads` and
  `threads_batch`, expected risk, note, and planning note. Null thread values use
  llama.cpp defaults.
- `candidate.expected_risk`, `candidate.planning_note`: hardware-aware planning annotations.
- Explicit MoE candidates requested with `--n-cpu-moe-values` are normal
  candidates in artifacts and plans. Their ids end in `-explicit`.
- `test_kind`: `tune`, `confirmation`, `fullctx`, `realistic-validation`,
  `thread-refinement`, or `thread-refinement-observation`. Confirmation runs
  repeat promising tune candidates for median ranking. The last kind records a
  tested thread pair that did not clear the 3% balanced-throughput acceptance
  gate and is excluded from recommendations.
- `requested_context`: context passed to the server.
- `validation_level`: serialized as `smoke`, `standard_ingest`, `realistic`, or
  `fullctx`; reports display `standard_ingest` as `standard-ingest`.
- `environment`: profiler, OS/architecture, CPU, memory, GPU, and `llama-server` snapshot.
- `compatibility`: environment validation (`current`, `legacy_missing_snapshot`,
  `server_changed`, or `hardware_changed`). Stale records may additionally use
  `legacy_missing_identity` or `model_changed` when model validation fails.
- `prompt_tokens`, `completion_tokens`: best available token counts, preferring server timing lines.
- `metrics.server_prompt_eval_toks_per_s`: parsed from llama.cpp `prompt eval time`.
- `metrics.server_generation_toks_per_s`: parsed from llama.cpp `eval time`.
- `metrics.client_ttft_ms`: client-observed first streamed content token latency.
- `metrics.total_wall_ms`: wall time for the probe sequence.
- `metrics.peak_vram_mib`, `metrics.min_free_vram_mib`: sampled with `nvidia-smi`.
- `metrics.gpu_util_avg_pct`, `metrics.gpu_util_max_pct`: sampled GPU utilization.
- `metrics.ram_available_min_mib`, `metrics.swap_delta_mib`, `metrics.process_rss_peak_mib`, `metrics.cpu_util_avg_pct`: system/process telemetry.
- `telemetry_status`: `measured` or `unknown`. Missing NVIDIA telemetry never
  receives a low-risk label and cannot qualify for `interactive-safe`.
- `realistic_validation`: optional baseline run id, target/requested and actual
  prompt/output token counts, prompt/output retained-throughput ratios, and an
  `incomplete_generation` early-EOS flag.
- `probes`: per-probe summaries for `sanity`, `output`, `ingest`, `realistic`,
  `near_full_ingest`, or `fullctx`. `tune` runs `sanity`, `output`, and
  `ingest`; `tune --near-full-ingest` and `recommend --near-full-ingest` also
  run `near_full_ingest`; final-stage validation runs only `realistic`; `fullctx`
  runs `sanity` and `fullctx`.
- `outcome`: `pass`, `oom`, `timeout`, `server_crash`, `too_tight`,
  `parse_partial`, `performance_degraded`, or `interrupted`.
- `artifacts`: paths to `command.sh`, `server.log`, `telemetry.jsonl`, `request.json`, `response.json`, and `result.json`. `request.json` and `response.json` wrap all probe payloads in a `probes` array.
- `note`: short human-readable note.

## `recommendations.json`

One file is written per model directory.

Important fields:

- `schema_version`
- `generated_at`
- `model_path`
- `model_identity`
- `profiles`: ranked profile recommendations.
- `rejected`: failed, OOM, timeout, swap-heavy, or too-tight candidate summaries.
- `stale`: legacy or changed-environment run summaries excluded from ranking.
- `environment`: current environment used when rebuilding recommendations.
- `environment_valid`: whether the saved environment still matches the current
  environment during report/serve validation.
- `coverage`: optional search scope for a tune run: preset, planned/tested
  candidate counts, searched and omitted dimensions, and confirmation-run count.
- `next_suggested_test`: a compact next action for future agents.

Each profile contains:

- `id`: `interactive-fast`, `interactive-safe`, `prompt-replay`, or `balanced`.
- `role`: human-readable role.
- `source_run_id`: run that produced the profile.
- `source_candidate_id`, `source_test_kind`, `requested_context`
- `validated_prompt_tokens`, `validation_level`, `compatibility`
- `model_identity`, `environment_valid`, `telemetry_status`
- `command`, `command_display`: exact `llama-server` command.
- `output_toks_per_s`, `prompt_toks_per_s`, `ttft_ms`
- `peak_vram_mib`, `headroom_mib`
- `confidence`, `measurement_count`: evidence label and number of repeated tune
  measurements supporting the selected candidate. A realistic-validation
  profile retains the count of its baseline candidate.
- `risk`: `low`, `medium`, `high`, or `unknown` when VRAM was not measured.
- `note`
- `realistic_validation` when the profile came from final-stage validation.

Profile ids may point at the same source run when one candidate is the best
observed configuration for multiple roles. Exact commands are stored as argv
plus a shell-escaped display string; `serve --port` rewrites the saved `--port`
value before running or printing the command.

## Reports

Markdown reports put the comparison table first:

| Profile | Role | Output tok/s | Prompt tok/s | TTFT | Peak VRAM | Headroom | Risk | Validation | Command |
|---|---|---:|---:|---:|---:|---:|---|---|---|

Interactive `tune` output ends with a concise summary of the selected
`interactive-fast` profile (or the first available profile), including generation
and prompt throughput, requested context, VRAM headroom, and a shell-escaped
`llama-cpp-profiler serve ... --profile ... --print` command.

`report --agent` prints one compact JSON object. Agent schema version `3` adds
search coverage and the expanded confidence labels. `agent_schema_version` is the
stable contract version, separate from the profiler `schema_version`:

- `best_profile_ids`: unambiguous `model-path#profile` keys for the best observed
  profiles.
- `exact_command`
- `confidence`: `provisional`, `benchmarked`, `confirmed`, or
  `full-context-validated` for the top command
- `coverage`: per-model candidate coverage, including dimensions not searched
- `key_metrics`
- `failures`
- `stale_profiles`
- `next_suggested_test`
- each metric also includes model identity, environment validity, telemetry
  status, risk, validation level, metrics, failures, and exact command.

Each metric includes model path, `profile_key`, profile id, source run id, source
candidate id, model kind, quant, native context, test kind, requested context,
validated prompt tokens, validation level, compatibility, metrics, risk,
confidence, a short `why`, and exact command.

This is the main interface for future agents that need a quick, low-token answer.

## `recommend --agent`

`recommend --agent` prints one compact JSON object for the selected model and
profile:

- `model_path`
- `agent_schema_version`, `schema_version`, `model_identity`
- `profile_id`
- `profile_key`
- `confidence`
- `goal`: selected workload goal (`generation`, `prompt`, or `balanced`)
- `measurement_count`, `coverage`
- `command`
- `exact_command`, `environment_valid`, `telemetry_status`, `risk`, `failures`, `stale`
- `output_toks_per_s`, `prompt_toks_per_s`, `ttft_ms`
- `requested_context`, `validated_prompt_tokens`, `validation_level`
- `realistic_validation`
- `next_suggested_test`
