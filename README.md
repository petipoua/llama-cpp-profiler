# llama-cpp-profiler

`llama-cpp-profiler` is a Linux Rust CLI for empirically finding safe and fast
`llama-server` configurations for local GGUF models. It measures runtime fit and
speed—context, throughput, TTFT, memory headroom, and failures—not intelligence,
coding quality, or model alignment.

The core tool is about `llama.cpp` server behavior. Client harnesses such as
opencode are optional export adapters because they only call the OpenAI-compatible
endpoint; they do not choose or load the GGUF.

## Quickstart

Install from this repository:

```bash
cargo install --path .
```

Discover local models and inspect one. `inspect` accepts either a GGUF file or a
directory containing GGUFs; when a directory is passed, the largest model GGUF is
selected:

```bash
llama-cpp-profiler scan ~/Models
llama-cpp-profiler inspect ~/Models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive --json
```

Run a bounded tuning pass and print the best saved server command:

```bash
llama-cpp-profiler tune ~/Models/Qwopus3.6-35B-A3B-Coder-MTP --preset quick --max-runs 2
llama-cpp-profiler serve ~/Models/Qwopus3.6-35B-A3B-Coder-MTP --profile interactive-fast --print
```

Or use the high-level command that tunes and prints the selected server command
in one step:

```bash
llama-cpp-profiler recommend ~/Models/Qwopus3.6-35B-A3B-Coder-MTP --preset quick
llama-cpp-profiler recommend ~/Models/Qwopus3.6-35B-A3B-Coder-MTP --preset quick --agent
```

## Usage

Scan a model store:

```bash
llama-cpp-profiler scan ~/Models
```

On a TTY, `scan` opens a compact searchable picker. In non-interactive shells,
or with `--no-tui`, it prints a table. Files with `mmproj` or `draft` in the
name are ignored.

Inspect GGUF metadata:

```bash
llama-cpp-profiler inspect ~/Models/<model-or-gguf> --json
```

This reads only the GGUF header and key/value metadata, then reports architecture,
quant, native context, MoE expert counts, chat-template presence, and prior runs.

Tune safely:

```bash
llama-cpp-profiler tune ~/Models/<model-or-gguf> --preset standard
```

`tune` starts one `llama-server` at a time on localhost ports beginning at `18180`,
runs sanity/output/ingest probes, writes raw artifacts, and stops the server after
each candidate. It can promote safer or more aggressive already-planned
candidates from observed results, but stays within the selected preset or
`--max-runs` budget. It does not run a 250k-token prompt.

The default `thinking` probe mode keeps the reasoning-oriented baseline:
generated server commands and chat probes use `--reasoning on`,
`--reasoning-budget 4096`, `--chat-template-kwargs '{"enable_thinking":true}'`,
`--temp 0.6`, `--top-p 0.95`, `--top-k 20`, `--min-p 0.0`,
`--presence-penalty 0.0`, and `--repeat-penalty 1.0`.

By default, tuning starts at the model's native maximum context when GGUF metadata
reports it. Use `--ctx` to cap or explicitly select a lower context. Lower-context
fallback candidates are kept in the plan and can be promoted after OOM, timeout,
crash, or too-tight runs. Plain `tune` uses `quick` and never runs a near-full
context probe automatically.

Use `--probe-mode generic` to omit reasoning-specific server arguments and request
fields. Thinking mode remains the default.

For an explicit near-full prompt-ingest check during tuning, opt in with
`--near-full-ingest`. The target is about 94% of the requested context, so a
266k context run targets roughly 250k estimated prompt tokens. Override it with
`--near-full-target-tokens`.

For MoE models where you already know the useful expert-placement boundary, put
specific `--n-cpu-moe` values at the front of the plan:

```bash
llama-cpp-profiler tune ~/Models/<model-or-gguf> --preset quick --n-cpu-moe-values 32,31,30 --max-runs 3
```

Explicit MoE candidates use the common high-throughput shape
`-ctk q8_0 -ctv q8_0 -b 16384 -ub 4096` and are still subject to the normal
safety limits.

Preview the hardware-aware candidate plan without starting servers. `--plan`
prints JSON; `--json` is accepted only with `--plan` for compatibility with
agent workflows.

```bash
llama-cpp-profiler tune ~/Models/<model-or-gguf> --preset quick --plan
```

Inspect the current profiler, hardware, and server environment:

```bash
llama-cpp-profiler doctor --json
```

Run explicit near-full context:

```bash
llama-cpp-profiler fullctx ~/Models/<model-or-gguf> --profile interactive-fast --target-tokens 250000
```

This is intentionally separate from `tune` because near-full prompts can take many
minutes and put real pressure on VRAM, RAM, and swap.

Read ranked recommendations:

```bash
llama-cpp-profiler report ~/Models
llama-cpp-profiler report ~/Models --agent
llama-cpp-profiler report ~/Models --include-stale
```

The human report prints comparison tables, native versus validated context,
rejected runs, telemetry safety status, and the next suggested test. `--agent`
emits stable JSON with an explicit agent schema version, model identity, environment
validity, telemetry status, exact command, metrics, failures, stale runs, and next
action.

Export managed notes or client labels:

```bash
llama-cpp-profiler export ~/Models --markdown --dry-run
llama-cpp-profiler export ~/Models --markdown --opencode ~/.config/opencode/opencode.json --write
```

Markdown export updates only the managed block between
`<!-- llama-cpp-profiler:start -->` and `<!-- llama-cpp-profiler:end -->` in
`RUNNING_NOTES.md`. Opencode export is an optional adapter: it only adds labels
for a client that points at the running `llama-server` endpoint and does not
affect profiler scoring. Its context limit comes from the selected profile's
validated server context. If no export target is selected, `export` prints the
generated Markdown block.

## Documentation

### Commands

```bash
llama-cpp-profiler scan PATH [--no-tui]
llama-cpp-profiler inspect PATH [--json]
llama-cpp-profiler tune PATH [--ctx TOKENS] [--preset quick|standard|thorough] [--probe-mode thinking|generic] [--max-runs N] [--min-vram-free-mib MIB] [--max-swap-delta-mib MIB] [--port-start PORT] [--gpu-index INDEX] [--n-cpu-moe-values VALUES] [--near-full-ingest] [--near-full-target-tokens TOKENS] [--plan] [--json]
llama-cpp-profiler recommend PATH [--ctx TOKENS] [--preset quick|standard|thorough] [--probe-mode thinking|generic] [--max-runs N] [--profile ID] [--port PORT] [--min-vram-free-mib MIB] [--max-swap-delta-mib MIB] [--port-start PORT] [--gpu-index INDEX] [--n-cpu-moe-values VALUES] [--near-full-ingest] [--near-full-target-tokens TOKENS] [--agent]
llama-cpp-profiler fullctx PATH [--profile ID] [--target-tokens TOKENS] [--ctx TOKENS] [--probe-mode thinking|generic] [--min-vram-free-mib MIB] [--max-swap-delta-mib MIB] [--port-start PORT] [--gpu-index INDEX]
llama-cpp-profiler report PATH [--agent] [--include-stale]
llama-cpp-profiler serve PATH [--profile ID] [--port PORT] [--print] [--allow-stale]
llama-cpp-profiler doctor [--json]
llama-cpp-profiler export PATH [--markdown] [--opencode PATH] [--dry-run] [--write]
```

### Defaults

- Tune port range starts at `18180`; serve defaults to `18080`.
- If GGUF metadata provides native context, that is the default requested context;
  `--ctx` caps it or supplies the fallback when native context is absent.
- Candidate plans try that native/explicit context first. Lower-context fallbacks
  are available for adaptive promotion after failed or too-tight runs.
- Safety defaults are `--min-vram-free-mib 512` and `--max-swap-delta-mib 1024`.
- `quick` runs at most 6 candidates; `standard` runs at most 16; `thorough` runs a broader sweep.
- Plain `tune` and `recommend` default to `quick`; `standard` and `thorough` are explicit deeper modes.
- `thinking` is the default probe mode; `generic` omits reasoning-specific arguments and request fields.
- `quick` runs are labeled `smoke`; `standard` and `thorough` runs are labeled
  `standard-ingest`; `fullctx` runs are labeled `fullctx`.
- `--n-cpu-moe-values` is a comma-separated MoE-only override that prepends
  explicit partial-MoE candidates, for example `32,31,30`.
- `fullctx` targets near-full prompts by default. `tune` and `recommend` can run
  one optional near-full ingest probe with `--near-full-ingest`.
- Stale or legacy runs are excluded from best-profile selection by default.
- Export is dry-run unless `--write` is present.
- `LLAMA_SERVER=/path/to/llama-server` overrides the executable used for
  `tune`, `fullctx`, `serve`, and `doctor`.

### Artifacts

Each model gets an isolated state directory (flat GGUFs in one directory cannot
overwrite one another):

```text
.llama-cpp-profiler/
  models/<model-path-hash>/
    manifest.json
    recommendations.json
    runs/<run-id>/
      command.sh
      server.log
      telemetry.jsonl
      request.json
      response.json
      result.json
    reports/latest.md
```

`request.json` and `response.json` each contain a `probes` array with all probe
requests/responses for the run. `result.json` stores the exact command, GGUF
metadata, probe summaries, parsed llama.cpp timing lines, client-observed TTFT,
telemetry peaks/minimums, outcome, environment snapshot, validation level, and
paths to raw artifacts.

### Profiles

Recommendations are derived from current-environment passed runs and safety
limits:

- `interactive-fast`: highest generation tok/s within safety limits.
- `interactive-safe`: highest generation tok/s with at least 1 GiB free VRAM.
- `prompt-replay`: highest prompt eval tok/s.
- `balanced`: harmonic mean of prompt and output throughput.

### Requirements

- Linux
- `llama-server` in `PATH`, or `LLAMA_SERVER=/path/to/llama-server`
- NVIDIA telemetry through `nvidia-smi` is optional. Without it, VRAM safety is
  `unknown`, and a profile cannot qualify as `interactive-safe`.
- Optional: a client such as opencode that talks to `http://127.0.0.1:18080/v1`

This beta supports Linux and `llama-server`. Multi-GPU placement is not fully
optimized unless the selected server command explicitly controls placement;
non-NVIDIA telemetry and backend-specific sweeps are limited or deferred.
`fullctx` and the `standard`/`thorough` presets can be expensive and may pressure
VRAM, RAM, and swap.

See [docs/schema.md](docs/schema.md) and
[docs/profiling-strategy.md](docs/profiling-strategy.md) for data contracts and
tuning logic.
