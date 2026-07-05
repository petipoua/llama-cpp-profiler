# llama-cpp-profiler

`llama-cpp-profiler` is a Rust CLI for empirically profiling GGUF models with
`llama-server`. It discovers models, runs bounded probes, captures server logs
and machine telemetry, ranks usable configs, and writes agent-readable reports.
Recommendations are tied to the hardware and `llama-server` environment that
produced them, so stale runs are kept as evidence but not used for best profiles.

The core tool is about `llama.cpp` server behavior. Client harnesses such as
opencode are optional export adapters because they only call the OpenAI-compatible
endpoint; they do not choose or load the GGUF.

## Quickstart

Install from this repository:

```bash
cargo install --path .
```

Discover local models and inspect one:

```bash
llama-cpp-profiler scan ~/Models
llama-cpp-profiler inspect ~/Models/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive --json
```

Run a bounded tuning pass and print the best saved server command:

```bash
llama-cpp-profiler tune ~/Models/Qwopus3.6-35B-A3B-Coder-MTP --ctx 262144 --preset quick --max-runs 2
llama-cpp-profiler serve ~/Models/Qwopus3.6-35B-A3B-Coder-MTP --profile interactive-fast --print
```

## Usage

Scan a model store:

```bash
llama-cpp-profiler scan ~/Models
```

On a TTY, `scan` opens a compact searchable picker. In non-interactive shells it
prints a table. Files with `mmproj` or `draft` in the name are ignored.

Inspect GGUF metadata:

```bash
llama-cpp-profiler inspect ~/Models/<model-or-gguf> --json
```

This reads only the GGUF header and key/value metadata, then reports architecture,
quant, native context, MoE expert counts, chat-template presence, and prior runs.

Tune safely:

```bash
llama-cpp-profiler tune ~/Models/<model-or-gguf> --ctx 262144 --preset standard
```

`tune` starts one `llama-server` at a time on localhost ports beginning at `18180`,
runs sanity/output/ingest probes, writes raw artifacts, and stops the server after
each candidate. It does not run a 250k-token prompt.

Preview the hardware-aware candidate plan without starting servers:

```bash
llama-cpp-profiler tune ~/Models/<model-or-gguf> --ctx 262144 --preset quick --plan
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

The human report prints comparison tables. `--agent` emits compact JSON with best
profile ids, source run/candidate/context, validation level, the exact best
command, key metrics, failures, stale runs, and the next suggested test.

Export managed notes or client labels:

```bash
llama-cpp-profiler export ~/Models --markdown --dry-run
llama-cpp-profiler export ~/Models --markdown --opencode ~/.config/opencode/opencode.json --write
```

Markdown export updates only the managed block between
`<!-- llama-cpp-profiler:start -->` and `<!-- llama-cpp-profiler:end -->` in
`RUNNING_NOTES.md`. Opencode export is optional and only adds labels for a client
that points at the running `llama-server` endpoint.

## Documentation

### Commands

```bash
llama-cpp-profiler scan PATH [--no-tui]
llama-cpp-profiler inspect PATH [--json]
llama-cpp-profiler tune PATH [--ctx TOKENS] [--preset quick|standard|thorough] [--max-runs N] [--plan --json]
llama-cpp-profiler fullctx PATH [--profile ID] [--target-tokens TOKENS]
llama-cpp-profiler report PATH [--agent] [--include-stale]
llama-cpp-profiler serve PATH [--profile ID] [--port PORT] [--print] [--allow-stale]
llama-cpp-profiler doctor [--json]
llama-cpp-profiler export PATH [--markdown] [--opencode PATH] [--dry-run] [--write]
```

### Defaults

- Tune port range starts at `18180`; serve defaults to `18080`.
- Default context cap is `262144`.
- Safety defaults are `--min-vram-free-mib 512` and `--max-swap-delta-mib 1024`.
- `quick` runs at most 6 candidates; `standard` runs at most 16; `thorough` runs a broader sweep.
- `fullctx` is the only command that targets near-full prompts.
- Stale or legacy runs are excluded from best-profile selection by default.
- Export is dry-run unless `--write` is present.

### Artifacts

Each model directory gets:

```text
.llama-cpp-profiler/
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

`result.json` stores the exact command, GGUF metadata, probe summaries, parsed
llama.cpp timing lines, client-observed TTFT, telemetry peaks/minimums, outcome,
environment snapshot, validation level, and paths to raw artifacts.

### Profiles

Recommendations are derived from current-environment passed runs and safety
limits:

- `interactive-fast`: highest generation tok/s within safety limits.
- `interactive-safe`: highest generation tok/s with at least 1 GiB free VRAM.
- `prompt-replay`: highest prompt eval tok/s.
- `balanced`: harmonic mean of prompt and output throughput.
- `quality-night`: highest rough quant tier that starts and passes sanity.

### Requirements

- Linux
- `llama-server` in `PATH`, or `LLAMA_SERVER=/path/to/llama-server`
- NVIDIA telemetry through `nvidia-smi` for GPU metrics
- Optional: a client such as opencode that talks to `http://127.0.0.1:18080/v1`

See [docs/schema.md](docs/schema.md) and
[docs/profiling-strategy.md](docs/profiling-strategy.md) for data contracts and
tuning logic.
