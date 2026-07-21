# Agent Workflow

This repository builds a Linux, single-GPU `llama-server` profiler for onboarding
local GGUF models on the current machine. It measures runtime fit and speed; it
does not evaluate model intelligence or coding quality. Keep profiler behavior
client-agnostic: opencode, launchers, and local aliases are optional adapters
that consume the OpenAI-compatible server endpoint.

## Onboard a model

1. Run `llama-cpp-profiler doctor --json` to capture the current profiler,
   hardware, telemetry, and `llama-server` environment. Set
   `LLAMA_SERVER=/path/to/llama-server` when the binary is not in `PATH`.
2. Run `llama-cpp-profiler scan ~/Models --no-tui` to discover GGUFs in an
   agent-friendly table. Files with `mmproj` or `draft` in the name are ignored.
3. Run `llama-cpp-profiler inspect <model-or-gguf> --json` before deciding
   whether a model is dense or MoE. A directory is accepted and resolves to its
   largest model GGUF.
4. Optionally preview the hardware-aware search without starting a server:
   `llama-cpp-profiler tune <model> --preset standard --plan`. `--json` is
   accepted only with `--plan`.
5. Prefer the high-level onboarding command:
   `llama-cpp-profiler recommend <model> --preset standard --goal generation --confirm-best --agent`.
   It tunes, performs realistic final validation, selects the generation profile,
   and emits compact JSON containing the exact `llama-server` command. Use the
   default `--goal balanced` for mixed prompt/output work or `--goal prompt` for
   prompt-ingest-heavy work.
6. For a short initial smoke run, use `recommend --preset quick --agent`.
   Quick skips realistic final validation unless `--validate-best` is added.
7. For existing results, read `report <model-or-model-root> --agent` before raw
   logs. Use `serve <model> --profile interactive-fast --print` when an exact
   saved generation-oriented server command is needed without rerunning tuning.

The 800 MiB free-VRAM floor and 1024 MiB swap-delta ceiling are safety gates,
not optimization targets. Recommendation selection first rejects candidates
outside those gates, then prefers KV-cache precision in the order Q8/Q8,
Q8/Q4, Q4/Q4. Workload throughput breaks ties only within a precision tier.
Override `--min-vram-free-mib` only when the user explicitly wants a different
margin.

Plain `tune` and `recommend` default to the bounded `quick` preset. Use
`standard` for normal onboarding; `thorough` can run up to 48 primary candidates
and should be explicit. Tuning starts one local server at a time, stores artifacts
beside the model under `.llama-cpp-profiler/models/<model-path-hash>/`, and may
add thread-refinement, confirmation, and final-validation runs outside the
primary candidate budget.

Avoid `fullctx` unless the user explicitly asks for near-full-context TTFT or
stability. Its default prompt target is 80% of the active profile's server
context, and it can run for many minutes and pressure VRAM, RAM, and swap.
Likewise, do not add `--near-full-ingest` during ordinary onboarding.

Saved results are tied to the model identity and runtime environment. Stale or
legacy runs are excluded from recommendation selection, and `serve` refuses a
stale recommendation by default. Reprofile after meaningful model,
`llama-server`, driver, or hardware changes; do not reach for `--allow-stale` as
a routine workaround.

Do not hard-code this machine's model names or opencode ids into profiler logic.
Machine-specific information belongs in generated reports or optional exports,
not in the scoring engine. Export is dry-run unless `--write` is present.

Documentation sync rule: code is the source of truth. When docs and behavior
diverge, update docs rather than changing profiler behavior to match prose.
