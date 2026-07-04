# Profiling Strategy

`llama-cpp-profiler` profiles runtime fit and speed. It does not benchmark model
intelligence, coding quality, or alignment behavior.

## General Evidence

The profiler records evidence that applies to any `llama-server` client:

- exact GGUF path and metadata
- exact `llama-server` command
- context, batch, microbatch, KV cache, fit target, and MoE placement
- prompt ingest throughput
- generation throughput
- client-observed TTFT through `/v1/chat/completions`
- VRAM peak and minimum free headroom
- RAM, swap, process RSS, and CPU utilization
- startup failures, OOMs, timeouts, and parse-partial runs

Client harnesses such as opencode are not part of the core scoring model. They
can be exported as labels that point at the server endpoint, but the running
`llama-server` process determines the actual GGUF and runtime config.

## Dense Models

Dense models usually have no MoE offload escape hatch, so the useful sweep is:

- context cap
- batch and microbatch
- KV cache type, starting with `q8_0` and then `q4_0`
- `--fit-target`, from safer high values toward tighter low values

Dense recommendations should be skeptical of speed gains that leave very little
VRAM headroom. Lower quant can improve output speed, but at long context the KV
and compute-buffer fit often dominate.

## MoE Models

MoE models need a separate search because CPU/GPU expert placement changes both
speed and memory behavior.

The sweep starts with safe `--cpu-moe` baselines, then moves through
`--n-cpu-moe` values from CPU-heavy toward more GPU-resident placements. A lower
`--n-cpu-moe` can be faster, but it may fail context allocation or leave too
little VRAM for a normal desktop session.

The profiler records failed startup and OOM boundaries because they are useful
evidence: future agents should not retry the same aggressive settings blindly.

## Probe Set

`tune` uses bounded probes:

- `sanity`: tiny answer, `max_tokens = 1`.
- `output`: small prompt, 128 generated tokens.
- `ingest`: repeated Apache-2.0 prompt around 16k tokens for `quick` and 64k for broader presets.

`fullctx` is explicit opt-in and targets a near-full prompt, defaulting to about
250k tokens. It exists for TTFT and stability checks, not for normal tuning.

Prompt token counts prefer server timing lines in `server.log`. Prompt generation
uses repeated local Apache-2.0 text so runs are deterministic and independent of
network access.

## Scoring

Profiles are selected from runs that pass safety limits:

- `interactive-fast`: maximize generation speed.
- `interactive-safe`: maximize generation speed with at least 1 GiB free VRAM.
- `prompt-replay`: maximize prompt ingest speed.
- `balanced`: maximize harmonic mean of generation and prompt ingest speed.
- `quality-night`: use quant tier only as a rough quality proxy, clearly labeled.

Rejected runs keep their reason: OOM, timeout, server crash, too-tight VRAM,
swap-thrashing, interrupted, or parse-partial.
