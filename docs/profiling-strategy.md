# Profiling Strategy

`llama-cpp-profiler` profiles runtime fit and speed. It does not benchmark model
intelligence, coding quality, or alignment behavior.

The profiling strategy targets consumer Linux systems with a single GPU. It is
not intended to optimize model placement across multiple GPUs.

## General Evidence

The profiler records evidence that applies to any `llama-server` client:

- exact GGUF path and metadata
- exact `llama-server` command
- profiler, hardware, GPU, driver, and `llama-server` environment snapshot
- context, batch, microbatch, KV cache, fit target, and MoE placement
- generation and prompt-processing thread counts when explicitly refined
- prompt ingest throughput
- generation throughput
- client-observed TTFT through `/v1/chat/completions`
- VRAM peak and minimum free headroom
- RAM, swap, process RSS, and CPU utilization
- startup failures, OOMs, timeouts, and parse-partial runs

Client harnesses such as opencode are not part of the core scoring model. They
can be exported as labels that point at the server endpoint, but the running
`llama-server` process determines the actual GGUF and runtime config. Opencode
is an optional adapter and receives the profile's validated context.

Recommendations are environment-bound. If the OS/architecture, CPU identity or
core count, RAM/swap totals, GPU backend, GPU inventory/driver, or
`llama-server` executable/help output changes, old runs are kept as stale
evidence and excluded from best-observed-profile selection.

## Dense Models

Dense models usually have no MoE offload escape hatch, so the useful sweep is:

- context cap
- batch and microbatch
- KV cache type, starting with `q8_0` and then `q4_0`
- `--fit-target`, from safer high values toward tighter low values

Dense recommendations should be skeptical of speed gains that leave very little
VRAM headroom. Lower quant can improve output speed, but at long context the KV
and compute-buffer fit often dominate.

Candidate planning uses the current environment snapshot to keep likely-safe
candidates ahead of aggressive ones. The risk heuristic compares model size plus
an approximate KV-cache footprint against total detected VRAM; if VRAM is not
available, the conservative default order is preserved. The first candidates use
the model's native context length, capped by an explicit `--ctx` when supplied.
Lower-context fallbacks are
kept later in the queue. During `tune`, passed runs with at least 2 GiB free VRAM
can promote more aggressive already-planned candidates, while OOM, timeout,
crash, or too-tight runs can promote safer already-planned candidates, including
lower-context fallbacks. This stays within the selected preset or `--max-runs`
budget. `tune --plan` exposes the initial ordered candidates as JSON without
starting servers.

For MoE models, `tune` and `recommend` also accept `--n-cpu-moe-values` to put
known expert-placement boundary values at the front of the plan. This is useful
when prior runs or neighboring models show that the strongest observed region is
around a narrow partial-MoE range such as `32,31,30`, and a bounded `quick` run
should spend its budget there instead of on conservative baselines.

## MoE Models

MoE models need a separate search because CPU/GPU expert placement changes both
speed and memory behavior.

The sweep starts with safe `--cpu-moe` baselines, then moves through
`--n-cpu-moe` values from CPU-heavy toward more GPU-resident placements. A lower
`--n-cpu-moe` can be faster, but it may fail context allocation or leave too
little VRAM for a normal desktop session.

The profiler records failed startup and OOM boundaries because they are useful
evidence: future agents should not retry the same aggressive settings blindly.

## Thread Refinement

After the bounded placement search, the profiler selects its best observed
balanced primary result. If that placement has CPU-resident MoE experts,
explicit partial GPU offload, or a server log that reports only some layers
offloaded, it runs a second-stage thread sweep. Fully GPU-resident selections and
servers without both `--threads` and `--threads-batch` skip it.

The sweep reruns the selected candidate with llama.cpp defaults and up to four
explicit, topology-derived pairs: half physical/all physical, all physical/all
physical, all physical/all logical, and all logical/all logical. Duplicate pairs
are removed. The contemporaneous default rerun is the baseline; an explicit pair is
eligible for recommendations only when its harmonic mean of prompt and output
throughput improves by at least 3%. All observations remain in the run artifacts.
The primary `--max-runs` budget does not include these up to five refinement runs.

## Final-Stage Realistic Validation

After placement search and any accepted thread refinement, `standard` and
`thorough` rank the current safe candidates by balanced throughput and validate
the best observed candidate with one combined request. `quick` does this only
with `--validate-best`. When `--confirm-best` is enabled, confirmation runs are
combined with their baseline by median throughput before this ranking, and the
eventual validated recommendation retains the repeated-measurement count. The
prompt target is
`min(max(context / 4, 16k), 64k)`, reduced when necessary to reserve room for up
to 1024 output tokens. The timeout is estimated from the baseline prompt and
generation speeds, doubled for margin, given 120 seconds of startup/variance
allowance, and bounded between 10 minutes and 2 hours.

The run records actual prompt and output tokens, TTFT, prompt and generation
throughput, VRAM headroom, RAM/swap telemetry, and retained-throughput ratios
against the short-probe baseline. Early EOS is a usable but incomplete result.
A crash, timeout, safety violation, or either retained ratio below 25% marks the
candidate as failed and advances to the next ranked candidate. This ratio gate
detects extreme collapse without imposing a model-independent tok/s target.
Once a realistic validation passes, recommendations are sourced from passed
realistic runs; failed candidates' short-probe baselines are excluded.

## Probe Set

Plain `tune` defaults to the bounded `quick` preset and never adds a near-full
context probe automatically. Its probes are:

- `sanity`: tiny answer, `max_tokens = 1`.
- `output`: small prompt, 128 generated tokens.
- `ingest`: repeated MIT license prompt around 16k tokens for `quick` and 64k for broader presets.

`quick` results are labeled with smoke validation even though the ingest probe is
still run at the smaller 16k-token target. `standard` and `thorough` results are
labeled `standard-ingest` before their final realistic validation stage.

`tune --near-full-ingest` and `recommend --near-full-ingest` add one opt-in
`near_full_ingest` probe using a one-shot repeated-text prompt just below the
requested context. The default target is about 94% of the requested context, so a
266k context run targets roughly 250k estimated prompt tokens. This is separate
from normal tuning because it can take substantially longer.

`fullctx` is explicit opt-in and sends a sanity probe followed by the near-full
prompt probe, defaulting to 80% of the active profile's server context and
`max_tokens = 1`. An explicit `--target-tokens` value is capped below that
context. It exists for TTFT and stability checks, not for normal tuning.

Prompt token counts prefer server timing lines in `server.log`. Prompt
generation uses `/usr/share/licenses/spdx/MIT.txt` when available, with the
project's MIT license text as fallback, so runs are deterministic and independent
of network access.

Thinking mode is the default probe behavior and includes reasoning-specific
server/request settings. `--probe-mode generic` omits those settings.

## Scoring

Profiles are the best observed configurations among runs that pass safety
limits:

- `interactive-fast`: best observed generation speed.
- `interactive-safe`: best observed generation speed with at least 1 GiB free
  VRAM.
- `prompt-replay`: best observed prompt ingest speed.
- `balanced`: best observed harmonic mean of generation and prompt ingest speed.

Rejected runs keep a compact reason: OOM, timeout, server crash, too-tight VRAM
or swap use, severe realistic-validation degradation, interrupted, or
parse-partial, with the first failure note line when
available. `parse-partial` is still usable for recommendation scoring when it
passes safety limits because the request completed but one or more llama.cpp
timing lines were missing.

Legacy, changed-model, or changed-environment runs are listed separately as
stale. They are never used for scoring. Missing NVIDIA telemetry leaves safety
unknown; such profiles may be ranked for speed but cannot be `interactive-safe`.
This beta supports Linux and `llama-server`, primarily on consumer single-GPU
systems. Multi-GPU hardware may be recorded in the environment snapshot, but
placement across GPUs is not modeled or optimized. Non-NVIDIA backend
optimization remains limited or deferred.
