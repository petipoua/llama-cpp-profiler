# Changelog

## 0.1.0-beta.1

First focused beta release for Linux `llama-server` profiling.

- Uses native GGUF context by default and bounded `quick` tuning.
- Adds isolated per-model state, versioned model identity, and stale-state validation.
- Adds explicit `thinking` and `generic` probe modes.
- Keeps runtime fit and speed as the only recommendation claims.
- Makes missing NVIDIA telemetry explicit and preserves startup, timeout, OOM, and interruption evidence.
- Adds conditional CPU thread refinement and final-stage realistic workload validation.
- Makes `--confirm-best` median measurements affect final-stage candidate
  selection and preserves confirmation confidence in validated recommendations.
- Defaults explicit `fullctx` probes to 80% of the active server context and
  bounds custom targets to that context.
- Uses the MIT license exclusively for package metadata and deterministic prompt
  text.
- Makes the fake-server end-to-end test independent of live GPU load.
- Labels bounded-search results as best observed configurations and prints a
  concise end-of-tune summary with the exact next `serve` command.
- Removes the misleading quality-based recommendation and documents multi-GPU/non-NVIDIA limitations.

Known limitations: model intelligence is not evaluated; backend-specific sweeps,
automatic multi-GPU placement optimization, speculative decoding, and other
advanced llama.cpp dimensions are deferred.
