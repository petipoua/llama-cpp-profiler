# Agent Workflow

This repository builds a general `llama-server` profiler. Keep the core behavior
client-agnostic: opencode, launchers, and local aliases are optional adapters that
consume the OpenAI-compatible server endpoint.

Preferred workflow for agents:

1. Run `llama-cpp-profiler scan ~/Models` to discover GGUFs.
2. Use `inspect <model> --json` before deciding whether the model is dense or MoE.
3. Use `tune --preset quick` for short smoke-labeled runs and `tune --preset standard` for normal recommendations.
4. Use `report --agent` as the token-saving interface before reading raw logs.
5. Use `serve --profile interactive-fast --print` to get the exact server command.

Avoid `fullctx` unless the user explicitly asks for near-full-context TTFT or
stability. It can run for many minutes and pressure VRAM, RAM, and swap.

Do not hard-code this machine's model names or opencode ids into profiler logic.
Machine-specific information belongs in generated reports or optional exports,
not in the scoring engine.

Documentation sync rule: code is the source of truth. When docs and behavior
diverge, update docs rather than changing profiler behavior to match prose.
