# TODO

Release-focused improvements for making `llama-cpp-profiler` a trustworthy,
minimal tool for humans and agents. The profiler should consistently describe
its result as the best observed configuration from a bounded empirical search,
not as a guaranteed global optimum.

1. Fix the nondeterministic end-to-end test — implemented

   The fake-server test currently reads live GPU telemetry and can fail when the
   desktop has less than the default 512 MiB VRAM headroom. Inject deterministic
   fake telemetry or explicitly disable telemetry in the test so the test suite
   never depends on current machine load.

   Implemented with a test-only fixed telemetry source. Production tuning still
   samples live telemetry, while the fake-server test injects deterministic VRAM,
   RAM, and swap values through the normal safety path.

2. Make release wording precise — implemented

   Use "best observed configuration" consistently in CLI output and
   documentation. Avoid claims that the profiler finds the absolute best
   configuration or maximizes every CPU, GPU, and VRAM parameter.

   CLI help, generated Markdown, profile roles, and tuning diagnostics now call
   results best observed configurations from the bounded search.

3. Add a concise tuning summary — implemented

   At the end of `tune`, print the selected profile's generation throughput,
   prompt throughput, context, VRAM headroom, and the exact next `serve` command.
   The result should be immediately useful without requiring a separate
   `report` invocation.

   `tune` now ends with the selected profile id, generation and prompt
   throughput, context, VRAM headroom, and a shell-escaped `serve --print`
   command.

4. Report what was and was not searched

   Add candidate coverage to human and agent reports, including the number of
   tested and planned candidates, searched dimensions, and important dimensions
   not searched. Clearly distinguish a quick bounded recommendation from a
   comprehensive search.

5. Add configurable workload goals

   Add a small `--goal` interface for selecting `generation`, `prompt`, or
   `balanced`. Reuse the existing recommendation profiles and make the selected
   goal determine the primary result and generated server command.

6. Add final-stage realistic workload validation — implemented

   After the normal search selects its best observed candidate, validate it with
   one combined long-prompt and sustained-generation request instead of making
   every candidate more expensive. Use up to 1024 output tokens and a bounded
   input target of `min(max(context / 4, 16k), 64k)` tokens. Scale the timeout
   from the selected candidate's measured prompt and generation throughput.

   Enable this by default for `standard` and `thorough`, and keep it opt-in for
   `quick` through a flag such as `--validate-best`. Record the actual generated
   token count, loaded-context prompt and output throughput, TTFT, VRAM headroom,
   RAM/swap behavior, and outcome. Treat early EOS as a successful but incomplete
   sustained-generation validation rather than requiring exactly 1024 tokens.

   Compare the result with the short-probe baseline and report the retained
   performance ratio without initially enforcing a universal tok/s threshold. If
   the candidate crashes, times out, becomes unsafe, or severely degrades, mark
   it as failing realistic validation and run the same validation against the
   next ranked candidate.

   Implemented as a post-search stage enabled by default for `standard` and
   `thorough`, or by `--validate-best` for `quick`. It uses a context-bounded
   16k–64k prompt plus up to 1024 output tokens, a throughput-scaled timeout,
   retained-throughput metadata, early-EOS labeling, severe-degradation and
   safety checks, and balanced-score fallback candidates.

7. Add thread-count candidates — implemented

   Add a conditional second-stage thread refinement after the normal search has
   selected its best model-placement candidate. Run it only when the candidate
   has meaningful CPU participation, such as CPU-resident MoE experts or partial
   GPU offload; skip it for fully GPU-resident configurations where thread tuning
   is unlikely to affect throughput.

   Test up to five topology-derived `--threads` and `--threads-batch`
   configurations, deduplicating values on smaller systems:

   1. llama.cpp defaults, with neither option set.
   2. Half the physical cores for generation and all physical cores for prompt processing.
   3. All physical cores for both generation and prompt processing.
   4. All physical cores for generation and all logical cores for prompt processing.
   5. All logical cores for both generation and prompt processing.

   Treat these as refinements of the selected candidate rather than multiplying
   every primary candidate by five. Prefer topology-aware core counts when the
   system exposes CPU or NUMA domains, record the exact tested values, and accept
   a new selection only when its improvement is larger than normal benchmark
   noise.

8. Improve repeated-run reliability

   Add optional confirmation runs for promising candidates and rank repeated
   measurements by their median. A `--confirm-best` workflow should reduce noise
   from GPU clocks, cache warming, desktop activity, and background CPU load.

9. Add confidence labels

   Derive a simple confidence label from validation depth, search breadth, and
   repeated measurements. Suggested labels are `provisional`, `benchmarked`,
   `confirmed`, and `full-context-validated`.

10. Remove machine-specific model names from README examples

    Replace local Qwen/Qwopus paths with generic `~/Models/<model>` examples so
    the Quickstart presents the profiler as a general-purpose tool rather than a
    configuration tailored to one machine.

11. Add basic release automation

    Add GitHub Actions checks for:

    ```bash
    cargo fmt --check
    cargo clippy --all-targets -- -D warnings
    cargo test
    cargo build --release
    cargo package --list
    ```

12. Add one copy-paste agent workflow

    Document a minimal agent command such as:

    ```bash
    llama-cpp-profiler recommend ~/Models/<model> --preset quick --agent
    ```

    Identify the stable fields an agent should consume: selected profile, exact
    command, metrics, validation level, risk, and next action.
