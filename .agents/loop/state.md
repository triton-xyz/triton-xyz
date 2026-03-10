# Mission

- Start real Triton-XYZ project development under the new loop by selecting and executing the first concrete repository task.

# Constraints

- Keep `llvm-triton/` and `third_party/triton/` read-only unless a task explicitly requires changes there.
- Prefer small, evidence-driven compiler or test slices over broad refactors.
- Validate claimed progress with real commands before marking work complete.
- Keep baseline and TTA behavior separated unless a change is required by both routes.

# Current Strategy

- Inspect the current repository state, identify one small implementation-ready compiler or test task in repo-owned code, then execute and validate that slice.

# Evidence

- 2026-03-11: `.agents/loop/loop.md` is now a stable entrypoint that explicitly invokes `$autonomous-loop` and uses `$git-safe` only for real checkpoints.
- 2026-03-11: `.agents/loop/state.template.md` now holds the reference structure, so this file can stay focused on live project state.
- 2026-03-11: No concrete compiler slice has been selected yet for the first autonomous project round.

# Next Options

- Inspect recent local changes and repository hotspots to identify the next concrete task.
- Compare baseline and TTA behavior on one focused input and pick the first mismatch worth fixing.
- Add a minimal reproducer for the first missing or incorrect TTA behavior found.

# Blockers

- The first concrete project task has not been selected yet.
