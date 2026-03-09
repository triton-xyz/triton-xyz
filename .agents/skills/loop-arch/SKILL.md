---
name: loop-arch
description: "Architecture-focused planning skill for this repository. Use when Codex must inspect current project state, narrow ambiguous goals, and define a minimal, executable implementation slice before coding."
---

# Loop Arch Skill

## Shared State Ownership

Read and update `.agents/loop/todo.md` as the planning source of truth.

- Keep `Goals` aligned with current repository reality.
- Add or revise todo entries for upcoming implementation slices.
- Ensure each todo entry is clear enough for `loop-build` to execute.

## Closing Contract

- If this round changes files, call the `loop-git` skill to stage, commit, and push.
- If there are no effective file changes, report `no changes to commit`.

## Use When

- The task goal is still ambiguous.
- Multiple implementation options exist with different tradeoffs.
- A safe loop-build slice is not yet obvious.
- Large refactor or cross-module change is likely.
- `.agents/loop/todo.md` is missing or empty.

## Workflow

### Step 1, Inspect real state

- Read `AGENTS.md` and relevant local instructions.
- Read `.agents/loop/todo.md` if it exists.
- Inspect repository delta and current quality signals.
- Identify constraints that must not be violated.

### Step 2, Frame one concrete target

- Restate the round target in one sentence.
- Define explicit non-goals for this round.
- Pick one smallest useful slice that can be built next.
- Update `Goals` and todo backlog in `.agents/loop/todo.md` when needed.

### Step 3, Decide architecture direction

- List only the viable options for this round.
- Select one option and give short, technical reasons.
- Record impacted files and expected behavior changes.

### Step 4, Loop-build handoff

Produce a loop-build-ready handoff with:

- Scope boundary.
- File-level change plan.
- Validation plan (commands to run).
- Exit criteria for done.
- Updated todo entries in `.agents/loop/todo.md`.

### Step 5, Finalize with loop-git skill

- Run `git status --short`.
- If changed, call `loop-git` skill and create one commit plus push for this loop-arch slice.
- Record commit hash, committed files, and push result in round output.

## Output Format

Return concise sections in this order:

1. `Target`
2. `Non-goals`
3. `Decision`
4. `Change Plan`
5. `Validation Plan`
6. `Done Criteria`
7. `Todo Updates`
8. `Git Finalization`
