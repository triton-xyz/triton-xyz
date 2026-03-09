---
name: loop-arch
description: "Architecture-focused planning skill for this repository. Use when the next coding slice is not ready, goals are ambiguous, backlog quality is weak, or `.agents/loop/todo.md` is missing, empty, thin, or blocked, and Codex must inspect project state, narrow scope, refresh goals or todo entries, and hand off a minimal executable slice to `loop-build` without doing the implementation itself."
---

# Loop Arch Skill

## Role Boundary

- Own backlog shaping, planning, and implementation handoff quality.
- Do not implement the target feature or test slice in this skill unless a tiny documentation or todo edit is the planning artifact itself.
- Prefer editing shared planning state over changing runtime code.

## Shared State Ownership

Read and update `.agents/loop/todo.md` as the planning source of truth.

- Keep `Goals` aligned with current repository reality.
- Keep a small ready backlog, not a single next step. When active work remains, leave `2` to `4` implementation-ready `todo` entries unless there is a clear reason not to.
- Add or revise todo entries for upcoming implementation slices.
- Ensure each todo entry is clear enough for `loop-build` to execute.
- Make each implementation-ready note specify target files or modules, expected behavior, and validation intent.
- Prefer consecutive slices in the same area so `loop-build` can complete multiple rounds before planning is needed again.
- Preserve user-authored priority and wording unless a change is required to remove ambiguity or unblock execution.

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
- Inspect recent loop cadence only as needed, for example the last `10` to `20` loop commits, to detect `loop-arch` or `loop-build` thrash or backlog starvation.
- Inspect repository delta and current quality signals.
- Identify constraints that must not be violated.

### Step 2, Frame one concrete target

- Restate the round target in one sentence.
- Define explicit non-goals for this round.
- Pick one smallest useful slice that can be built next.
- When backlog is empty, thin (`pending_count <= 1`), blocked, or low-quality, update `Goals` and replenish the todo backlog in `.agents/loop/todo.md`.
- Prefer adding a primary next slice plus `1` to `3` reserve slices that share the same goal area and can be executed without more architecture work.

### Step 3, Decide architecture direction

- List only the viable options for this round.
- Select one option and give short, technical reasons.
- Record impacted files and expected behavior changes.
- If recent history shows repeated `loop-build -> loop-arch -> loop-build` handoffs, bias toward backlog replenishment over single-entry planning.

### Step 4, Loop-build handoff

Produce a loop-build-ready handoff with:

- Scope boundary.
- File-level change plan.
- Validation plan (commands to run).
- Exit criteria for done.
- Updated todo entries in `.agents/loop/todo.md`.
- A clearly identified primary todo and any reserve todos that should allow consecutive `loop-build` rounds.
- Notes that let `loop-build` execute without guessing.

### Step 5, Finalize with loop-git skill

- Run `git status --short`.
- If changed, call `loop-git` skill and create one commit plus push for this loop-arch slice.
- Record commit hash, committed files, and push result in round output.

## Reporting Contract

When used under `loop-start`, supply content that fits the final round report order from `.agents/loop/loop.md`. When used directly, return concise sections in this order:

1. `Target`
2. `Non-goals`
3. `Decision`
4. `Changes Made`
5. `Checks Run`
6. `Todo Updates`
7. `Git Finalization`
8. `Next Step`

Use:

- `Changes Made` for planning artifacts such as goal refreshes, todo rewrites, and handoff decisions.
- `Checks Run` for repository inspection commands or any validation actually executed.
- `Next Step` to name the primary todo that `loop-build` should take next.
