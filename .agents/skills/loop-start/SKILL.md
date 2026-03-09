---
name: loop-start
description: "Startup and dispatch skill for autonomous development rounds. Use when running one round from .agents/loop/loop.md to inspect todo state, select loop-arch or loop-build by startup rules, delegate execution, and report round results."
---

# Loop Start Skill

## Use When

- A round is triggered with command like: `codex --yolo exec "$(< .agents/loop/loop.md)"`.
- The agent must choose one primary focus between `loop-arch` and `loop-build`.
- The round needs startup selection from `.agents/loop/todo.md`.

## Shared State

Use `.agents/loop/todo.md` as the shared dynamic state.

- Parse only valid todo entries for selection metrics.
- Valid entry format: `- [status] TID | title | notes`.
- `status` is exactly `todo` or `done`.
- `TID` matches `T[0-9]{3,}`.
- `title` and `notes` are both non-empty.
- Ignore non-matching lines for metrics.

## Startup Selection Rules

At round start, evaluate `.agents/loop/todo.md` first.

1. If `.agents/loop/todo.md` does not exist, select `loop-arch`.
2. If `.agents/loop/todo.md` exists but has no valid todo entries, select `loop-arch`.
3. If pending todo count (`status = todo`) is `>= 5`, select `loop-build`.
4. If `done_ratio` is in `[0.65, 0.75]`, randomly select `loop-arch` or `loop-build` with equal probability.
5. Otherwise, if there is any pending todo, select `loop-build`.
6. Otherwise, if all todos are done, select `loop-arch`.

Definitions:

- `total_count`: number of valid todo entries.
- `done_count`: number of valid entries with `status = done`.
- `pending_count`: number of valid entries with `status = todo`.
- `done_ratio = done_count / total_count` when `total_count > 0`, else `0`.

Randomization note:

- In the random branch, use a simple 50 or 50 coin flip.
- Record the random outcome in round output for traceability.

Do not run `loop-arch` and `loop-build` as dual primary focus in one round.

## Workflow

### Step 1, Inspect state

- Read `AGENTS.md` and relevant local instructions.
- Read `.agents/loop/todo.md` if it exists.
- Inspect repository delta and recent check signals as needed.

### Step 2, Compute metrics

- Compute `total_count`, `done_count`, `pending_count`, and `done_ratio`.
- Record metrics and the matched startup rule.
- If the random branch is used, record the random outcome explicitly.

### Step 3, Select focus

- Select exactly one primary focus by the startup rules.
- Record concise selection reason.

### Step 4, Delegate execution

- If selected `loop-arch`, call `loop-arch` and continue with its workflow.
- If selected `loop-build`, call `loop-build` and continue with its workflow.
- Delegated skill owns implementation/planning work and loop-git finalization.

### Step 5, Report round result

Return concise sections in this order:

1. `Focus Selected`
2. `Selection Reason`
3. `Todo Metrics`
4. `Changes Made`
5. `Checks Run`
6. `Todo Updates`
7. `Git Finalization`
8. `Next Step`

## Guardrails

- Keep startup logic reproducible. If the random branch is used, record the random outcome.
- Keep selection based on parsed todo data, not intuition.
- Do not duplicate `loop-arch` or `loop-build` internals in this skill.
