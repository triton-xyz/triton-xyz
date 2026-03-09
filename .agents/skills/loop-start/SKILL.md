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
- Treat the highest-priority pending entry as implementation-ready only if it describes a concrete code or test change and the notes provide enough execution detail, such as target files, modules, or validation commands.

## Startup Selection Rules

At round start, evaluate `.agents/loop/todo.md` first.

1. If `.agents/loop/todo.md` does not exist, select `loop-arch`.
2. If `.agents/loop/todo.md` exists but has no valid todo entries, select `loop-arch`.
3. If `pending_count = 0`, select `loop-arch`.
4. If there is at least one pending todo and the highest-priority pending entry is implementation-ready, select `loop-build`.
5. Otherwise, select `loop-arch`.

Definitions:

- `total_count`: number of valid todo entries.
- `done_count`: number of valid entries with `status = done`.
- `pending_count`: number of valid entries with `status = todo`.

Do not run `loop-arch` and `loop-build` as dual primary focus in one round.

## Workflow

### Step 1, Inspect state

- Read `AGENTS.md` and relevant local instructions.
- Read `.agents/loop/todo.md` if it exists.
- Inspect repository delta and recent check signals as needed.

### Step 2, Compute metrics

- Compute `total_count`, `done_count`, and `pending_count`.
- Record whether the top pending todo is implementation-ready and whether backlog depth is `empty`, `thin`, or `healthy`.
- Record metrics and the matched startup rule.

### Step 3, Select focus

- Select exactly one primary focus by the startup rules.
- Prefer `loop-build` whenever an executable todo exists; use `loop-arch` to replenish or repair the backlog, not as a routine alternation step.
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

- Keep startup logic reproducible and deterministic.
- Keep selection based on parsed todo data, not intuition.
- Do not duplicate `loop-arch` or `loop-build` internals in this skill.
