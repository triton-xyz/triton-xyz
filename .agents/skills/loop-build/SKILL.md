---
name: loop-build
description: "Implementation-focused skill for this repository. Use when Codex has a clear scope and must deliver one complete, validated change slice with minimal risk."
---

# Loop Build Skill

## Shared State Contract

Read `.agents/loop/todo.md` at round start and execute from pending todo entries.

- Select entries with `status = todo`.
- After successful implementation and validation, change entry status to `done`.
- Keep edits to `todo.md` minimal and execution-oriented.

## Closing Contract

- If this round changes files, call the `loop-git` skill to stage, commit, and push.
- If there are no effective file changes, report `no changes to commit`.

## Use When

- Scope is clear enough to implement now.
- Required files and behavior changes are already identified.
- The round can end with runnable checks.
- `.agents/loop/todo.md` has pending todo entries.

## Workflow

### Step 1, Confirm slice boundary

- Read `.agents/loop/todo.md` and pick one pending entry.
- Restate target and done criteria.
- Avoid expanding scope during implementation.

### Step 2, Implement minimal delta

- Modify only required files.
- Keep changes small, readable, and reversible.
- Prefer direct, deterministic commands.

### Step 3, Validate

Run the smallest relevant checks for changed scope, for example:

- `pixi run -e default ruff check .`
- `pixi run -e default ty check .`
- Targeted script or command tied to changed behavior.

If checks cannot run, report exactly why.

### Step 4, Close round

Summarize:

- Files changed.
- Behavior delivered.
- Validation results.
- Updated todo status in `.agents/loop/todo.md`.
- Next smallest step.

### Step 5, Finalize with loop-git skill

- Run `git status --short`.
- If changed, call `loop-git` skill and create one commit plus push for this loop-build slice.
- Record commit hash, committed files, and push result in round output.

## Guardrails

- Do not introduce unrelated refactors.
- Do not mark done without validation evidence.
- Do not hide failures; report and bound them.
- Do not replan the backlog structure; backlog editing belongs to `loop-arch`.
- Do not bypass the `loop-git` skill when committing round changes.
