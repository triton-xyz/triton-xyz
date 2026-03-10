---
name: loop-core
description: "Minimal autonomous loop skill for this repository. Use when the agent must run one self-directed round by reading `current.md`, `memory.md`, and repository state, choosing whether to act or reflect, executing one useful step, validating when needed, updating shared state, and closing with `loop-git` when files changed."
---

# Loop Core Skill

## Use When

- A round is triggered from `.agents/loop/loop.md`.
- The user wants the agent to work autonomously toward a project goal.
- The next step is not fully specified up front.

## Shared State

Read these files when they exist:

- `.agents/loop/current.md`
- `.agents/loop/memory.md`

Treat them differently:

- `current.md` is the live view: goal, current picture, next move, and risks.
- `memory.md` is compressed durable context: reusable learnings, not a transcript.

Repository state is the first source of truth. If memory disagrees with the code, tests, or git state, follow the repository and mark the stale note.

## Actions

Choose exactly one action for the round:

- `act`: Do the smallest useful step that moves the goal forward.
- `reflect`: Update understanding, notes, or direction when acting now would be wasteful or blind.

This is not a stage machine. Choose the action that best advances the goal now.

## Workflow

### Step 1, Re-ground on reality

- Read `AGENTS.md` and the loop state files that matter for this round.
- Inspect relevant code, git delta, and recent validation signals.
- Restate the current goal, constraints, and main uncertainty.
- Check whether `current.md` still matches repository reality.

### Step 2, Choose the round action

- List the few viable next moves.
- Choose `act` when one useful step is clear enough to execute now.
- Choose `reflect` when the goal, notes, or assumptions need correction before acting.
- Record why that choice wins now.

### Step 3, Execute one main slice

- Keep the round bounded to one main slice.
- If the action is `act`, edit only what is needed for that step.
- If the action is `reflect`, update `current.md`, `memory.md`, or other small planning artifacts.
- Do not preserve stale plans just because they were written earlier.
- Prefer direct evidence over speculative decomposition.

### Step 4, Validate

- Run the smallest real checks that can confirm or falsify the round outcome.
- If behavior changed under `act`, validation is expected unless the environment makes it impossible.
- Record the actual commands run and whether they passed or failed.
- If validation cannot run, explain the blocker precisely.

### Step 5, Distill shared state

Update loop state with compression in mind:

- Keep `current.md` short and current.
- Write only durable, reusable facts to `memory.md`.
- Do not copy raw console output, long reasoning traces, or redundant prose into memory files.

### Step 6, Finalize with `loop-git`

- Run `git status --short`.
- If files changed, call `loop-git`.
- Commit the round slice without pulling unrelated work into scope.

## Reporting Contract

When used under `.agents/loop/loop.md`, return concise sections in this order:

1. `Current Snapshot`
2. `Selected Action`
3. `Selection Reason`
4. `Changes Made`
5. `Checks Run`
6. `State Updates`
7. `Git Finalization`
8. `Next Move`

## Guardrails

- Keep one round focused on one main slice.
- Do not let note structure grow into a workflow engine.
- Do not treat `memory.md` as a transcript dump.
- Prefer correcting stale memory over preserving it.
- Do not let template completeness block useful work.
- Keep the repository shippable and the loop legible.
