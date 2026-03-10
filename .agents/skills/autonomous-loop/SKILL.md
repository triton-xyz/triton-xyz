---
name: autonomous-loop
description: "Run a self-directed repository work loop by reading `.agents/loop/state.md`, choosing the next action from current evidence instead of a rigid backlog, executing inspect/edit/test steps in one round, updating durable state, and calling `git-safe` only when the work reaches a real checkpoint."
---

# Autonomous Loop

## Use When

- Running `.agents/loop/loop.md`.
- The repository should progress under an AI-driven loop.
- The next action should be chosen from current evidence, not a fixed todo queue.

## Core Model

Use `.agents/loop/state.md` as durable mission state.

- `Mission` and `Constraints` are authoritative.
- `Current Strategy` is a hypothesis, not a promise.
- `Evidence` stores commands, outcomes, failures, and observations worth keeping.
- `Next Options` are candidates, not a strict priority queue.
- Rewrite stale strategy or options freely when new evidence proves them wrong.
- Use `.agents/loop/state.template.md` only as a reference shape when `state.md` must be rebuilt.

## Role Boundary

- Own the full round: inspect, decide, implement, validate, and update state.
- Do not split work into mandatory planning and build phases.
- Keep hard determinism only at side-effect boundaries: validation evidence, state updates, and git operations.
- Call `git-safe` only when the round produced a checkpoint worth preserving.

## Workflow

### 1. Load state and reality

- Read `AGENTS.md`, `.agents/loop/state.md`, git status, and nearby repository context.
- Restate the active mission in one sentence.
- Identify the strongest current constraint and the biggest uncertainty.

### 2. Choose the best next move

- Pick the action most likely to reduce uncertainty or deliver concrete progress.
- Valid actions include inspecting code, editing code, running tests, updating strategy, or stopping early when blocked.
- You may ignore stale `Next Options` if current evidence points elsewhere.

### 3. Run an inner work loop

- Execute as many small inspect/edit/test cycles as needed to reach one stop condition.
- Prefer direct experimentation over speculative planning.
- Validate each claimed behavior change with real commands when feasible.
- If confidence drops, narrow scope or stop and record the blocker.

### 4. Update durable state

- Refresh only the sections that changed.
- Keep `Evidence` factual and compressed.
- Replace stale strategy instead of preserving process history for its own sake.
- Keep `Next Options` to `1` to `5` concrete candidates.

### 5. Checkpoint with `git-safe`

- If the work reached a meaningful checkpoint, call `git-safe`.
- If the round only produced dead ends or no effective changes, do not commit.

## Stop Conditions

Stop the round when one of these is true:

- The mission or current subgoal is complete with validation evidence.
- A meaningful checkpoint is ready to preserve.
- Progress is blocked by missing information, external dependency, or low confidence.
- The remaining work needs a fresh round more than more local iteration.

## Reporting Contract

When used directly or from `.agents/loop/loop.md`, return concise sections in this order:

1. `Mission`
2. `Work Performed`
3. `Evidence`
4. `State Updates`
5. `Git Finalization`
6. `Next Move`

## Guardrails

- Do not claim completion without evidence.
- Do not preserve a bad plan just because it was written earlier.
- Do not manufacture backlog structure when a hypothesis plus evidence is enough.
- Treat user edits to `.agents/loop/state.md` as authoritative.
