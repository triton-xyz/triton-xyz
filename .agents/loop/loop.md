Run one Codex autonomous loop round for this repository.

Startup:

- Invoke `loop-core` as the primary skill.
- Read `AGENTS.md` before any action.
- Read `.agents/loop/current.md` and `.agents/loop/memory.md` if they exist.
- Treat repository state as the first source of truth.

Human intervention contract:

- The user may edit `current.md` and `memory.md` at any time.
- Always treat the latest user edits as authoritative intent.
- Do not preserve stale notes when code or tests prove them wrong.

Execution contract:

- Re-ground on the current goal and repository reality at the start of every round.
- Choose one action: `act` or `reflect`.
- `act` means do the smallest useful validated step.
- `reflect` means update understanding, state, or direction when acting would be wasteful or blind.
- Keep one round focused on one main step.
- Distill only reusable notes back into shared state.
- If files changed, finalize with `loop-git` for stage, commit, and push.

Return sections in this order:

1. Current Snapshot
2. Selected Action
3. Selection Reason
4. Changes Made
5. Checks Run
6. State Updates
7. Git Finalization
8. Next Move
