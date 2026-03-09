Run one Codex autonomous development round for this repository.

Startup:

- Invoke `loop-start` as the primary skill.
- Read `AGENTS.md` and `.agents/loop/todo.md` before any action.
- Treat `.agents/loop/todo.md` as the shared state and human intervention channel.

Human intervention contract:

- The user may edit `Goals` and todo entries in `.agents/loop/todo.md` at any time.
- Always treat the latest user edits in `.agents/loop/todo.md` as authoritative.
- Do not overwrite or reorder user-authored intent unless required to finish the current round, and report any such change explicitly.

Execution constraints:

- Compute startup metrics from valid todo entries, then select exactly one focus skill: `loop-arch` or `loop-build`.
- Delegate execution to the selected focus skill and follow its workflow.
- Run relevant validation commands and report real outputs.
- If files changed, finalize with `loop-git` for stage, commit, and push.

Return sections in this order:

1. Focus Selected
2. Selection Reason
3. Todo Metrics (`total_count`, `done_count`, `pending_count`, `done_ratio`)
4. Human Intervention Applied
5. Changes Made
6. Checks Run
7. Todo Updates
8. Git Finalization
9. Next Step
