Run one self-directed repository round with `$autonomous-loop`.

Startup:

- Use `$autonomous-loop` as the primary skill for this round.
- Read `AGENTS.md` and `.agents/loop/state.md` before taking action.
- Treat `.agents/loop/state.md` as durable mission memory, not as a rigid backlog.
- Use `.agents/loop/state.template.md` only as a structure reference when `state.md` needs to be reset or bootstrapped.
- Choose the next step from current evidence, constraints, and repository reality.

Working rules:

- You may inspect, replan, implement, and validate within the same round.
- Prefer direct experimentation over phase switching.
- Keep hard determinism only at side-effect boundaries: validation evidence, state updates, and git operations.
- Use `$git-safe` only when the current work reaches a meaningful checkpoint.

Human intervention:

- The user may rewrite any section of `.agents/loop/state.md` at any time.
- Always treat the latest user edits as authoritative goals and constraints.
- Preserve useful evidence, but rewrite stale strategy freely.

Stop conditions:

- The current mission or subgoal is complete with validation evidence.
- A meaningful checkpoint is ready to preserve.
- Progress is blocked by missing information, external dependency, or low confidence.
- More local iteration would add less value than ending the round and updating state.

Return sections in this order:

1. Mission
2. Work Performed
3. Evidence
4. State Updates
5. Git Finalization
6. Next Move
