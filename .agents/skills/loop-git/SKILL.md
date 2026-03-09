---
name: loop-git
description: "Git finalization skill for this repository. Use when a round produced file changes and needs deterministic staging, commit, push, and result reporting."
---

# Loop Git Skill

## Use When

- The current round produced repository changes.
- `git status --short` is not empty for the intended scope.
- `loop-arch` or `loop-build` is closing a round.

## Workflow

### Step 1, Preconditions

- Ensure this is a git repository with `git rev-parse --is-inside-work-tree`.
- Resolve current branch with `git branch --show-current`.
- If branch is empty (detached HEAD), stop and report `detached HEAD, cannot choose safe push target`.
- Check remote `origin` exists with `git remote get-url origin`.

### Step 2, Inspect and scope

- Run `git status --short`.
- Confirm only intended files are included.
- If unrelated dirty files exist, stage only files for this round.

### Step 3, Stage

- Stage exact files for the current slice with `git add <paths>`.
- Recheck staged scope with `git diff --cached --name-only`.
- If staged diff is empty, stop and report `no changes to commit`.

### Step 4, Commit

- Commit once per round slice.
- Message format: `<skill>: <slice summary>`.
- Example: `loop-arch: refresh todo backlog for parser split`.
- Example: `loop-build: implement T004 cli validation`.

### Step 5, Push

- Detect upstream with `git rev-parse --abbrev-ref --symbolic-full-name @{u}`.
- If upstream exists, run `git push`.
- If upstream does not exist, run `git push -u origin <branch>`.

### Step 6, Report

- Return commit hash from `git rev-parse --short HEAD`.
- Return branch and upstream.
- Return committed files list.
- Return commit subject.
- Return push result.

## Edge and Exception Handling

- Case: no effective changes after staging.
- Action: skip commit and push.
- Report: `no changes to commit`.

- Case: detached HEAD.
- Action: do not push automatically.
- Report: `detached HEAD, manual branch selection required`.

- Case: no `origin` remote.
- Action: do not push automatically.
- Report: `origin remote missing, configure remote then retry`.

- Case: no upstream configured.
- Action: push once with `git push -u origin <branch>`.
- Report: include upstream initialization result.

- Case: push rejected as non-fast-forward.
- Action: stop, do not auto rebase or merge.
- Report: `push rejected non-fast-forward, sync branch then retry`.

- Case: push failed due to auth or network.
- Action: retry push once.
- Report: if retry still fails, include stderr summary and stop.

- Case: commit blocked by hook or policy.
- Action: do not bypass with force flags.
- Report: include failure reason and stop.

## Execution Order

Execution order is fixed: preconditions -> scope -> stage -> commit -> push -> report.

## Guardrails

- Do not use destructive commands.
- Do not amend commits unless explicitly requested.
- Do not include unrelated files in the commit.
- Do not force push in this workflow.
- If there is no effective change, skip commit and report `no changes to commit`.
