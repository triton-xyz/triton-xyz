---
name: git-safe
description: "Finalize an autonomous checkpoint by staging only the intended files, creating one safe commit, pushing to the current branch or `origin` without force, and reporting the exact git result."
---

# Git Safe

## Use When

- A round produced changes worth preserving.
- The caller has already decided the checkpoint is meaningful.
- Git state needs safe finalization without force or history rewriting.

## Workflow

### 1. Preconditions

- Ensure the current directory is a git repository with `git rev-parse --is-inside-work-tree`.
- Resolve the current branch with `git branch --show-current`.
- If the branch is empty, stop and report `detached HEAD, cannot choose safe push target`.
- Check that `origin` exists with `git remote get-url origin`.

### 2. Scope the checkpoint

- Run `git status --short`.
- Stage only the files that belong to the intended checkpoint.
- Exclude unrelated dirty files from staging.
- Recheck the staged scope with `git diff --cached --name-only`.
- If the staged diff is empty, stop and report `no changes to commit`.

### 3. Commit once

- Create one commit for the checkpoint.
- Message format: `<skill>: <summary>`.
- Prefer `autonomous-loop: <summary>` when called from that skill.
- Keep the subject short, concrete, and mostly lowercase.

### 4. Push safely

- Detect upstream with `git rev-parse --abbrev-ref --symbolic-full-name @{u}`.
- If upstream exists, run `git push`.
- Otherwise, run `git push -u origin <branch>`.
- Never force push, rebase, merge, or amend in this workflow.

### 5. Report

- Return the commit hash from `git rev-parse --short HEAD`.
- Return the commit subject, branch, upstream, committed files, and push result.

## Edge Cases

- If there are no effective staged changes, report `no changes to commit`.
- If `origin` is missing, report `origin remote missing, configure remote then retry`.
- If push is rejected as non-fast-forward, report `push rejected non-fast-forward, sync branch then retry`.
- If push fails for auth or network reasons, retry once, then report the stderr summary.
- If commit is blocked by a hook or policy, report the failure and stop.

## Guardrails

- Do not use destructive git commands.
- Do not include unrelated files in the commit.
- Do not bypass hooks or policy checks.
- Do not force push.
