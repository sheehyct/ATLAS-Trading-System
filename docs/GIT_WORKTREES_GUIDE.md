# Git Worktrees Guide for Parallel Strategy Development

A beginner-friendly guide for working on multiple strategies simultaneously using Git worktrees.

## What Are Worktrees?

Think of worktrees as **separate copies of your project** that all share the same Git history. Instead of:
- Cloning the repo multiple times (wasteful)
- Switching branches constantly (annoying)
- Running multiple Claude sessions in one window (UI bugs)

You get independent folders, each on their own branch, that you can open in separate VS Code windows.

```
C:\Strat_Trading_Bot\
├── vectorbt-workspace/      # Main project (main branch)
├── vectorbt-strat-crypto/   # Worktree (crypto-strategy branch)
├── vectorbt-strat-options/  # Worktree (options-strategy branch)
└── vectorbt-backtest-fixes/ # Worktree (backtest-fixes branch)
```

---

## Quick Reference Commands

| Action | Command |
|--------|---------|
| Create worktree | `git worktree add ../folder-name branch-name` |
| List worktrees | `git worktree list` |
| Remove worktree | `git worktree remove ../folder-name` |
| Open in VS Code | `code ../folder-name` |

---

## Step-by-Step Workflows

### 1. Creating a New Worktree for Strategy Work

```bash
# Make sure you're in the main project
cd C:\Strat_Trading_Bot\vectorbt-workspace

# Create a worktree with a NEW branch (most common)
git worktree add ../vectorbt-crypto-strategy -b crypto-strategy

# Or create a worktree from an EXISTING branch
git worktree add ../vectorbt-crypto-strategy crypto-strategy
```

**What this does:**
- Creates folder `C:\Strat_Trading_Bot\vectorbt-crypto-strategy`
- Creates new branch `crypto-strategy` (starting from current commit)
- You can now open this as a separate VS Code window

```bash
# Open in new VS Code window
code ../vectorbt-crypto-strategy
```

### 2. Setting Up the Worktree Environment

Each worktree needs its own Python environment:

```bash
# Navigate to the new worktree
cd ../vectorbt-crypto-strategy

# Create/sync the virtual environment
uv sync

# Now you can run Claude Code here
claude
```

### 3. Working in Multiple Worktrees

You now have completely separate workspaces:

```
Window 1: vectorbt-workspace (main branch)
  └── Terminal: claude  →  Working on ATLAS

Window 2: vectorbt-crypto-strategy (crypto-strategy branch)
  └── Terminal: claude  →  Working on crypto strategy

Window 3: vectorbt-options-backtest (options-backtest branch)
  └── Terminal: claude  →  Fixing options backtest bugs
```

Each has its own:
- VS Code file explorer
- Terminal
- Claude Code session
- Git branch

---

## Merging Your Work Back to Main

When you're done with a strategy/feature, merge it back:

### Option A: Simple Merge (Recommended for Beginners)

```bash
# Go back to main project
cd C:\Strat_Trading_Bot\vectorbt-workspace

# Make sure main is up to date
git checkout main
git pull  # if using remote

# Merge your strategy branch
git merge crypto-strategy

# If there are conflicts, VS Code will highlight them
# After resolving, commit the merge
git add .
git commit -m "Merge crypto-strategy into main"
```

### Option B: Squash Merge (Cleaner History)

This combines all your branch commits into one:

```bash
cd C:\Strat_Trading_Bot\vectorbt-workspace
git checkout main
git merge --squash crypto-strategy
git commit -m "feat: add crypto strategy implementation"
```

### Understanding Merge Conflicts

If Git says "CONFLICT", it means the same lines were changed in both branches.

```bash
# VS Code will show conflicts like this:
<<<<<<< HEAD
code from main branch
=======
code from your branch
>>>>>>> crypto-strategy

# Edit the file to keep what you want, remove the markers
# Then:
git add the-conflicted-file.py
git commit -m "Resolve merge conflict"
```

---

## Cleaning Up Worktrees

After merging, clean up:

```bash
# From main project directory
cd C:\Strat_Trading_Bot\vectorbt-workspace

# List all worktrees
git worktree list

# Remove a worktree (deletes the folder too)
git worktree remove ../vectorbt-crypto-strategy

# Delete the branch if you're done with it
git branch -d crypto-strategy
```

**If removal fails** (uncommitted changes):

```bash
# Force remove (careful - loses uncommitted work!)
git worktree remove ../vectorbt-crypto-strategy --force
```

---

## Common Scenarios

### Scenario 1: Start Fresh Strategy Development

```bash
cd C:\Strat_Trading_Bot\vectorbt-workspace
git worktree add ../vectorbt-new-strategy -b new-strategy
code ../vectorbt-new-strategy
# In new window: uv sync && claude
```

### Scenario 2: Fix Bugs While Working on Features

```bash
# You're working on crypto-strategy, but need to fix a bug in main
git worktree add ../vectorbt-hotfix -b hotfix-backtest
code ../vectorbt-hotfix
# Fix the bug, merge to main, delete worktree
```

### Scenario 3: Test Someone Else's Branch

```bash
# Fetch the branch first
git fetch origin their-branch-name
git worktree add ../vectorbt-review their-branch-name
code ../vectorbt-review
```

### Scenario 4: Abandon Work (Start Over)

```bash
# Just delete the worktree, branch keeps the history
git worktree remove ../vectorbt-failed-experiment --force
# Optionally delete the branch too
git branch -D failed-experiment
```

---

## Tips and Gotchas

### Do's
- Always `uv sync` in new worktrees before running code
- Use descriptive branch names: `crypto-strat-v2`, `fix-options-greeks`, etc.
- Commit frequently in worktrees (easy to merge later)
- Keep worktree folders as siblings to main project (easier to manage)

### Don'ts
- Don't checkout the same branch in multiple worktrees (Git won't allow it anyway)
- Don't delete worktree folders manually - use `git worktree remove`
- Don't forget to merge before removing if you want to keep the work

### If Things Go Wrong

```bash
# See status of all worktrees
git worktree list

# Prune dead worktree references (if folder was manually deleted)
git worktree prune

# Check which branch each worktree is on
git worktree list --porcelain
```

---

## Cheat Sheet for This Project

```bash
# === SETUP NEW STRATEGY WORK ===
cd C:\Strat_Trading_Bot\vectorbt-workspace
git worktree add ../vectorbt-TASKNAME -b TASKNAME
code ../vectorbt-TASKNAME
# In new window terminal:
uv sync
claude

# === MERGE WHEN DONE ===
cd C:\Strat_Trading_Bot\vectorbt-workspace
git merge TASKNAME
# or: git merge --squash TASKNAME && git commit -m "description"

# === CLEANUP ===
git worktree remove ../vectorbt-TASKNAME
git branch -d TASKNAME

# === CHECK STATUS ===
git worktree list
git branch -a
```

---

## Further Learning

Once you're comfortable with worktrees, these concepts will make more sense:
- **Rebasing** - Alternative to merging (cleaner history, more complex)
- **Cherry-picking** - Pull specific commits between branches
- **Stashing** - Temporarily save uncommitted work
- **Interactive rebase** - Clean up commits before merging

But for now, worktrees + basic merge will handle 90% of parallel development needs.
