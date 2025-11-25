# Installing the STRAT Skill in Claude Code

## Key Differences: Claude Code vs Desktop vs API

| Method | Installation | Format |
|--------|-------------|---------|
| **Claude Code** | Filesystem directory | Folder with SKILL.md |
| **Claude Desktop** | ZIP upload via UI | .zip file |
| **Claude API** | API upload | .zip file |

**For Claude Code: NO ZIP FILE - just create a folder structure**

---

## Installation Steps for Claude Code

### Step 1: Choose Installation Location

**Personal Skill (recommended):**
- Location: `~/.claude/skills/strat-methodology/`
- Available across ALL your projects
- Not shared with team

**Project Skill:**
- Location: `.claude/skills/strat-methodology/` (within project directory)
- Checked into git
- Shared with team automatically

**For STRAT: Use Personal Skill** (unless your whole team needs it)

### Step 2: Create the Skill Directory

Copy-paste this into Claude Code:

```bash
# Create personal skill directory
mkdir -p ~/.claude/skills/strat-methodology

# Verify it was created
ls -la ~/.claude/skills/strat-methodology
```

### Step 3: Create SKILL.md File

The skill needs a `SKILL.md` file with:
1. **YAML frontmatter** (metadata)
2. **Markdown content** (instructions)

**Requirements:**
- `name`: lowercase, hyphens only, max 64 chars
- `description`: max 1024 chars, include trigger keywords
- Must start with `---` and end with `---`

### Step 4: Add Your STRAT Skill Content

You have two options:

#### Option A: Have Claude Code Create It

Tell Claude Code:
```
Create the STRAT skill file at ~/.claude/skills/strat-methodology/SKILL.md

Use this frontmatter:
---
name: strat-methodology
description: Implements STRAT trading methodology with bar classification (Type 1/2U/2D/3), pattern detection (2-1-2, 3-1-2, 2-2, Rev Strats), timeframe continuity (4 C's, MOAF, institutional flips), entry/exit mechanics, position management, and options integration. Use when building algorithmic trading systems, detecting STRAT patterns, calculating mechanical entries, implementing multi-timeframe analysis, managing positions, or integrating options with technical patterns. Requires VectorBT PRO for backtesting.
---

Then add the full STRAT methodology content below.
```

#### Option B: Copy Existing Skill Content

If you already have the STRAT skill content:
```bash
# Copy your existing skill file
cp /path/to/your/STRAT_SKILL_v2.md ~/.claude/skills/strat-methodology/SKILL.md

# Verify it copied correctly
cat ~/.claude/skills/strat-methodology/SKILL.md | head -n 10
```

### Step 5: Verify YAML Frontmatter

**Critical: The frontmatter MUST be valid YAML**

Check the file starts correctly:
```bash
head -n 10 ~/.claude/skills/strat-methodology/SKILL.md
```

Should show:
```yaml
---
name: strat-methodology
description: Implements STRAT trading methodology...
---

# STRAT Methodology

[content...]
```

**Common mistakes:**
- ❌ Missing opening `---`
- ❌ Missing closing `---`
- ❌ Tabs instead of spaces
- ❌ Name has uppercase or underscores
- ❌ Description too long (>1024 chars)

### Step 6: Add Supporting Files (Optional)

If your skill has multiple sections, create supplementary files:

```bash
# Create additional documentation files
touch ~/.claude/skills/strat-methodology/PATTERNS.md
touch ~/.claude/skills/strat-methodology/TIMEFRAMES.md
touch ~/.claude/skills/strat-methodology/EXECUTION.md
touch ~/.claude/skills/strat-methodology/VECTORBT_IMPLEMENTATION.md

# List all files
ls -la ~/.claude/skills/strat-methodology/
```

**Structure:**
```
~/.claude/skills/strat-methodology/
├── SKILL.md (main file - REQUIRED)
├── PATTERNS.md (optional)
├── TIMEFRAMES.md (optional)
├── EXECUTION.md (optional)
└── VECTORBT_IMPLEMENTATION.md (optional)
```

Reference these from SKILL.md:
```markdown
For pattern detection details, see [PATTERNS.md](PATTERNS.md)
For VectorBT integration, see [VECTORBT_IMPLEMENTATION.md](VECTORBT_IMPLEMENTATION.md)
```

### Step 7: Verify Installation

```bash
# Check skill exists
ls ~/.claude/skills/strat-methodology/SKILL.md

# View metadata
head -n 5 ~/.claude/skills/strat-methodology/SKILL.md

# Check file size (should be reasonable, not empty)
wc -l ~/.claude/skills/strat-methodology/SKILL.md
```

### Step 8: Test Auto-Triggering

**Restart Claude Code** (if it was already running), then test:

```
Help me implement STRAT bar classification using VectorBT Pro
```

**Expected behavior:**
1. Claude should automatically recognize this matches the skill description
2. Claude may read the skill file (you might see: `cat ~/.claude/skills/strat-methodology/SKILL.md`)
3. Claude should use the skill content for implementation

**If it doesn't trigger:**
- Check description has keywords: "STRAT", "bar classification", "VectorBT Pro"
- Verify YAML frontmatter is valid
- Make sure skill is in correct location

---

## Installation Checklist

Use this checklist WITH Claude Code:

```
Please help me install the STRAT skill. Let's go through this checklist:

1. Create directory: mkdir -p ~/.claude/skills/strat-methodology
2. Verify directory exists: ls -la ~/.claude/skills/strat-methodology
3. Create SKILL.md with proper frontmatter
4. Verify YAML is valid: head -n 10 ~/.claude/skills/strat-methodology/SKILL.md
5. Check file isn't empty: wc -l ~/.claude/skills/strat-methodology/SKILL.md
6. Test triggering: "Implement STRAT bar classification"
```

Claude Code will execute each step and confirm success.

---

## Folder Naming Rules

**From Anthropic docs:**

`name` field requirements:
- ✅ lowercase letters: `a-z`
- ✅ numbers: `0-9`
- ✅ hyphens: `-`
- ❌ NO uppercase
- ❌ NO underscores
- ❌ NO spaces
- ❌ NO special characters
- Maximum 64 characters

**Examples:**
- ✅ `strat-methodology`
- ✅ `trading-signals`
- ✅ `vectorbt-integration`
- ❌ `STRAT_Methodology`
- ❌ `strat_methodology`
- ❌ `strat methodology`

**The folder name should match the `name` field in YAML frontmatter.**

---

## Description Requirements

**From Anthropic docs:**

The `description` must:
- Be non-empty
- Maximum 1024 characters
- Cannot contain XML tags
- Should include BOTH:
  - What the skill does
  - When Claude should use it

**Good description template:**
```
Implements [METHODOLOGY] with [KEY FEATURES]. Use when [TRIGGER SCENARIOS].
```

**For STRAT:**
```yaml
description: Implements STRAT trading methodology with bar classification (Type 1/2U/2D/3), pattern detection (2-1-2, 3-1-2, 2-2, Rev Strats), timeframe continuity (4 C's, MOAF, institutional flips), entry/exit mechanics, position management, and options integration. Use when building algorithmic trading systems, detecting STRAT patterns, calculating mechanical entries, implementing multi-timeframe analysis, managing positions, or integrating options with technical patterns. Requires VectorBT PRO for backtesting.
```

Include trigger keywords users would naturally say:
- "STRAT"
- "bar classification"
- "pattern detection"
- "VectorBT Pro"
- "algorithmic trading"
- "timeframe continuity"

---

## Differences from Desktop/API

| Feature | Claude Code | Desktop/API |
|---------|-------------|-------------|
| **Installation** | Create folder + files | Upload .zip via UI/API |
| **Location** | `~/.claude/skills/` or `.claude/skills/` | Uploaded to platform |
| **Format** | Folder structure | .zip archive |
| **Sharing** | Via git (project skills) or plugins | Per-workspace (API) or per-user (Desktop) |
| **Editing** | Direct file editing | Re-upload new .zip |
| **Dependencies** | Install globally with permission | Pre-installed packages only |
| **Network** | Full network access | Limited (Desktop) or none (API) |

**Key difference: Claude Code uses filesystem, Desktop/API use uploads**

---

## Quick Commands Reference

```bash
# Personal skill location
~/.claude/skills/strat-methodology/

# Project skill location (within project)
.claude/skills/strat-methodology/

# Create directory
mkdir -p ~/.claude/skills/strat-methodology

# Verify installation
ls ~/.claude/skills/strat-methodology/SKILL.md

# View metadata
head -n 10 ~/.claude/skills/strat-methodology/SKILL.md

# Check file size
wc -l ~/.claude/skills/strat-methodology/SKILL.md

# Edit skill
code ~/.claude/skills/strat-methodology/SKILL.md

# Remove skill
rm -rf ~/.claude/skills/strat-methodology
```

---

## What to Tell Claude Code

**Simple installation prompt:**

```
Please help me install the STRAT methodology skill:

1. Create the directory: ~/.claude/skills/strat-methodology/
2. Create SKILL.md with this frontmatter:
   ---
   name: strat-methodology
   description: Implements STRAT trading methodology with bar classification (Type 1/2U/2D/3), pattern detection (2-1-2, 3-1-2, 2-2, Rev Strats), timeframe continuity (4 C's, MOAF, institutional flips), entry/exit mechanics, position management, and options integration. Use when building algorithmic trading systems, detecting STRAT patterns, calculating mechanical entries, implementing multi-timeframe analysis, managing positions, or integrating options with technical patterns.
   ---
3. Add the full STRAT methodology content below the frontmatter
4. Verify the YAML is valid
5. Confirm the skill is ready to use

Show me each step as you complete it.
```

Claude Code will then:
- Create the directory structure
- Create the SKILL.md file
- Validate the YAML
- Confirm installation
- Be aware of the skill for future use

---

## Troubleshooting

### Skill Doesn't Auto-Trigger

**Check 1: Description has trigger keywords**
```bash
grep "description:" ~/.claude/skills/strat-methodology/SKILL.md
```

**Check 2: YAML is valid**
```bash
head -n 10 ~/.claude/skills/strat-methodology/SKILL.md
```

**Check 3: File exists in correct location**
```bash
ls ~/.claude/skills/strat-methodology/SKILL.md
```

### Invalid YAML Error

**View the problematic frontmatter:**
```bash
head -n 15 ~/.claude/skills/strat-methodology/SKILL.md
```

Common fixes:
- Ensure opening `---` is on line 1
- Ensure closing `---` before content
- Remove tabs (use spaces only)
- Check no special characters in `name`

### Skill Content Not Loading

**Verify file isn't empty:**
```bash
wc -l ~/.claude/skills/strat-methodology/SKILL.md
```

**Check file permissions:**
```bash
ls -la ~/.claude/skills/strat-methodology/SKILL.md
```

Should be readable (`-rw-r--r--`)

---

## Summary

**For Claude Code:**
1. ✅ Create folder: `~/.claude/skills/strat-methodology/`
2. ✅ Create file: `SKILL.md` (with YAML frontmatter)
3. ✅ No zip needed - just files in a directory
4. ✅ Skills auto-trigger based on description
5. ✅ Claude reads skill files autonomously when needed

**Let Claude Code help you install it - just give it the directory name, frontmatter, and content!**
