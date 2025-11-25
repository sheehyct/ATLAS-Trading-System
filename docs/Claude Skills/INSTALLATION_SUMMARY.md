# STRAT Skill Installation Resources - Summary

## What We Created

### 1. INSTALL_STRAT_SKILL_CLAUDE_CODE.md (Complete Guide)
**Purpose**: Comprehensive reference for installing Agent Skills in Claude Code
**Size**: ~400 lines of detailed documentation

**Contents:**
- Key differences: Claude Code vs Desktop vs API
- Step-by-step installation process
- Folder naming rules (from Anthropic docs)
- Description requirements
- Supporting files structure
- Verification steps
- Troubleshooting guide
- Quick commands reference

**When to use**: First-time skill installation, troubleshooting, understanding the architecture

### 2. INSTALL_SKILL_SIMPLE.txt (Copy-Paste Prompt) ⭐ **USE THIS**
**Purpose**: Simple prompt to paste into Claude Code for assisted installation
**Size**: ~30 lines - quick and actionable

**What it does:**
- Tells Claude Code to create the skill directory
- Provides the exact frontmatter with proper description
- Walks through verification steps
- Makes Claude aware of the skill being added

**When to use**: RIGHT NOW, to actually install the skill

---

## Key Facts from Anthropic Documentation

### Claude Code Uses Filesystem (Not Zip)

| Product | Format | Location |
|---------|--------|----------|
| **Claude Code** | Folder with files | `~/.claude/skills/` or `.claude/skills/` |
| **Claude Desktop** | .zip upload | Uploaded via UI |
| **Claude API** | .zip upload | Uploaded via API |

**For Claude Code: NO ZIP FILE NEEDED**

### Required Structure

```
~/.claude/skills/strat-methodology/
├── SKILL.md (required - with YAML frontmatter)
├── PATTERNS.md (optional)
├── TIMEFRAMES.md (optional)
└── VECTORBT_IMPLEMENTATION.md (optional)
```

### SKILL.md Format

```yaml
---
name: strat-methodology
description: [What it does] + [When to use it]. Include trigger keywords.
---

# Skill Title

[Markdown content with instructions]
```

### Naming Rules

**Folder name AND `name` field must:**
- ✅ Use lowercase only
- ✅ Use hyphens for spaces
- ✅ Be 64 characters max
- ❌ NO uppercase
- ❌ NO underscores
- ❌ NO spaces

Examples:
- ✅ `strat-methodology`
- ❌ `STRAT_Methodology`
- ❌ `strat_methodology`

### Auto-Triggering

**You do NOT need to tell Claude to use the skill.**

According to Anthropic:
> "Skills are model-invoked—Claude autonomously decides when to use them based on your request and the Skill's description."

**What happens automatically:**
1. User asks: "Implement STRAT bar classification"
2. Claude detects this matches skill description
3. Claude reads skill file via bash: `cat ~/.claude/skills/strat-methodology/SKILL.md`
4. Claude uses skill content for implementation

**No explicit prompting needed!**

---

## Installation Workflow

### Option A: Use Claude Code to Install (Recommended)

1. **Open Claude Code** in your terminal
2. **Copy** `INSTALL_SKILL_SIMPLE.txt` 
3. **Paste** into Claude Code
4. **Claude will:**
   - Create the directory
   - Create SKILL.md with frontmatter
   - Ask you to add content
   - Verify installation
   - Confirm it's ready

### Option B: Manual Installation

1. Create directory:
   ```bash
   mkdir -p ~/.claude/skills/strat-methodology
   ```

2. Create SKILL.md:
   ```bash
   nano ~/.claude/skills/strat-methodology/SKILL.md
   ```

3. Add frontmatter + content:
   ```yaml
   ---
   name: strat-methodology
   description: [your description]
   ---
   
   # STRAT Methodology
   [content]
   ```

4. Save and verify:
   ```bash
   cat ~/.claude/skills/strat-methodology/SKILL.md | head -n 10
   ```

---

## Verification Steps

After installation, verify:

```bash
# 1. Directory exists
ls ~/.claude/skills/strat-methodology/

# 2. SKILL.md exists
ls ~/.claude/skills/strat-methodology/SKILL.md

# 3. Frontmatter is valid
head -n 10 ~/.claude/skills/strat-methodology/SKILL.md

# 4. File has content
wc -l ~/.claude/skills/strat-methodology/SKILL.md
```

Should show:
```
---
name: strat-methodology
description: Implements STRAT trading...
---

# STRAT Methodology
```

---

## Testing Auto-Trigger

After installation, test with Claude Code:

```
Help me implement STRAT bar classification using VectorBT Pro
```

**Expected:**
- ✅ Claude automatically recognizes STRAT request
- ✅ Claude may read skill file (you might see bash command)
- ✅ Claude uses skill content for implementation

**If it doesn't trigger:**
- Check description has keywords: "STRAT", "bar classification", "VectorBT Pro"
- Verify YAML frontmatter is valid (no syntax errors)
- Restart Claude Code

---

## Personal vs Project Skills

### Personal Skill: `~/.claude/skills/`
- Available across ALL your projects
- Not shared with team
- Use for your own workflows

### Project Skill: `.claude/skills/`
- Within specific project directory
- Checked into git
- Automatically shared with team

**For STRAT: Use Personal Skill** (unless whole team needs it)

---

## Next Steps

**Ready to install? Here's what to do:**

1. ✅ Open Claude Code in your terminal
2. ✅ Copy the contents of `INSTALL_SKILL_SIMPLE.txt`
3. ✅ Paste into Claude Code
4. ✅ Follow Claude's prompts to complete installation
5. ✅ Test by asking: "Implement STRAT bar classification"

**After installation:**
- Skill auto-triggers when you mention STRAT, patterns, VectorBT Pro, etc.
- No need to explicitly tell Claude to use the skill
- Claude reads skill files autonomously when needed
- You can edit skill anytime: `code ~/.claude/skills/strat-methodology/SKILL.md`

---

## File Locations

All installation resources are in: `/mnt/user-data/outputs/`

- **INSTALL_SKILL_SIMPLE.txt** ← Copy-paste this into Claude Code
- **INSTALL_STRAT_SKILL_CLAUDE_CODE.md** ← Reference guide
- **This file** ← Overview

---

## Remember

✅ **Claude Code uses folders, not zip files**
✅ **Skills auto-trigger based on description**
✅ **Claude reads skill files autonomously**
✅ **No explicit prompting needed after installation**

**You were right - I overcomplicated it initially!**
**The skill will auto-trigger once properly installed.**
