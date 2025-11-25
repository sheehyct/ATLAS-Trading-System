# Development Resource Guides - Overview

**Date**: October 18, 2025
**Purpose**: Quick reference for navigating the resource utilization guides

---

## What You Now Have

I've created two comprehensive guides based on your project documentation:

### 1. RESOURCE_UTILIZATION_GUIDE.md (29KB)
**The "What and When" Guide**

This is your decision tree and reference manual. It tells you:
- **Which tool to use** for each type of task
- **When to use** VBT MCP tools vs filesystem vs project knowledge
- **How to choose** between OpenAI, Claude Max, and Gemini for different tasks
- **API provider setup** with exact configuration code
- **Documentation hierarchy** (what to read first, what to read when)

**Key Sections:**
- Decision Tree (start here when unsure)
- VBT MCP Tools (search, find, get_attrs, get_source, run_code)
- Filesystem Operations (read/write files, HANDOFF.md priority)
- Project Documentation Hierarchy (TIER 1-4 priority system)
- API Provider Selection (cost matrix and use cases)
- Troubleshooting Guide (common problems and solutions)

### 2. PRACTICAL_DEVELOPMENT_EXAMPLES.md (32KB)
**The "How" Guide**

This is your cookbook with copy-paste examples. It shows you:
- **Session Startup Pattern** (mandatory first steps)
- **VBT API Discovery Workflow** (complete 6-step process)
- **Implementing New Components** (with full code examples)
- **Debugging VBT Integration** (reproduce → fix → validate)
- **Data Pipeline Development** (fetch → filter → validate)
- **Risk Management Implementation** (calculate → gate → track)

**Key Sections:**
- Real-world scenarios with complete code
- Step-by-step workflows you can follow
- Copy-paste snippets for common tasks
- Integration tests and validation patterns

---

## How to Use These Guides

### When Starting Development
```
1. Read: HANDOFF.md (always first!)
2. Reference: RESOURCE_UTILIZATION_GUIDE.md Section 1 (Decision Tree)
3. Follow: PRACTICAL_DEVELOPMENT_EXAMPLES.md Section 1 (Session Startup)
```

### When Learning VBT API
```
1. Reference: RESOURCE_UTILIZATION_GUIDE.md Section 2 (VBT MCP Tools)
2. Follow: PRACTICAL_DEVELOPMENT_EXAMPLES.md Section 2 (VBT API Discovery)
3. Execute: The 6-step workflow (search → resolve → get_attrs → find → get_source → test)
```

### When Implementing Features
```
1. Follow: PRACTICAL_DEVELOPMENT_EXAMPLES.md Section 3 (Implementation Pattern)
2. Reference: RESOURCE_UTILIZATION_GUIDE.md Section 6 (Development Workflows)
3. Test: Using run_code before full implementation
```

### When Debugging
```
1. Follow: PRACTICAL_DEVELOPMENT_EXAMPLES.md Section 4 (Debugging Workflow)
2. Reference: RESOURCE_UTILIZATION_GUIDE.md Section 7 (Troubleshooting)
3. Search: VBT messages for similar issues
```

### When Unsure About Anything
```
1. Check: RESOURCE_UTILIZATION_GUIDE.md Section 1 (Decision Tree)
2. Then: Follow appropriate section in either guide
```

---

## Quick Reference: Tool Selection

**Need to understand VBT functionality?**
→ VBT MCP tools (start with `search()`)

**Need to read/write project files?**
→ Filesystem tools (read HANDOFF.md first!)

**Need to research trading concepts?**
→ Project Knowledge Search → Then web search if needed

**Need LLM for VBT embeddings?**
→ OpenAI API (only provider that works)

**Need interactive development?**
→ Claude Max (what you're using now)

**Need to test VBT code quickly?**
→ VBT `run_code()` tool

**Something broke?**
→ Read HANDOFF.md → Check CLAUDE.md → Follow troubleshooting guide

---

## Integration with Existing Docs

These guides **complement** your existing documentation:

**HANDOFF.md** - Current state (read first, every session)
**CLAUDE.md** - Development rules and constraints
**System_Architecture_Reference.md** - System design
**VBT_Pro_API_Provider_Comparison.md** - API cost analysis

**NEW: RESOURCE_UTILIZATION_GUIDE.md** - Which tool to use when
**NEW: PRACTICAL_DEVELOPMENT_EXAMPLES.md** - How to use each tool

---

## Critical Rules (Repeated from CLAUDE.md)

1. **NO EMOJIS** - ASCII only (Windows compatibility)
2. **Read HANDOFF.md FIRST** - Every session
3. **VBT docs before coding** - Don't assume
4. **Test before claiming** - No unverified claims
5. **Delete redundant files** - Don't archive

---

## Example Workflow: Implementing New Strategy Component

Let's say you need to implement ATR-based position sizing:

**Step 1: Session Startup**
```python
# PRACTICAL_EXAMPLES Section 1
read_file("HANDOFF.md")  # What's the current state?
run_code("import vbt; print('OK')")  # Environment working?
```

**Step 2: Research VBT API**
```python
# PRACTICAL_EXAMPLES Section 2 (full 6-step workflow)
search("position sizing ATR risk management")
resolve_refnames(["vbt.Portfolio", "vbt.PF.from_signals"])
get_attrs("vbt.PF.from_signals")
find(refnames=["vbt.PF.from_signals"], asset_names=["examples"])
get_source("vbt.PF.from_signals")
run_code(minimal_test)
```

**Step 3: Implement**
```python
# PRACTICAL_EXAMPLES Section 3 (full implementation example)
# Copy the position_sizing.py code
# Adapt to your needs
# Test with run_code
```

**Step 4: Integrate**
```python
# PRACTICAL_EXAMPLES Section 3 Step 5 (VBT integration test)
# Test with VBT portfolio
# Verify metrics make sense
```

**Step 5: Finalize**
```python
# Move to outputs/
# Update HANDOFF.md
```

---

## Key Insights from Your Project Docs

### VBT MCP Tools Priority
1. **search()** - When you don't know the API
2. **find()** - When you know the object, need examples
3. **get_attrs()** - When exploring what's available
4. **get_source()** - When docs are unclear
5. **run_code()** - When testing integration
6. **resolve_refnames()** - When verifying multiple refs

### Documentation Priority (from HANDOFF.md)
**TIER 1** (every session):
- docs/HANDOFF.md
- docs/CLAUDE.md

**TIER 2** (when relevant):
- VectorBT Pro Official Documentation/README.md
- docs/System_Architecture_Reference.md
- docs/VBT_Pro_API_Provider_Comparison.md

**TIER 3** (strategy-specific):
- docs/STRATEGY_*.md files
- docs/Claude Desktop Analysis/*.md

**TIER 4** (research reference):
- docs/research/*.md
- archives/*/ (failed attempts)

### API Provider Strategy (from VBT_Pro_API_Provider_Comparison.md)
- **OpenAI**: VBT embeddings + completions (~$1-10/month)
- **Claude Max**: Interactive development (already paid)
- **Gemini**: Optional backup for cost optimization (~$0-0.50/month)

**CRITICAL**: Anthropic (Claude API) does NOT support embeddings - must use OpenAI for VBT Pro features

---

## Next Steps

1. **Bookmark these guides** in your VS Code/IDE
2. **Print the Decision Tree** (RESOURCE_UTILIZATION_GUIDE.md Section 1)
3. **Practice the 6-step VBT workflow** (PRACTICAL_EXAMPLES Section 2)
4. **Follow Session Startup Pattern** (PRACTICAL_EXAMPLES Section 1)

---

## Questions to Ask When Stuck

**"Which tool do I use?"**
→ RESOURCE_UTILIZATION_GUIDE.md Section 1 (Decision Tree)

**"How do I use this tool?"**
→ PRACTICAL_DEVELOPMENT_EXAMPLES.md (find your scenario)

**"What's the current project state?"**
→ HANDOFF.md (always the answer)

**"What are the development rules?"**
→ CLAUDE.md (constraints and policies)

**"How is the system architected?"**
→ System_Architecture_Reference.md

**"Which API provider should I use?"**
→ VBT_Pro_API_Provider_Comparison.md OR RESOURCE_UTILIZATION_GUIDE.md Section 5

---

**Your New Development Workflow:**
```
HANDOFF.md → Decision Tree → Practical Example → Test → Implement → Update HANDOFF
```

---

## Document Locations

All guides are in `/mnt/user-data/outputs/`:
- RESOURCE_UTILIZATION_GUIDE.md (29KB)
- PRACTICAL_DEVELOPMENT_EXAMPLES.md (32KB)
- This file: DEVELOPMENT_GUIDES_OVERVIEW.md

**Keep these accessible** - they're designed as quick references, not one-time reads.

---

**Version:** 1.0
**Created:** October 18, 2025
**Maintainer:** Development Team
