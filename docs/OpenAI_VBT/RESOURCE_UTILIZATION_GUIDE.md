# Resource Utilization Guide for Algorithmic Trading Development
## VectorBT Pro + Claude + MCP Tools

**Document Purpose**: Definitive guide on which resource to use for each development task
**Version**: 1.0
**Last Updated**: October 18, 2025
**Target Audience**: Development team (Claude Code sessions, human developers)

---

## Table of Contents

1. [Decision Tree: Which Resource to Use](#decision-tree-which-resource-to-use)
2. [VectorBT Pro MCP Tools](#vectorbt-pro-mcp-tools)
3. [Filesystem Operations](#filesystem-operations)
4. [Project Documentation Hierarchy](#project-documentation-hierarchy)
5. [API Provider Selection](#api-provider-selection)
6. [Development Workflow Patterns](#development-workflow-patterns)
7. [Troubleshooting Guide](#troubleshooting-guide)

---

## Decision Tree: Which Resource to Use

### Question 1: What am I trying to do?

```
START HERE
    |
    v
[Need to understand VBT API/functionality?]
    |
    YES --> Use VectorBT Pro MCP Tools (Section 2)
    |       Priority: search -> find -> get_attrs -> get_source
    |
    NO
    v
[Need to read/write project files?]
    |
    YES --> Use Filesystem Tools (Section 3)
    |       Check: HANDOFF.md first, then relevant docs
    |
    NO
    v
[Need to research trading strategies or techniques?]
    |
    YES --> Use Project Knowledge Search (Section 4)
    |       Then: Web search if needed
    |
    NO
    v
[Need LLM assistance for embeddings or completions?]
    |
    YES --> Use OpenAI API (Section 5)
    |       Exception: Use Claude Max for interactive dev
    |
    NO
    v
[Need to test VBT code quickly?]
    |
    YES --> Use VectorBT Pro run_code tool (Section 2.6)
```

---

## VectorBT Pro MCP Tools

### Overview

VectorBT Pro provides 6 MCP tools for API discovery and code execution:

1. **search** - Natural language search across all VBT assets
2. **resolve_refnames** - Verify reference names are valid
3. **find** - Find assets mentioning specific VBT objects
4. **get_attrs** - List attributes of an object (like `dir()`)
5. **get_source** - Get source code of any object
6. **run_code** - Execute code snippets in Jupyter kernel

### 2.1 VectorBT Pro: search

**When to Use:**
- Starting research on a VBT feature
- Don't know the exact API name
- Want to understand general concepts
- Looking for examples across multiple asset types

**Usage Pattern:**
```python
# Example: Understanding portfolio optimization
from VectorBT_Pro_tools import search

results = search(
    query="portfolio optimization risk management",
    asset_names=["api", "docs", "examples"],  # Search order matters
    search_method="hybrid",  # Best for balanced search
    max_tokens=2000,  # Control response size
    n=5  # Number of results per page
)
```

**Asset Types:**
- `"api"` - API reference (best for specific API queries)
- `"docs"` - General documentation (best for concepts)
- `"messages"` - Discord discussions (best for support queries)
- `"examples"` - Code examples (best for practical implementation)

**Best Practices:**
- Use 2-4 substantive keywords (no generic verbs like "discuss")
- Start with "hybrid" search method
- Use specific asset names if you know the category
- Check examples first, then API docs

**Anti-Patterns:**
- Don't use: "tell me about portfolio"
- Do use: "portfolio position sizing ATR"
- Don't search for: "VectorBT Pro portfolio" (implied)
- Do search for: "portfolio heat management overlapping positions"

---

### 2.2 VectorBT Pro: resolve_refnames

**When to Use:**
- Before using `find()` or `get_attrs()` on multiple objects
- Verifying that a class/method exists before coding
- Checking if short names are unique

**Usage Pattern:**
```python
# Verify multiple references before using them
from VectorBT_Pro_tools import resolve_refnames

results = resolve_refnames(
    refnames=["Portfolio", "vbt.PF", "AlpacaData"]
)

# Output format:
# OK Portfolio vectorbtpro.portfolio.base.Portfolio
# OK vbt.PF vectorbtpro.portfolio.base.Portfolio
# OK AlpacaData vectorbtpro.data.custom.AlpacaData
```

**Return Values:**
- `OK <input> <resolved>` - Success
- `FAIL <input>` - Invalid reference

**Best Practices:**
- Always verify before batch operations with `find()`
- Use fully-qualified names when ambiguity exists
- Common short names: `vbt.PF`, `vbt.Portfolio`, `vbt.AlpacaData`

---

### 2.3 VectorBT Pro: find

**When to Use:**
- Finding all examples/docs that mention specific VBT objects
- Discovering how a class is typically used
- Finding Discord discussions about specific methods

**Usage Pattern:**
```python
# Find all assets mentioning Portfolio and position sizing
from VectorBT_Pro_tools import find

results = find(
    refnames=["vbt.Portfolio", "vbt.PF.from_signals"],
    asset_names=["examples", "messages", "api"],  # Order matters
    aggregate_api=False,  # True = include all children
    aggregate_messages=True,  # True = include full thread
    max_tokens=2000
)
```

**Key Parameters:**

**resolve** (default: True):
- `True`: Find VBT objects (verifies refnames are valid)
- `False`: Find any string (e.g., "SQLAlchemy", "Pandas")

**aggregate_api** (default: False):
- `False`: Returns only object description
- `True`: Returns object + all children (methods, properties, etc.)
- Warning: Can return very large context for modules/classes

**aggregate_messages** (default: False):
- `False`: Returns only the matching message
- `True`: Returns entire thread (question + all replies)

**Best Practices:**
- Use `resolve_refnames()` first if using multiple references
- Set `resolve=False` when searching for external libraries
- Use `aggregate_messages=True` for support questions
- Use `aggregate_api=False` unless you need full class details

**Example Use Cases:**
```python
# Find how others use from_signals
find(refnames=["vbt.PF.from_signals"], asset_names=["examples", "messages"])

# Find SQLAlchemy integration examples
find(refnames=["SQLAlchemy"], resolve=False, asset_names=["examples"])

# Get full Portfolio class documentation
find(refnames=["vbt.Portfolio"], asset_names=["api"], aggregate_api=True)
```

---

### 2.4 VectorBT Pro: get_attrs

**When to Use:**
- Exploring available methods/properties on a class
- Discovering what's in a module
- Understanding class hierarchy (inherited attributes)

**Usage Pattern:**
```python
# Explore Portfolio class
from VectorBT_Pro_tools import get_attrs

attrs = get_attrs(
    refname="vbt.Portfolio",
    own_only=False,  # Include inherited attributes
    incl_private=False,  # Exclude _private attributes
    incl_types=True,  # Show attribute types
    incl_refnames=True  # Show where defined
)

# Output format:
# sharpe_ratio [property] (@ vectorbtpro.portfolio.base.Portfolio)
# total_return [property] (@ vectorbtpro.portfolio.base.Portfolio)
# from_signals [classmethod] (@ vectorbtpro.portfolio.base.Portfolio)
```

**Key Parameters:**

**own_only**:
- `False`: Include inherited attributes (most useful)
- `True`: Only attributes defined directly on this object

**incl_private**:
- `False`: Exclude _private and __dunder__ methods
- `True`: Show everything (rarely needed)

**incl_types**:
- `True`: Show [property], [classmethod], [function], etc.
- `False`: Just show names

**incl_refnames**:
- `True`: Show where each attribute is defined
- `False`: Just show the attribute

**Best Practices:**
- Start with defaults: `own_only=False, incl_private=False, incl_types=True`
- Use `vbt` as refname to explore the entire module
- Look for [property] vs [function] to know if you need `()`
- Compare with Python's native `help()` for full signatures

**Example Use Cases:**
```python
# What methods are available on Portfolio?
get_attrs("vbt.Portfolio", incl_types=True)

# What's in the vectorbtpro module?
get_attrs("vbt", own_only=True)

# What properties can I access on trades?
get_attrs("vbt.Portfolio.trades", incl_types=True, own_only=False)
```

---

### 2.5 VectorBT Pro: get_source

**When to Use:**
- Understanding exact implementation
- Debugging unexpected behavior
- Learning VBT patterns by reading source
- Copying/adapting complex logic

**Usage Pattern:**
```python
# Get source code for a method
from VectorBT_Pro_tools import get_source

source = get_source(refname="vbt.Portfolio.from_signals")

# Can also get source for:
# - Classes: get_source("vbt.Portfolio")
# - Modules: get_source("vbt.portfolio.base")
# - Functions: get_source("vbt.talib.SMA")
```

**Best Practices:**
- Use when documentation is unclear
- Read source to understand default parameter values
- Look for edge case handling
- Learn vectorization patterns

**When NOT to Use:**
- For basic API questions (use search/find instead)
- When examples exist (use find() to get examples)
- For high-level understanding (use docs)

---

### 2.6 VectorBT Pro: run_code

**When to Use:**
- Quick experimentation with VBT APIs
- Testing minimal examples before full implementation
- Debugging data format issues
- Verifying VBT accepts your data structure

**Usage Pattern:**
```python
# Test if VBT accepts our position sizing format
from VectorBT_Pro_tools import run_code

code = """
import vectorbtpro as vbt
import pandas as pd
import numpy as np

# Test data
close = pd.Series([100, 101, 102, 103, 104])
entries = pd.Series([True, False, False, False, False])
exits = pd.Series([False, False, False, False, True])
sizes = pd.Series([10, 10, 10, 10, 10])  # Our format

# Test VBT integration
pf = vbt.PF.from_signals(
    close=close,
    entries=entries,
    exits=exits,
    size=sizes,
    init_cash=10000
)

print(f"Total Return: {pf.total_return}")
print(f"Sharpe: {pf.sharpe_ratio}")
"""

result = run_code(code, restart=False)
print(result)
```

**Key Parameters:**

**restart** (default: False):
- `False`: Reuse existing kernel and variables
- `True`: Fresh kernel (use when testing conflicting code)

**exec_timeout** (default: None):
- `None`: No timeout
- `60`: Timeout after 60 seconds (use for long-running code)

**Critical Warnings:**
- Don't run untrusted code (can execute arbitrary code)
- Avoid side effects (no file I/O, no API calls, no state modification)
- Keep code snippets minimal (long code may block kernel)
- Don't install dependencies (environment is fixed)

**Best Practices:**
- Test VBT data format compatibility BEFORE full implementation
- Use `from vectorbtpro import *` (already imported)
- Available: `vbt`, `pd` (Pandas), `np` (NumPy), `njit` (Numba)
- Always verify output makes sense

**Development Workflow:**
```python
# Step 1: Search for relevant API
search(query="portfolio from signals position sizing")

# Step 2: Get attributes to understand what's available
get_attrs("vbt.PF.from_signals")

# Step 3: Find examples
find(refnames=["vbt.PF.from_signals"], asset_names=["examples"])

# Step 4: Test minimal example
run_code(minimal_test_code)

# Step 5: If unclear, read source
get_source("vbt.PF.from_signals")

# Step 6: Implement full code in project
```

---

## Filesystem Operations

### 3.1 MANDATORY: Read HANDOFF.md First

**CRITICAL RULE**: Before ANY coding work, ALWAYS read:
```
/mnt/user-data/uploads/docs/HANDOFF.md
```

**HANDOFF.md Contains:**
- Current session state and progress
- Recent changes and decisions
- What's working vs broken
- Immediate next steps
- File status (keep/delete/create)

**Example:**
```python
from filesystem import read_file

# FIRST ACTION in every session
handoff = read_file("/mnt/user-data/uploads/docs/HANDOFF.md")
print(handoff)

# Now you know:
# - What was just completed
# - What broke and why
# - What NOT to touch
# - What to work on next
```

---

### 3.2 Reading Project Files

**Priority Order:**
1. **HANDOFF.md** - Current state
2. **CLAUDE.md** - Development rules and guidelines
3. **System_Architecture_Reference.md** - Complete system design
4. **VBT_Pro_API_Provider_Comparison.md** - API setup guide
5. **Strategy-specific docs** - In `docs/` directory

**Pattern:**
```python
from filesystem import read_file, read_multiple_files

# Read multiple related files
files = read_multiple_files([
    "/mnt/user-data/uploads/docs/HANDOFF.md",
    "/mnt/user-data/uploads/docs/CLAUDE.md",
    "/mnt/user-data/uploads/docs/System_Architecture_Reference.md"
])

for file_path, content in files.items():
    print(f"\n{'='*60}\n{file_path}\n{'='*60}")
    print(content[:500])  # First 500 chars
```

**When to Use Each:**
- **HANDOFF.md**: Every session, immediately
- **CLAUDE.md**: When unsure about development practices
- **System_Architecture_Reference.md**: When designing new components
- **VBT_Pro_API_Provider_Comparison.md**: When setting up LLM APIs
- **Strategy docs**: When implementing specific strategies

---

### 3.3 Writing Project Files

**CRITICAL RULES:**
1. **NO EMOJIS OR UNICODE** - Windows compatibility (ASCII only)
2. **Delete redundant files** - Don't archive
3. **Keep <15 core Python files** - Simplicity over features
4. **Test before claiming success** - No unverified claims

**Pattern:**
```python
from filesystem import write_file, create_directory

# Create directory structure
create_directory("/mnt/user-data/outputs/utils")

# Write Python module
code = """
import pandas as pd
import numpy as np

def calculate_position_size(capital, risk_per_trade, atr, close):
    \"\"\"
    Calculate position size based on ATR.
    
    Parameters:
    -----------
    capital : float
        Account capital
    risk_per_trade : float
        Risk per trade as decimal (e.g., 0.02 for 2%)
    atr : pd.Series
        Average True Range values
    close : pd.Series
        Close prices
    
    Returns:
    --------
    pd.Series
        Position sizes in shares
    \"\"\"
    risk_amount = capital * risk_per_trade
    stop_distance = 2.5 * atr
    position_size = risk_amount / stop_distance
    
    # Capital constraint: max 100% of capital
    max_shares = capital / close
    position_size = position_size.clip(upper=max_shares)
    
    return position_size.astype(int)
"""

write_file(
    path="/mnt/user-data/outputs/utils/position_sizing.py",
    content=code
)
```

**Best Practices:**
- Write to `/mnt/user-data/outputs/` for final deliverables
- Use `/home/claude/` for temporary work
- Always include docstrings
- No emojis in comments or strings
- Use `\n` for newlines (not unicode line separators)

---

### 3.4 Exploring Directory Structure

**Pattern:**
```python
from filesystem import list_directory, directory_tree

# Quick listing
files = list_directory("/mnt/user-data/uploads/docs")
print(files)
# Output: [FILE] HANDOFF.md, [FILE] CLAUDE.md, [DIR] research

# Full tree view
tree = directory_tree("/mnt/user-data/uploads")
print(tree)
# Output: JSON structure showing full hierarchy
```

**Use Cases:**
- Finding relevant documentation
- Understanding project structure
- Verifying files exist before reading
- Checking for redundant files to delete

---

### 3.5 File Editing (str_replace)

**When to Use:**
- Making precise changes to existing files
- Fixing bugs in production code
- Updating configuration

**Pattern:**
```python
from filesystem import edit_file

# Make precise line-based edits
edits = [
    {
        "oldText": "risk_per_trade = 0.01",
        "newText": "risk_per_trade = 0.02"
    },
    {
        "oldText": "stop_multiplier = 2.0",
        "newText": "stop_multiplier = 2.5"
    }
]

result = edit_file(
    path="/mnt/user-data/outputs/config.py",
    edits=edits,
    dryRun=True  # Preview changes first
)
print(result)  # Shows git-style diff

# Apply if looks good
edit_file(path="/mnt/user-data/outputs/config.py", edits=edits, dryRun=False)
```

**Best Practices:**
- Always `dryRun=True` first
- Match text exactly (including whitespace)
- Make atomic changes (one logical change per edit)
- Verify with `read_file()` after editing

---

## Project Documentation Hierarchy

### 4.1 Documentation Priority

**TIER 1: MUST READ EVERY SESSION**
```
docs/HANDOFF.md                     # Current state
docs/CLAUDE.md                      # Development rules
```

**TIER 2: READ WHEN RELEVANT**
```
docs/System_Architecture_Reference.md      # System design
docs/VBT_Pro_API_Provider_Comparison.md   # API setup
VectorBT Pro Official Documentation/README.md  # VBT navigation
```

**TIER 3: STRATEGY-SPECIFIC**
```
docs/STRATEGY_1_BASELINE_RESULTS.md
docs/STRATEGY_2_ORB_RESULTS.md
docs/POSITION_SIZING_VERIFICATION.md
docs/Claude Desktop Analysis/*.md
```

**TIER 4: RESEARCH REFERENCE**
```
docs/research/*.md
docs/Algorithmic Systems Research/*.md
archives/*/   # Failed attempts (valuable learning)
```

---

### 4.2 VectorBT Pro Documentation

**PRIMARY RESOURCE:**
```
VectorBT Pro Official Documentation/README.md
```

**Navigation Workflow:**
1. Read README.md to find the right section
2. Navigate to the relevant folder/file
3. Read the ENTIRE relevant file, not snippets
4. Verify methods exist: `get_attrs()` or Python `help()`
5. Test with minimal example before full implementation

**LLM Documentation Files:**
```
LLM Docs/
├── 1 Documentation_File_Locations.md  # File mapping
├── 2 General Documentation.md         # Comprehensive general docs
├── 3 API Documentation.md             # Complete API (242k+ lines)
└── 4 Alpaca_API_LLM_Documentation.md  # Alpaca-specific
```

**When to Use:**
- **README.md**: Finding which doc to read
- **API Documentation.md**: Specific API questions (methods, parameters)
- **General Documentation.md**: Concepts, tutorials, best practices
- **Alpaca_API_LLM_Documentation.md**: AlpacaData methods

**Example Workflow:**
```
Task: Implement position sizing with VBT

1. Read: VectorBT Pro Official Documentation/README.md
   → Directs you to portfolio section

2. Search VBT MCP: search(query="position sizing from_signals")
   → Find relevant examples

3. Get attributes: get_attrs("vbt.PF.from_signals")
   → See what parameters are available

4. Find examples: find(refnames=["vbt.PF.from_signals"], asset_names=["examples"])
   → See working code

5. Test minimal: run_code(test_snippet)
   → Verify VBT accepts your format

6. Read LLM Docs: API Documentation.md section on Portfolio
   → Understand edge cases

7. Implement: Write full code
```

---

### 4.3 Using Project Knowledge Search

**When to Use:**
- Question about project-specific information
- Looking for strategy documentation
- Finding past research or decisions
- Unclear about project conventions

**Pattern:**
```python
from project_knowledge_search import search

# Search project docs
results = search(
    query="position sizing ATR strategy implementation",
    max_text_results=8,
    max_image_results=2
)

# Results include:
# - Text chunks from project docs
# - Links to source documents
# - Images from documentation
```

**Best Practices:**
- Use 3-5 substantive keywords
- Search before asking questions
- Prefer project knowledge over web search for internal matters
- Check HANDOFF.md if search returns empty (may be recent)

---

## API Provider Selection

### 5.1 Decision Matrix

**Task → Provider → Cost**

| Task | Provider | Why | Monthly Cost |
|------|----------|-----|--------------|
| **VBT Embeddings** | OpenAI | Only proven provider | $0.01 |
| **VBT Completions (dev)** | OpenAI gpt-4o-mini | Default, stable | $0.70 |
| **Interactive Development** | Claude Max | Already paid, excellent | $100* |
| **Batch Processing** | Google Gemini 2.5-flash-lite | Ultra-cheap | $0.45 |
| **Complex Strategy Analysis** | OpenAI gpt-4o | Quality matters | $0.50 |
| **Code Assistance** | Claude Max or Haiku 4.5 | Existing sub | Included* |

*Already paid via subscription

**Key Rule:**
- **OpenAI**: For VBT Pro features (embeddings + completions)
- **Claude Max**: For interactive development and strategy ideation
- **Gemini**: Optional backup for cost optimization

---

### 5.2 OpenAI Setup (Required for VBT)

**Why OpenAI:**
- Only proven provider with full VBT embedding support
- Default provider in VBT Pro (zero configuration)
- Excellent documentation
- Stable API

**Setup:**
```python
import os
os.environ["OPENAI_API_KEY"] = "your-key-here"

import vectorbtpro as vbt

# Embeddings (default)
vbt.settings.set("knowledge.chat.embeddings", "openai")
vbt.settings.set("knowledge.chat.embeddings_configs.openai.model", "text-embedding-3-small")

# Completions
vbt.settings.set("knowledge.chat.completions", "openai")
vbt.settings.set("knowledge.chat.completions_configs.openai.model", "gpt-4o-mini")
```

**Expected Cost**: $1-10/month for typical usage

---

### 5.3 Claude Max Usage (Interactive Development)

**When to Use Claude Max:**
- Strategy ideation and planning
- Interactive coding sessions (like this one)
- Complex analysis requiring extended thinking
- Code reviews and explanations
- Architecture decisions

**When NOT to Use:**
- VBT Pro embeddings (Anthropic doesn't support)
- Automated batch processing (use OpenAI/Gemini)
- Simple queries (waste of premium capability)

**Configuration (for completions only):**
```python
# CRITICAL: Anthropic CANNOT do embeddings
# Must use OpenAI for embeddings even with Anthropic completions

os.environ["ANTHROPIC_API_KEY"] = "your-key-here"
os.environ["OPENAI_API_KEY"] = "your-openai-key"  # REQUIRED

# Embeddings MUST use OpenAI
vbt.settings.set("knowledge.chat.embeddings", "openai")
vbt.settings.set("knowledge.chat.embeddings_configs.openai.model", "text-embedding-3-small")

# Completions can use Anthropic
vbt.settings.set("knowledge.chat.completions", "anthropic")
vbt.settings.set("knowledge.chat.completions_configs.anthropic.model", "claude-haiku-4-5")
```

---

### 5.4 Google Gemini Setup (Optional, Cost-Effective)

**When to Use:**
- FREE embeddings (backup for OpenAI)
- Ultra-cheap completions for batch work
- Massive context window (1M tokens)

**Setup:**
```python
os.environ["GEMINI_API_KEY"] = "your-key-here"

# Embeddings (FREE within limits)
vbt.settings.set("knowledge.chat.embeddings", "gemini")
vbt.settings.set("knowledge.chat.embeddings_configs.gemini.model", "gemini-embedding-001")

# Completions (cheap)
vbt.settings.set("knowledge.chat.completions", "gemini")
vbt.settings.set("knowledge.chat.completions_configs.gemini.model", "gemini-2.5-flash-lite")
```

**Expected Cost**: $0.00-0.50/month

---

## Development Workflow Patterns

### 6.1 Starting a New Session

```python
# STEP 1: Read HANDOFF.md (MANDATORY)
handoff = read_file("/mnt/user-data/uploads/docs/HANDOFF.md")
print("Current State:", handoff)

# STEP 2: Read CLAUDE.md (development rules)
claude_rules = read_file("/mnt/user-data/uploads/docs/CLAUDE.md")

# STEP 3: Verify environment
run_code(code='import pandas as pd; import numpy as np; print("Environment OK")')

# STEP 4: Understand context (from HANDOFF)
# - What's working?
# - What broke?
# - What's next?

# STEP 5: Plan approach
# - Which VBT APIs needed?
# - Which files to modify?
# - What tests required?
```

---

### 6.2 Implementing a New Feature

**Example: Adding Position Sizing**

```python
# STEP 1: Research VBT API
search_results = search(
    query="position sizing risk management from_signals",
    asset_names=["examples", "api", "docs"]
)

# STEP 2: Verify API availability
attrs = get_attrs("vbt.PF.from_signals")
print("Available parameters:", attrs)

# STEP 3: Find working examples
examples = find(
    refnames=["vbt.PF.from_signals"],
    asset_names=["examples"]
)

# STEP 4: Test minimal example
test_code = """
import vectorbtpro as vbt
import pandas as pd

close = pd.Series([100, 101, 102, 103, 104])
entries = pd.Series([True, False, False, False, False])
exits = pd.Series([False, False, False, False, True])
sizes = pd.Series([10, 10, 10, 10, 10])

pf = vbt.PF.from_signals(close, entries, exits, size=sizes, init_cash=10000)
print(f"Works: {pf.total_return is not None}")
"""

result = run_code(test_code)
print("Integration test:", result)

# STEP 5: Implement full feature
code = """
# Full implementation here
"""

write_file(
    path="/mnt/user-data/outputs/utils/position_sizing.py",
    content=code
)

# STEP 6: Update HANDOFF.md with results
```

---

### 6.3 Debugging an Issue

```python
# STEP 1: Reproduce the error
error_code = """
# Code that's failing
"""
error_result = run_code(error_code)
print("Error:", error_result)

# STEP 2: Check VBT documentation
search_results = search(query="error message keywords")

# STEP 3: Read source code
source = get_source("vbt.problematic_method")
print("Implementation:", source)

# STEP 4: Test fix with minimal example
fix_test = """
# Minimal test of fix
"""
fix_result = run_code(fix_test)

# STEP 5: Apply fix if successful
```

---

### 6.4 Researching Best Practices

```python
# STEP 1: Search project knowledge
project_results = project_knowledge_search.search(
    query="position sizing best practices ATR"
)

# STEP 2: Search VBT assets
vbt_results = search(
    query="position sizing risk management",
    asset_names=["docs", "messages", "examples"]
)

# STEP 3: Check Discord discussions
discussions = find(
    refnames=["position sizing"],
    resolve=False,  # Not a VBT object
    asset_names=["messages"],
    aggregate_messages=True  # Get full threads
)

# STEP 4: Read relevant research docs
research = read_file("/mnt/user-data/uploads/docs/research/position_sizing_research.md")
```

---

## Troubleshooting Guide

### 7.1 "I don't know which VBT method to use"

**Solution:**
```python
# 1. Search for the concept
search(query="your concept here")

# 2. Explore the relevant class
get_attrs("vbt.RelevantClass")

# 3. Find examples
find(refnames=["vbt.RelevantClass"], asset_names=["examples"])
```

---

### 7.2 "VBT method doesn't work as expected"

**Solution:**
```python
# 1. Verify method exists and get signature
get_attrs("vbt.ClassName")
get_source("vbt.ClassName.method_name")

# 2. Find working examples
find(refnames=["vbt.ClassName.method_name"], asset_names=["examples"])

# 3. Test minimal example
run_code(minimal_test)

# 4. Check Discord for issues
find(refnames=["vbt.ClassName.method_name"], asset_names=["messages"], aggregate_messages=True)
```

---

### 7.3 "I need to understand project architecture"

**Solution:**
```python
# 1. Read core docs
handoff = read_file("/mnt/user-data/uploads/docs/HANDOFF.md")
architecture = read_file("/mnt/user-data/uploads/docs/System_Architecture_Reference.md")

# 2. Explore directory structure
tree = directory_tree("/mnt/user-data/uploads")

# 3. Search project knowledge
results = project_knowledge_search.search(query="architecture design patterns")
```

---

### 7.4 "Setup isn't working"

**Solution:**
```python
# 1. Check API provider setup
api_guide = read_file("/mnt/user-data/uploads/docs/VBT_Pro_API_Provider_Comparison.md")

# 2. Verify environment
run_code(code="""
import vectorbtpro as vbt
print(vbt.__version__)
print(vbt.settings.get("knowledge.chat.embeddings"))
""")

# 3. Test minimal VBT operation
run_code(code='vbt.search("test")')
```

---

### 7.5 "I broke something"

**Solution:**
```python
# 1. Check what changed
handoff = read_file("/mnt/user-data/uploads/docs/HANDOFF.md")

# 2. Read development rules
rules = read_file("/mnt/user-data/uploads/docs/CLAUDE.md")

# 3. Check file status section in HANDOFF
# - What should be deleted?
# - What should be kept?
# - What was recently changed?

# 4. Test with known-good code
# - Use examples from VBT docs
# - Use code from HANDOFF.md "What's Working" section
```

---

## Summary: Quick Reference

### Task-Based Quick Lookup

**"I need to understand VBT API"**
→ `search()` → `get_attrs()` → `find()` → `get_source()`

**"I need to test VBT integration"**
→ `run_code()`

**"I need to read project files"**
→ Read `HANDOFF.md` first → Then relevant docs via `read_file()`

**"I need to write new code"**
→ Research with VBT tools → Test with `run_code()` → Write to `/mnt/user-data/outputs/`

**"I need to setup LLM APIs"**
→ Read `VBT_Pro_API_Provider_Comparison.md` → Use OpenAI for embeddings

**"I need strategy research"**
→ `project_knowledge_search()` → Then web search if needed

**"Something broke"**
→ Read `HANDOFF.md` → Check what changed → Read `CLAUDE.md` for rules

---

## Critical Rules (from CLAUDE.md)

1. **NO EMOJIS OR UNICODE** - ASCII only (Windows compatibility)
2. **Read HANDOFF.md FIRST** - Every session, immediately
3. **VBT documentation before coding** - Never assume methods exist
4. **Brutal honesty policy** - Say "I don't know" instead of guessing
5. **Delete redundant files** - Don't archive
6. **Test before claiming** - No unverified claims
7. **NYSE market hours** - Filter data BEFORE resampling
8. **Quality over speed** - Accuracy is paramount

---

**Document Version:** 1.0
**Last Updated:** October 18, 2025
**Maintainer:** Development Team
**Status:** Active Reference
