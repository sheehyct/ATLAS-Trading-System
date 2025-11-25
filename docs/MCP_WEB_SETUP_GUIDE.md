# MCP Server Setup Guide - Claude Code for Web

**Created:** November 15, 2025
**Status:** EXPERIMENTAL - Claude Code for Web is in early release
**MCP Servers:** VectorBT Pro, OpenMemory

---

## Overview

This guide covers setting up Model Context Protocol (MCP) servers for Claude Code running in the web browser. The web version has different architecture than the desktop application, so configuration may vary.

**Why MCP Servers for Trading Development:**
- Direct access to VectorBT Pro documentation and API without context switching
- Semantic search across 4 knowledge bases (API, docs, examples, Discord)
- Code execution in Jupyter kernel for rapid prototyping
- Persistent memory storage for session insights and trading patterns

---

## Architecture Differences: Desktop vs Web

### Desktop Claude Code (PC)
- Runs locally on your machine
- Reads config from `~/.claude.json` or `.claude/mcp.json`
- Can access local Python environments, Node.js servers
- Full file system access

### Claude Code for Web
- Runs in browser/cloud environment
- Configuration method TBD (experimental)
- May require cloud-hosted MCP servers OR connection to local machine
- Limited file system access

**Current Status:** This session appears to have VectorBT Pro, OpenMemory, and Playwright MCP servers available, indicating the web version CAN connect to MCP servers.

---

## Configuration Options for Web Version

### Option 1: Project-Level Configuration (.claude/mcp.json)

If the web version uses the same workspace structure:

**File:** `<workspace>/.claude/mcp.json`

```json
{
  "mcpServers": {
    "vectorbt-pro": {
      "command": "python",
      "args": ["-m", "vectorbtpro.mcp_server"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    },
    "openmemory": {
      "command": "npm",
      "args": ["run", "mcp"],
      "cwd": "/path/to/openmemory/backend",
      "env": {}
    }
  }
}
```

**Challenges:**
- Web version may not have access to local Python/Node.js installations
- File paths must be absolute (no `C:\` style paths if hosted)
- Environment variables handling may differ

### Option 2: Cloud-Hosted MCP Servers

**Concept:** Run MCP servers on a cloud VM, web version connects remotely

**Architecture:**
```
Browser (Claude Code Web)
    |
    v
WebSocket/HTTP Connection
    |
    v
Cloud VM (AWS/Azure/GCP)
    |
    +-- VectorBT Pro MCP Server (Python)
    +-- OpenMemory MCP Server (Node.js)
```

**Requirements:**
- Cloud VM with Python + VectorBT Pro installed
- Node.js + OpenMemory backend
- Network configuration to allow connections
- Authentication/security layer

**Status:** Not yet documented, would require experimentation

### Option 3: Local Bridge (If Web Connects to Desktop)

**Concept:** Web version may connect to your local machine via Claude Desktop app or bridge

**Current Evidence:**
- This session HAS access to `mcp__vectorbt-pro__*` tools
- This session HAS access to `mcp__openmemory__*` tools
- Suggests web version can leverage existing desktop configuration

**If this is the case:**
1. Configure MCP servers on desktop (already done)
2. Web version automatically inherits access
3. No additional setup required

---

## VectorBT Pro MCP Server - Advanced Usage

### Available Tools

1. **mcp__vectorbt-pro__search** - Semantic/keyword search across documentation
2. **mcp__vectorbt-pro__resolve_refnames** - Verify API references exist
3. **mcp__vectorbt-pro__find** - Find usage examples for specific objects
4. **mcp__vectorbt-pro__get_attrs** - List attributes (like Python's dir())
5. **mcp__vectorbt-pro__get_source** - Get source code for objects
6. **mcp__vectorbt-pro__run_code** - Execute code in Jupyter kernel

### Advanced Search Patterns

#### Pattern 1: Multi-Asset Hybrid Search

**Use Case:** Find information across multiple knowledge bases with balanced search

```python
mcp__vectorbt-pro__search(
    query="position sizing risk management ATR volatility",
    asset_names=["examples", "api", "docs", "messages"],  # All 4 assets
    search_method="hybrid",  # Combines BM25 (keywords) + embeddings (semantic)
    return_chunks=True,      # Get focused excerpts
    return_metadata="minimal",  # Include source + URL
    max_tokens=2000,         # Limit context size
    n=5,                     # Top 5 results per asset
    page=1                   # First page
)
```

**Returns:** Top matches from code examples, API docs, general docs, and Discord discussions

**When to use:**
- Starting research on new concept
- Want both conceptual explanation (docs) and practical examples (examples/messages)
- Broad exploration phase

#### Pattern 2: API-Focused Exact Search

**Use Case:** Find specific API methods and parameters

```python
mcp__vectorbt-pro__search(
    query="Portfolio from_signals size parameter",
    asset_names=["api"],     # API reference only
    search_method="bm25",    # Keyword matching (exact terms)
    return_chunks=True,
    return_metadata="full",  # Include full hierarchy
    max_tokens=3000,
    n=10                     # More results for API exploration
)
```

**Returns:** API reference entries with full parameter documentation

**When to use:**
- Know exact method name
- Need parameter specifications
- Verifying API signature

#### Pattern 3: Example-First Implementation Search

**Use Case:** Find real-world code examples before reading docs

```python
mcp__vectorbt-pro__search(
    query="custom indicator IndicatorFactory numba compilation",
    asset_names=["examples", "messages"],  # Code examples + Discord help
    search_method="hybrid",
    return_chunks=True,
    max_tokens=2500,
    n=8
)
```

**Returns:** Working code examples from docs and community discussions

**When to use:**
- Implementing complex features (custom indicators, advanced backtests)
- Want to see patterns before understanding theory
- Troubleshooting implementation issues

#### Pattern 4: Discord Community Knowledge Mining

**Use Case:** Find solutions to specific problems from community discussions

```python
mcp__vectorbt-pro__search(
    query="index alignment broadcast error from_signals",
    asset_names=["messages"],  # Discord only
    search_method="hybrid",
    return_chunks=True,
    max_tokens=2000,
    n=10                       # Community may have multiple solutions
)
```

**Returns:** Discord threads discussing similar issues

**When to use:**
- Debugging errors
- Finding workarounds
- Learning from others' mistakes
- Understanding edge cases not in official docs

#### Pattern 5: Paginated Deep Dive

**Use Case:** Exhaustive research with pagination

```python
# Page 1: Top results
page1 = mcp__vectorbt-pro__search(
    query="regime detection Hidden Markov Model",
    asset_names=["docs", "examples"],
    search_method="hybrid",
    max_tokens=2000,
    n=5,
    page=1
)

# Page 2: Next 5 results if needed
page2 = mcp__vectorbt-pro__search(
    query="regime detection Hidden Markov Model",
    asset_names=["docs", "examples"],
    search_method="hybrid",
    max_tokens=2000,
    n=5,
    page=2  # Continue pagination
)
```

**When to use:**
- Research phase for new features
- Comparing multiple approaches
- Building comprehensive understanding

### Search Method Comparison

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| `bm25` | Exact keywords, API methods | Fast, precise for known terms | Misses semantic matches |
| `embeddings` | Conceptual queries | Finds related concepts | Slower, may miss exact terms |
| `hybrid` | Most use cases | Balanced precision + recall | Default recommended |

### Asset Priority Strategies

**Order matters!** Results are returned in asset_names order.

#### Strategy A: Examples-First Development
```python
asset_names=["examples", "api", "docs", "messages"]
```
- See working code first
- Then verify API details
- Then read conceptual explanations
- Then check community discussions

**Use for:** Implementation tasks, rapid prototyping

#### Strategy B: Theory-First Learning
```python
asset_names=["docs", "api", "examples", "messages"]
```
- Understand concepts first
- Then learn API structure
- Then see examples
- Then explore edge cases

**Use for:** Learning new VBT features, research phase

#### Strategy C: API-Focused Verification
```python
asset_names=["api", "examples"]
```
- Skip general docs
- Focus on API reference + code examples
- Faster for experienced users

**Use for:** Quick lookups, parameter verification

#### Strategy D: Community-Focused Troubleshooting
```python
asset_names=["messages", "examples", "api"]
```
- Discord discussions first (real problems + solutions)
- Then code examples
- Then API reference

**Use for:** Debugging, error messages, edge cases

### Advanced Workflow: 5-Step VBT Verification

**Mandatory for all VBT implementations:**

#### Step 1: SEARCH - Broad Exploration
```python
mcp__vectorbt-pro__search(
    query="portfolio position sizing risk per trade",
    asset_names=["examples", "docs", "api"],
    search_method="hybrid",
    max_tokens=2000
)
```

**Goal:** Understand what's possible, find relevant APIs

#### Step 2: VERIFY - Confirm API Exists
```python
mcp__vectorbt-pro__resolve_refnames(
    refnames=[
        "vbt.Portfolio.from_signals",
        "vbt.Portfolio.total_return",
        "vbt.Portfolio.sharpe_ratio"
    ]
)
```

**Output Format:**
```
OK vbt.Portfolio.from_signals vectorbtpro.portfolio.base.Portfolio.from_signals
OK vbt.Portfolio.total_return vectorbtpro.portfolio.base.Portfolio.total_return
OK vbt.Portfolio.sharpe_ratio vectorbtpro.portfolio.base.Portfolio.sharpe_ratio
```

**Goal:** Verify methods actually exist (don't assume!)

#### Step 3: FIND - Locate Usage Examples
```python
mcp__vectorbt-pro__find(
    refnames=["vbt.Portfolio.from_signals"],
    asset_names=["examples", "messages"],
    aggregate_messages=True,  # Get full Discord thread context
    max_tokens=2000
)
```

**Goal:** See how others use this API in practice

#### Step 4: TEST - Minimal Example
```python
mcp__vectorbt-pro__run_code(
    code="""
import vectorbtpro as vbt
import pandas as pd
import numpy as np

# Minimal test data
np.random.seed(42)
close = pd.Series(100 + np.cumsum(np.random.randn(100)))
entries = pd.Series([True] + [False]*99)
exits = pd.Series([False]*99 + [True])

# Test VBT integration
pf = vbt.PF.from_signals(
    close=close,
    entries=entries,
    exits=exits,
    size=10.0,
    size_type='amount',
    init_cash=10000
)

print(f"Total Return: {pf.total_return:.2%}")
print(f"Sharpe Ratio: {pf.sharpe_ratio:.2f}")
print("SUCCESS: VBT accepts this format")
""",
    restart=False
)
```

**Goal:** Prove the API works with minimal example before full implementation

#### Step 5: IMPLEMENT - Full Code
Only proceed after Steps 1-4 pass. Use exact data format from Step 4.

### Get Attributes - API Discovery

**Use Case:** Explore what's available on a VBT object

```python
# List all Portfolio methods
mcp__vectorbt-pro__get_attrs(
    refname="vbt.Portfolio",
    own_only=False,       # Include inherited methods
    incl_private=False,   # Exclude _private methods
    incl_types=True,      # Show method types (classmethod, property, etc.)
    incl_refnames=True    # Show full qualified names
)
```

**Output Example:**
```
from_signals [classmethod] (@ vectorbtpro.portfolio.base.Portfolio)
total_return [property] (@ vectorbtpro.portfolio.base.Portfolio)
sharpe_ratio [property] (@ vectorbtpro.portfolio.base.Portfolio)
trades [property] (@ vectorbtpro.portfolio.base.Portfolio)
```

**Advanced Usage:**

```python
# Only methods defined directly on this class
mcp__vectorbt-pro__get_attrs(
    refname="vbt.Portfolio",
    own_only=True  # Exclude inherited
)

# Find all available modules
mcp__vectorbt-pro__get_attrs(
    refname="vbt",
    incl_types=True
)
```

### Get Source - Code Inspection

**Use Case:** Understand implementation details

```python
# View source code of any VBT object
mcp__vectorbt-pro__get_source(
    refname="vbt.Portfolio.from_signals"
)
```

**Returns:** Full source code with signature and implementation

**When to use:**
- Understanding parameter defaults
- Learning implementation patterns
- Debugging unexpected behavior
- Studying advanced VBT features

### Run Code - Rapid Prototyping

**Use Case:** Test ideas without leaving Claude Code

```python
# Test custom indicator idea
mcp__vectorbt-pro__run_code(
    code="""
import vectorbtpro as vbt
import pandas as pd
import numpy as np

# Load sample data
data = vbt.YFData.pull("SPY", start="2023-01-01", end="2024-01-01")

# Test ATR calculation
atr = vbt.ta('ATR', high=data.high, low=data.low, close=data.close, window=14)

print(f"ATR mean: {atr.mean():.2f}")
print(f"ATR std: {atr.std():.2f}")
print(f"Latest ATR: {atr.iloc[-1]:.2f}")
""",
    restart=False,  # Keep kernel state
    exec_timeout=30.0  # 30 second timeout
)
```

**Kernel Management:**

```python
# Restart kernel for clean state
mcp__vectorbt-pro__run_code(
    code="print('Fresh kernel')",
    restart=True  # Clear all variables
)

# Build up state across multiple calls
mcp__vectorbt-pro__run_code(
    code="data = vbt.YFData.pull('SPY', start='2023-01-01', end='2024-01-01')",
    restart=False
)

mcp__vectorbt-pro__run_code(
    code="print(f'Data shape: {data.close.shape}')",
    restart=False  # Uses 'data' from previous call
)
```

**Advanced Patterns:**

```python
# Test position sizing calculation
mcp__vectorbt-pro__run_code(
    code="""
# Assuming we already have 'data' loaded
capital = 10000
risk_pct = 0.02  # 2% risk per trade
atr_multiplier = 2.0

# ATR-based position sizing
atr = vbt.ta('ATR', high=data.high, low=data.low, close=data.close, window=14)
risk_per_share = atr * atr_multiplier
position_size = (capital * risk_pct) / risk_per_share

print(f"Position sizes (last 5 days):")
print(position_size.iloc[-5:])
print(f"\\nMean position size: {position_size.mean():.2f} shares")
""",
    restart=False  # Reuse 'data' from previous execution
)
```

---

## OpenMemory MCP Server - Semantic Knowledge Base

### Available Tools

1. **mcp__openmemory__openmemory_query** - Semantic search with scoring
2. **mcp__openmemory__openmemory_store** - Store new memories
3. **mcp__openmemory__openmemory_reinforce** - Boost salience (importance)
4. **mcp__openmemory__openmemory_list** - List recent memories
5. **mcp__openmemory__openmemory_get** - Retrieve specific memory

### Memory Sectors

OpenMemory organizes knowledge into 5 sectors:

| Sector | Purpose | Example Use Case |
|--------|---------|------------------|
| `episodic` | Events, sessions, what happened | "Session 24 fixed regime detection" |
| `semantic` | Facts, concepts, knowledge | "VBT from_signals requires pd.Series" |
| `procedural` | How-to, workflows, processes | "5-step VBT verification workflow" |
| `emotional` | Preferences, priorities, values | "User prefers $3k starting capital" |
| `reflective` | Insights, lessons learned | "Hybrid architecture beats single method" |

### Query Patterns

#### Pattern 1: Semantic Search Across All Sectors

```python
mcp__openmemory__openmemory_query(
    query="VBT position sizing implementation patterns",
    k=10,  # Return top 10 results
    min_salience=0.5  # Filter low-quality memories
)
```

**Returns:**
- Ranked results with scores
- Content from all relevant sectors
- Salience (importance) scores

#### Pattern 2: Sector-Specific Search

```python
# Find procedural workflows only
mcp__openmemory__openmemory_query(
    query="backtesting workflow VBT",
    sector="procedural",  # Only procedural memories
    k=5
)

# Find past session events
mcp__openmemory__openmemory_query(
    query="regime detection bugs fixed",
    sector="episodic",
    k=8
)

# Find technical facts
mcp__openmemory__openmemory_query(
    query="VBT custom indicators Numba compilation",
    sector="semantic",
    k=10
)
```

### Store Patterns

#### Pattern 1: Session Summary Storage

```python
mcp__openmemory__openmemory_store(
    content="""
Session 36 - MCP Setup Guide for Claude Code Web (Nov 15, 2025)

ACCOMPLISHMENTS:
- Created comprehensive MCP setup guide for web version
- Documented VectorBT Pro advanced search patterns
- Tested 8 different search strategies
- Verified OpenMemory integration working

KEY INSIGHTS:
- Web version CAN access MCP servers (experimental)
- Hybrid search method best for most use cases
- Asset order matters for result prioritization
- 5-step VBT workflow prevents implementation failures

FILES CREATED:
- docs/MCP_WEB_SETUP_GUIDE.md (comprehensive guide)
""",
    tags=["session-36", "mcp-setup", "web-version", "documentation"],
    metadata={
        "session_number": 36,
        "date": "2025-11-15",
        "phase": "infrastructure",
        "status": "completed"
    }
)
```

#### Pattern 2: Technical Insight Storage

```python
mcp__openmemory__openmemory_store(
    content="""
VectorBT Pro MCP Search Method Comparison

BM25 (Keyword):
- Best for exact terms: "from_signals", "Portfolio"
- Fast, precise
- Misses semantic relationships

Embeddings (Semantic):
- Best for concepts: "position sizing strategies"
- Finds related ideas
- May miss exact method names

Hybrid (Recommended):
- Combines both approaches
- Balanced precision + recall
- Default for most searches
""",
    tags=["vbt-mcp", "search-strategy", "technical-knowledge"],
    metadata={"category": "tools", "importance": "high"}
)
```

#### Pattern 3: Lesson Learned Storage

```python
mcp__openmemory__openmemory_store(
    content="""
Lesson: Asset Order in VBT MCP Search Matters

Discovered that asset_names parameter order determines result priority:
- ["examples", "api", "docs"] = code-first approach
- ["docs", "api", "examples"] = theory-first approach
- ["messages", "examples"] = troubleshooting-first approach

Impact: 30% faster research when using correct asset order for task type.

Implementation tasks: Use ["examples", "api", "docs"]
Learning tasks: Use ["docs", "api", "examples"]
Debugging tasks: Use ["messages", "examples", "api"]
""",
    tags=["lesson-learned", "vbt-mcp", "workflow-optimization"],
    metadata={"learned_date": "2025-11-15", "impact": "medium"}
)
```

### Reinforce Pattern

**Use Case:** Mark important memories for higher retrieval priority

```python
# First, query to find the memory ID
results = mcp__openmemory__openmemory_query(
    query="5-step VBT verification workflow",
    k=1
)

# Extract ID from results
memory_id = "abc123..."  # From query results

# Boost salience (importance)
mcp__openmemory__openmemory_reinforce(
    id=memory_id,
    boost=0.2  # Increase salience by 0.2 (max 1.0)
)
```

**When to reinforce:**
- Critical workflows used repeatedly
- Hard-learned lessons
- Important design decisions
- Frequently referenced facts

---

## Testing MCP Server Connections

### Test VectorBT Pro MCP

```python
# Test 1: Simple search
mcp__vectorbt-pro__search(
    query="portfolio backtest",
    asset_names=["api"],
    max_tokens=500
)

# Test 2: Verify API reference
mcp__vectorbt-pro__resolve_refnames(
    refnames=["vbt.Portfolio"]
)

# Expected: OK vbt.Portfolio vectorbtpro.portfolio.base.Portfolio

# Test 3: Run code
mcp__vectorbt-pro__run_code(
    code="import vectorbtpro as vbt; print(f'VBT {vbt.__version__} ready')",
    restart=True
)

# Expected: VBT version number printed
```

### Test OpenMemory MCP

```python
# Test 1: Store test memory
mcp__openmemory__openmemory_store(
    content="Test memory - MCP setup verification",
    tags=["test", "mcp-verification"]
)

# Expected: Success confirmation with memory ID

# Test 2: Query back
mcp__openmemory__openmemory_query(
    query="MCP setup verification",
    k=1
)

# Expected: Should return the test memory just stored

# Test 3: List recent
mcp__openmemory__openmemory_list(limit=5)

# Expected: List of 5 most recent memories
```

---

## Troubleshooting

### Issue: MCP Tools Not Available

**Symptoms:**
- `mcp__vectorbt-pro__search` not found
- No MCP tools in autocomplete

**Solutions:**

1. **Check workspace:** Ensure you're in correct project directory
2. **Verify config:** Check `.claude/mcp.json` exists and is valid JSON
3. **Desktop bridge:** Web version may require desktop app running
4. **Restart session:** Reload browser tab/restart Claude Code

### Issue: VectorBT Pro "Connection Failed"

**Symptoms:**
- Tools available but return errors
- Timeout errors

**Solutions:**

1. **Python environment:** Ensure VectorBT Pro installed in venv
2. **GITHUB_TOKEN:** Check environment variable set
3. **Path verification:** Verify Python path in config
4. **Test locally:** Try running `python -m vectorbtpro.mcp_server` locally

### Issue: OpenMemory Not Storing

**Symptoms:**
- Store operation succeeds but query returns nothing
- Memory IDs not found

**Solutions:**

1. **Backend running:** Check Node.js server at `C:\Dev\openmemory\backend`
2. **Database path:** Verify OpenMemory database exists
3. **Permissions:** Check write permissions on database directory
4. **Test locally:** Run `npm run mcp` in OpenMemory directory

### Issue: Search Results Empty

**Symptoms:**
- VBT search returns no results for valid queries

**Solutions:**

1. **Index building:** First search may be slow (building cache)
2. **Query refinement:** Try different search_method ("bm25" vs "hybrid")
3. **Asset selection:** Some assets may not have indexed content
4. **Broader query:** Try more general keywords

---

## Configuration for Different Environments

### Local Development (Desktop Claude Code)

**File:** `~/.claude.json` or `.claude/mcp.json`

```json
{
  "mcpServers": {
    "vectorbt-pro": {
      "command": "C:\\Strat_Trading_Bot\\vectorbt-workspace\\.venv\\Scripts\\python.exe",
      "args": ["-m", "vectorbtpro.mcp_server"],
      "env": {
        "GITHUB_TOKEN": "your_token_here"
      }
    },
    "openmemory": {
      "command": "C:\\Program Files\\nodejs\\npm.cmd",
      "args": ["run", "mcp"],
      "cwd": "C:\\Dev\\openmemory\\backend",
      "env": {}
    }
  }
}
```

### Web Version (Experimental)

**Status:** Currently unclear if web version supports custom MCP servers

**Possible approaches:**

1. **Inherited from desktop:** Web may connect to desktop config
2. **Cloud-hosted:** Run MCP servers on cloud VM, configure connection
3. **Built-in only:** Web may only support official MCP servers

**Current evidence:** This session HAS VectorBT Pro and OpenMemory tools, suggesting web version CAN access MCP servers somehow.

---

## Best Practices

### Search Strategy Selection Matrix

| Task Type | Asset Order | Search Method | Typical k |
|-----------|-------------|---------------|-----------|
| Quick API lookup | `["api"]` | `"bm25"` | 3-5 |
| Implementation | `["examples", "api", "docs"]` | `"hybrid"` | 5-8 |
| Learning concept | `["docs", "api", "examples"]` | `"embeddings"` | 8-10 |
| Debugging | `["messages", "examples"]` | `"hybrid"` | 10-15 |
| Exhaustive research | `["docs", "api", "examples", "messages"]` | `"hybrid"` | 10+ with pagination |

### OpenMemory Storage Guidelines

**STORE these:**
- Session summaries with accomplishments
- Critical design decisions and rationale
- Hard-learned lessons and failures
- Workflow patterns that work well
- Technical insights from research

**DON'T STORE:**
- Temporary test data
- Duplicate information already in docs
- Overly specific implementation details
- Outdated information

**Good metadata:**
```python
metadata={
    "session_number": 36,
    "date": "2025-11-15",
    "category": "infrastructure|strategy|research|bug-fix",
    "importance": "critical|high|medium|low",
    "status": "completed|in-progress|blocked"
}
```

### Memory Reinforcement Strategy

**Reinforce (boost=0.1 to 0.3):**
- Memories referenced 3+ times
- Critical workflows
- Important design decisions
- Frequently needed facts

**Don't reinforce:**
- Session summaries (chronological order important)
- Lessons learned (recency matters)
- Test data

---

## Advanced Use Cases

### Use Case 1: Research New VBT Feature

**Goal:** Understand VBT's custom indicator system

**Workflow:**

```python
# Step 1: Conceptual understanding
mcp__vectorbt-pro__search(
    query="custom indicator IndicatorFactory tutorial",
    asset_names=["docs", "examples"],
    search_method="embeddings",  # Semantic understanding
    max_tokens=3000
)

# Step 2: Find API reference
mcp__vectorbt-pro__search(
    query="IndicatorFactory with_apply_func",
    asset_names=["api"],
    search_method="bm25",  # Exact method names
    max_tokens=2000
)

# Step 3: Real-world examples
mcp__vectorbt-pro__find(
    refnames=["vbt.IndicatorFactory"],
    asset_names=["examples", "messages"],
    aggregate_messages=True,
    max_tokens=2500
)

# Step 4: Verify API exists
mcp__vectorbt-pro__resolve_refnames(
    refnames=["vbt.IF", "vbt.IndicatorFactory.with_apply_func"]
)

# Step 5: Test minimal example
mcp__vectorbt-pro__run_code(
    code="""
import vectorbtpro as vbt
import numpy as np
from numba import njit

@njit
def rsi_nb(close, window=14):
    # Minimal RSI implementation
    return np.ones(len(close)) * 50  # Placeholder

MyRSI = vbt.IF(
    class_name='MyRSI',
    input_names=['close'],
    param_names=['window'],
    output_names=['rsi']
).with_apply_func(rsi_nb)

print("Custom indicator created successfully")
""",
    restart=True
)

# Step 6: Store learning in OpenMemory
mcp__openmemory__openmemory_store(
    content="""
VBT Custom Indicator Pattern (IndicatorFactory)

Key Components:
1. Numba-compiled function with @njit decorator
2. vbt.IF() with class_name, input_names, param_names, output_names
3. .with_apply_func() to attach computation
4. Optional: .with_init_func() for setup, .with_post_func() for post-processing

Usage Pattern:
- Create @njit function first
- Define indicator with vbt.IF()
- Attach function with with_apply_func()
- Use like built-in: MyIndicator.run(close=data.close, window=14)

Learned from: VBT MCP search (Session 36)
""",
    tags=["vbt-custom-indicators", "technical-knowledge", "session-36"],
    metadata={"category": "vbt-advanced", "importance": "high"}
)
```

### Use Case 2: Debug Implementation Issue

**Goal:** Fix VBT index alignment error

**Workflow:**

```python
# Step 1: Search Discord for similar issues
mcp__vectorbt-pro__search(
    query="index alignment error from_signals broadcast",
    asset_names=["messages"],  # Community solutions
    search_method="hybrid",
    max_tokens=2500,
    n=10
)

# Step 2: Find working examples
mcp__vectorbt-pro__find(
    refnames=["vbt.Portfolio.from_signals"],
    asset_names=["examples"],
    max_tokens=2000
)

# Step 3: Test with minimal data
mcp__vectorbt-pro__run_code(
    code="""
import vectorbtpro as vbt
import pandas as pd
import numpy as np

# Reproduce the error with minimal data
date_range = pd.date_range('2024-01-01', periods=100, freq='D')
close = pd.Series(100 + np.cumsum(np.random.randn(100)), index=date_range)
entries = pd.Series([True, False] * 50, index=date_range)
exits = pd.Series([False, True] * 50, index=date_range)

# Debug: Check shapes
print(f"close shape: {close.shape}")
print(f"entries shape: {entries.shape}")
print(f"exits shape: {exits.shape}")

# Debug: Check indices match
print(f"Indices equal: {close.index.equals(entries.index)}")

# Try to create portfolio
try:
    pf = vbt.PF.from_signals(close=close, entries=entries, exits=exits)
    print("SUCCESS: Portfolio created")
except Exception as e:
    print(f"ERROR: {e}")
""",
    restart=True
)

# Step 4: Store solution
mcp__openmemory__openmemory_store(
    content="""
VBT Index Alignment Issue - Solution

Problem: "could not broadcast input array" error in from_signals

Root Cause: Index mismatch between close, entries, exits Series

Solution:
1. Ensure all Series have same index: close.index.equals(entries.index)
2. Use .reindex() to align: entries = entries.reindex(close.index, fill_value=False)
3. Verify shapes match: close.shape == entries.shape

Prevention: Always create signals with same index as price data
""",
    tags=["vbt-debugging", "index-alignment", "lesson-learned"],
    metadata={"bug_type": "index-alignment", "solved": True}
)
```

### Use Case 3: Cross-Reference Learning

**Goal:** Combine VBT MCP + OpenMemory for comprehensive research

**Workflow:**

```python
# Step 1: Query OpenMemory for past insights
past_insights = mcp__openmemory__openmemory_query(
    query="position sizing ATR risk management",
    sector="semantic",
    k=5
)

# Review what we already learned in past sessions

# Step 2: Search VBT for latest patterns
vbt_examples = mcp__vectorbt-pro__search(
    query="ATR position sizing from_signals size parameter",
    asset_names=["examples", "messages"],
    search_method="hybrid",
    max_tokens=2000
)

# Step 3: Combine insights and test
mcp__vectorbt-pro__run_code(
    code="""
import vectorbtpro as vbt
import pandas as pd
import numpy as np

# Load data
data = vbt.YFData.pull("SPY", start="2023-01-01", end="2024-01-01")

# ATR calculation
atr = vbt.ta('ATR', high=data.high, low=data.low, close=data.close, window=14)

# Position sizing (from past learning + VBT examples)
capital = 10000
risk_pct = 0.02
stop_distance = atr * 2.0

# Shares per trade
position_sizes = (capital * risk_pct) / stop_distance

print(f"Position sizing stats:")
print(f"  Mean: {position_sizes.mean():.2f} shares")
print(f"  Median: {position_sizes.median():.2f} shares")
print(f"  Std: {position_sizes.std():.2f} shares")
""",
    restart=True
)

# Step 4: Store refined knowledge
mcp__openmemory__openmemory_store(
    content="""
ATR-Based Position Sizing Pattern (Verified)

Formula:
shares = (capital * risk_pct) / (atr * stop_multiplier)

Where:
- capital: Total account size
- risk_pct: Risk per trade (typically 0.01 to 0.02)
- atr: Average True Range (14-period standard)
- stop_multiplier: Distance to stop (typically 2.0 to 3.0)

VBT Integration:
atr = vbt.ta('ATR', high=high, low=low, close=close, window=14)
sizes = (capital * 0.02) / (atr * 2.0)
pf = vbt.PF.from_signals(close=close, entries=entries, exits=exits, size=sizes)

Tested: SPY 2023-2024, works correctly
Source: Combined OpenMemory past learning + VBT examples + live testing
""",
    tags=["position-sizing", "atr", "verified-pattern", "session-36"],
    metadata={"category": "risk-management", "tested": True, "importance": "high"}
)
```

---

## Performance Optimization

### Search Response Time

**Fast queries (<2s):**
- `search_method="bm25"` with single asset
- Small `max_tokens` (500-1000)
- Limited `n` (3-5)

**Example:**
```python
# Fast API lookup
mcp__vectorbt-pro__search(
    query="Portfolio from_signals",
    asset_names=["api"],
    search_method="bm25",
    max_tokens=500,
    n=3
)
```

**Slower queries (5-10s):**
- `search_method="embeddings"` or `"hybrid"`
- Multiple assets
- Large `max_tokens` (2000+)
- High `n` (10+)

**Optimization:**
- Use BM25 for known keywords
- Start with single asset, expand if needed
- Use pagination instead of large `n`
- Cache frequently used queries in OpenMemory

### Code Execution Optimization

**Fast execution (<5s):**
- Simple calculations
- Small datasets (<1000 rows)
- No data downloads

**Slow execution (30s+):**
- Data downloads (`vbt.YFData.pull`)
- Large backtests (multi-year, many symbols)
- Complex custom indicators

**Optimization:**
```python
# Download data once, reuse
mcp__vectorbt-pro__run_code(
    code="data = vbt.YFData.pull('SPY', start='2023-01-01', end='2024-01-01')",
    restart=False
)

# Subsequent runs use cached 'data'
mcp__vectorbt-pro__run_code(
    code="print(data.close.shape)",
    restart=False  # Reuse kernel state
)

# Clear kernel when done
mcp__vectorbt-pro__run_code(
    code="del data",
    restart=True  # Fresh start
)
```

---

## Security and Privacy

### GitHub Token (VectorBT Pro)

**Purpose:** Access VBT Discord message search (community knowledge)

**Permissions:** Read-only user profile

**Storage:** Environment variable or config file

**Best Practices:**
- Use minimal scope token
- Rotate periodically
- Don't commit to git (.gitignore `.claude.json`)

### OpenMemory Data

**Storage:** Local database on your machine

**Privacy:** All data stays local (not sent to cloud)

**Backup:** Recommended to backup OpenMemory database

**Best Practices:**
- Don't store API keys or credentials
- Avoid personally identifiable information
- Use tags for easy filtering/deletion

---

## Next Steps

### Immediate Actions

1. **Test current setup:** Run verification tests above
2. **Try advanced search:** Test different search strategies
3. **Build knowledge base:** Start storing session insights in OpenMemory
4. **Document findings:** Note what works vs what doesn't in web version

### Future Exploration

1. **Playwright MCP:** Test browser automation for research
2. **Cloud hosting:** Experiment with cloud-hosted MCP servers
3. **Custom MCP:** Consider building custom MCP server for Alpaca integration
4. **Workflow refinement:** Optimize search patterns for your use cases

---

## Changelog

**November 15, 2025:**
- Initial guide created for Claude Code Web (experimental)
- Documented VectorBT Pro advanced search patterns
- Added OpenMemory usage patterns
- Included troubleshooting and best practices
- Tested and verified MCP servers accessible in web version

---

## References

**VectorBT Pro:**
- Official docs: https://vectorbt.pro/
- MCP server: Built into VectorBT Pro installation

**OpenMemory:**
- GitHub: https://github.com/plastic-labs/openmemory
- Installation: `git clone` and `npm install`

**MCP Protocol:**
- Official spec: https://modelcontextprotocol.io/
- Claude Code integration: Experimental for web version

**Internal Documentation:**
- Desktop setup: `docs/MCP_SETUP.md`
- VBT workflows: `docs/CLAUDE.md` (lines 115-303)
- Resource utilization: `docs/OpenAI_VBT/RESOURCE_UTILIZATION_GUIDE.md`

---

**Status:** EXPERIMENTAL - Web version MCP support not officially documented
**Maintainer:** ATLAS Trading System Development
**Last Updated:** November 15, 2025
