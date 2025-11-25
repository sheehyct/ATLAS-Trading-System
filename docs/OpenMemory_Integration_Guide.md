# OpenMemory Integration Guide for ATLAS Trading System

**Version:** 1.0
**Created:** October 25, 2025
**Purpose:** Hybrid context management (HANDOFF.md + OpenMemory)
**Status:** Experimental (2-week trial recommended)

---

## Executive Summary

This guide implements a **hybrid approach** to context management:
- **HANDOFF.md:** Session narratives, decisions, next actions (human-curated)
- **OpenMemory:** Queryable facts database, semantic search (machine-indexed)

**Goal:** Solve HANDOFF.md scaling issues while preserving accuracy and narrative context.

**Design Philosophy:** OpenMemory doesn't replace HANDOFF.md - it extends it. Think of HANDOFF.md as the "story" of ATLAS development and OpenMemory as the "index" that lets you quickly find specific facts within that story. This separation preserves human oversight while adding machine efficiency.

---

## CRITICAL: Deployment Scenarios

**IMPORTANT:** Choose your deployment based on whether you use the same machine or different machines.

### Scenario A: Same Machine, Different Networks (RECOMMENDED FOR MOST USERS)

**Your Setup:**
- One laptop used at work, home, coffee shop, etc.
- Different Wi-Fi networks / internet connections
- Same physical machine

**Deployment:**
- **Use localhost deployment (this guide - Parts 1-10)**
- OpenMemory runs on `localhost:8080` on your laptop
- Database stored locally: `C:/Dev/openmemory/data/atlas_memory.sqlite`
- Works offline (no internet required)
- Zero cost

**Why This Works:**
`localhost:8080` always means "this computer, port 8080" regardless of what network you're connected to. Whether you're on your home Wi-Fi, office network, or mobile hotspot, `localhost` always points to your own machine.

**Network Changes Don't Affect Localhost:**
- Work network: `localhost:8080` → Your laptop
- Home network: `localhost:8080` → Your laptop
- Coffee shop: `localhost:8080` → Your laptop

**Database is Local:**
Your SQLite database is a file on your laptop's hard drive. It moves with your laptop everywhere you go. No cloud sync needed.

**Verdict:** If you use ONE laptop for all ATLAS development, use localhost deployment (current guide).

---

### Scenario B: Different Machines (Advanced - VPS Required)

**Your Setup:**
- Desktop at home + laptop at work (two different computers)
- OR multiple team members working on ATLAS
- Need same database accessible from both machines

**Deployment:**
- **Use VPS deployment (see Part 11 below)**
- OpenMemory runs on cloud server (DigitalOcean, AWS, etc.)
- Database centralized on VPS
- Accessible from any machine with internet
- Cost: $5-10/month

**Why VPS is Needed:**
Each computer has its own `localhost`. Your work desktop's `localhost` is different from your home desktop's `localhost`. To share the database, you need a centralized server both machines can reach.

**Verdict:** If you use MULTIPLE computers for ATLAS development, use VPS deployment (Part 11).

---

### Quick Decision Tree

```
Question: Do you use the SAME physical machine for all ATLAS work?

YES (one laptop everywhere)
  → Use localhost deployment (Parts 1-10)
  → Works at work, home, anywhere
  → Zero cost
  → Stop reading, start Part 1

NO (desktop at home + laptop at work)
  → Use VPS deployment (Part 11)
  → $5-10/month cost
  → Skip to Part 11 for VPS setup
```

---

## Part 1: Installation & Setup (Windows)

### Prerequisites Checklist

```bash
# Verify Node.js 20+ installed
node --version  # Should show v20.x.x or higher

# If not installed, download from https://nodejs.org/
# Choose LTS version for Windows

# Verify Git available
git --version

# Verify SQLite 3 (bundled with OpenMemory, but check)
```

### Step 1: Clone OpenMemory Repository

```bash
# Navigate to development directory (outside ATLAS workspace)
cd C:\Dev

# Clone repository
git clone https://github.com/caviraoss/openmemory.git
cd openmemory

# Verify structure
ls
# Should see: backend/, frontend/, README.md, etc.
```

### Step 2: Configure Backend

```bash
cd backend

# Copy environment template
cp .env.example .env

# Open .env in text editor
notepad .env
```

**Recommended .env Configuration for ATLAS:**

```bash
# OpenMemory Configuration for ATLAS Trading System

# Port (default 8080, change if conflict)
PORT=8080

# Embedding Provider (start with OpenAI for quality, switch to Ollama later for cost)
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here

# Alternative: Use Ollama for zero-cost local embeddings
# EMBEDDING_PROVIDER=ollama
# OLLAMA_MODEL=nomic-embed-text
# OLLAMA_BASE_URL=http://localhost:11434

# Memory Decay (default 0.02 - slower decay for trading knowledge)
DECAY_LAMBDA=0.01

# Authentication (generate strong bearer token)
BEARER_TOKEN=atlas_openmemory_dev_token_2025

# Database Path (Windows-compatible)
DB_PATH=C:/Dev/openmemory/data/atlas_memory.sqlite

# Enable MCP Server for Claude Code integration
MCP_ENABLED=true
MCP_PORT=8081
```

**Embedding Provider Choice:** Start with OpenAI embeddings ($0.13 per 1M tokens) for quality during the 2-week trial. If successful, switch to Ollama (local, zero-cost) for production. Trading system context is valuable enough to justify initial cost for accurate semantic search.

### Step 3: Install Dependencies & Start Server

```bash
# Install packages
npm install

# Development mode (auto-restart on changes)
npm run dev

# Production mode (for long-term use)
npm run build
npm start
```

**Expected Output:**
```
OpenMemory API listening on http://localhost:8080
MCP Server listening on http://localhost:8081
SQLite database initialized at C:/Dev/openmemory/data/atlas_memory.sqlite
```

### Step 4: Test Installation

```bash
# Open new terminal (keep server running in first terminal)

# Health check
curl http://localhost:8080/health

# Expected response: {"status":"ok","version":"1.2.0"}

# Test memory storage
curl -X POST http://localhost:8080/memory/add \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer atlas_openmemory_dev_token_2025" \
  -d "{\"content\": \"OpenMemory installation test successful\"}"

# Test memory query
curl -X POST http://localhost:8080/memory/query \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"installation test\"}"

# Should return the test memory with similarity score
```

---

## Part 2: ATLAS Memory Schema Design

### Memory Type Definitions for Trading System

OpenMemory uses 5 memory types. Here's how they map to ATLAS development:

| Memory Type | ATLAS Usage | Examples |
|-------------|-------------|----------|
| **Episodic** | Session events, testing activities, bug fixes | "Tested NVDA ORB 5-min range on Oct 24, 2025", "Fixed RTH filtering bug in Session 5" |
| **Semantic** | Validated facts, empirical findings, parameters | "5-min opening range outperforms 30-min by 251x", "2.0x volume threshold mandatory for ORB" |
| **Procedural** | Workflows, rules, best practices | "Follow 5-step VBT verification before implementing", "Always test on volatile stocks for ORB" |
| **Emotional** | User frustrations, preferences, concerns | "User concerned about zero trades issue", "Accuracy over speed mandate" |
| **Reflective** | Insights, lessons learned, architectural decisions | "ORB is symbol-specific, requires selective universe", "Walk-forward validation premature" |

### Metadata Schema for ATLAS

**Standard Fields (all memories):**
```json
{
  "project": "ATLAS",
  "session": 8,
  "date": "2025-10-24",
  "phase": "Phase 3",
  "component": "strategies/orb.py"
}
```

**Strategy-Specific Fields:**
```json
{
  "strategy": "ORB",
  "symbol": "NVDA",
  "parameter": "opening_range",
  "test_type": "empirical_validation"
}
```

**Finding-Specific Fields:**
```json
{
  "finding_type": "performance_metric",
  "metric_name": "sharpe_ratio",
  "metric_value": 0.13,
  "target_value": 2.0,
  "status": "below_target"
}
```

---

## Part 3: Populating OpenMemory from Session 8

### Example: Session 8 ORB Validation

**Helper Script (save as `scripts/populate_openmemory.sh`):**

```bash
#!/bin/bash
# ATLAS OpenMemory Population Script
# Usage: ./scripts/populate_openmemory.sh session8

API_URL="http://localhost:8080/memory/add"
AUTH_TOKEN="atlas_openmemory_dev_token_2025"

# Function to add memory
add_memory() {
    local content="$1"
    local type="$2"
    local metadata="$3"

    curl -X POST "$API_URL" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $AUTH_TOKEN" \
      -d "{
        \"content\": \"$content\",
        \"type\": \"$type\",
        \"metadata\": $metadata
      }"

    echo "" # Newline for readability
}

# Session 8: ORB Empirical Validation
SESSION=8
DATE="2025-10-24"
PHASE="Phase3"

# Episodic: Test activities
add_memory \
  "Tested NVDA ORB with 5-min opening range over 6 months (Jan-Jun 2024): 33 trades, 54.5% win rate, +2.51% return, Sharpe 0.13" \
  "episodic" \
  "{\"session\": $SESSION, \"date\": \"$DATE\", \"strategy\": \"ORB\", \"symbol\": \"NVDA\", \"test_period\": \"6_months\"}"

add_memory \
  "Tested NVDA ORB with 30-min opening range: 32 trades, 37.5% win rate, +0.01% return, Sharpe 0.00" \
  "episodic" \
  "{\"session\": $SESSION, \"date\": \"$DATE\", \"strategy\": \"ORB\", \"symbol\": \"NVDA\", \"parameter\": \"opening_range_30min\"}"

add_memory \
  "Tested TSLA ORB 5-min range: 32 trades, 40.6% win rate, -3.94% return - UNPROFITABLE" \
  "episodic" \
  "{\"session\": $SESSION, \"date\": \"$DATE\", \"strategy\": \"ORB\", \"symbol\": \"TSLA\", \"result\": \"unprofitable\"}"

add_memory \
  "Tested AAPL ORB 5-min range: 22 trades, 18.2% win rate, -4.01% return - UNPROFITABLE" \
  "episodic" \
  "{\"session\": $SESSION, \"date\": \"$DATE\", \"strategy\": \"ORB\", \"symbol\": \"AAPL\", \"result\": \"unprofitable\"}"

# Semantic: Validated facts
add_memory \
  "5-min opening range outperforms 30-min range by 251x in returns on NVDA" \
  "semantic" \
  "{\"session\": $SESSION, \"finding_type\": \"parameter_validation\", \"parameter\": \"opening_range\", \"optimal_value\": \"5min\"}"

add_memory \
  "Volume threshold 2.0x superior to 1.5x: 2.4x better returns, higher win rate (54.5% vs 43.6%)" \
  "semantic" \
  "{\"session\": $SESSION, \"finding_type\": \"parameter_validation\", \"parameter\": \"volume_threshold\", \"optimal_value\": \"2.0x\"}"

add_memory \
  "ORB win rate on NVDA: 54.5% (exceeds specification target of 15-25%)" \
  "semantic" \
  "{\"session\": $SESSION, \"metric\": \"win_rate\", \"value\": 0.545, \"status\": \"exceeds_target\"}"

add_memory \
  "ORB Sharpe ratio on NVDA: 0.13 (below specification target of 2.0)" \
  "semantic" \
  "{\"session\": $SESSION, \"metric\": \"sharpe_ratio\", \"value\": 0.13, \"target\": 2.0, \"status\": \"below_target\"}"

# Procedural: Workflows and rules
add_memory \
  "Always test ORB strategy on volatile stocks, not stable indices like SPY" \
  "procedural" \
  "{\"session\": $SESSION, \"rule_type\": \"symbol_selection\", \"strategy\": \"ORB\"}"

add_memory \
  "Use session-scoped pytest fixtures to reduce API calls during testing" \
  "procedural" \
  "{\"session\": $SESSION, \"rule_type\": \"testing_best_practice\", \"component\": \"tests/conftest.py\"}"

add_memory \
  "Follow 5-step VBT Pro verification workflow before implementing any VBT feature" \
  "procedural" \
  "{\"session\": $SESSION, \"rule_type\": \"development_workflow\", \"critical\": true}"

# Reflective: Insights and lessons
add_memory \
  "ORB strategy is symbol-specific: works on NVDA but fails on TSLA and AAPL - requires selective universe screening" \
  "reflective" \
  "{\"session\": $SESSION, \"insight_type\": \"strategy_limitation\", \"strategy\": \"ORB\"}"

add_memory \
  "6 months may be insufficient validation period - NVDA results could be lucky period" \
  "reflective" \
  "{\"session\": $SESSION, \"insight_type\": \"validation_concern\", \"recommendation\": \"test_longer_period\"}"

add_memory \
  "Specification targets (Sharpe >2.0, R:R >3:1) not met - may need optimization or longer period" \
  "reflective" \
  "{\"session\": $SESSION, \"insight_type\": \"performance_gap\", \"strategy\": \"ORB\"}"

# Emotional: User preferences and concerns
add_memory \
  "User mandate: accuracy over speed every time" \
  "emotional" \
  "{\"session\": $SESSION, \"preference_type\": \"development_priority\"}"

add_memory \
  "User concerned about zero trades issue from past sessions" \
  "emotional" \
  "{\"session\": $SESSION, \"concern_type\": \"bug_recurrence\"}"

echo "Session 8 memories populated successfully!"
```

**Make script executable and run:**
```bash
# Windows (Git Bash)
chmod +x scripts/populate_openmemory.sh
./scripts/populate_openmemory.sh

# Windows (PowerShell) - convert to .ps1 or run commands manually
```

---

## Part 4: Query Patterns for ATLAS Workflows

### Common Queries for Development Sessions

**1. Strategy Performance Lookup**
```bash
# "What do we know about NVDA performance?"
curl -X POST http://localhost:8080/memory/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "NVDA ORB performance metrics results",
    "top_k": 5
  }'
```

**2. Parameter Validation History**
```bash
# "What parameters have we validated?"
curl -X POST http://localhost:8080/memory/query \
  -d '{
    "query": "parameter validation optimal values",
    "filters": {"finding_type": "parameter_validation"}
  }'
```

**3. Symbol Testing History**
```bash
# "What symbols have we tested?"
curl -X POST http://localhost:8080/memory/query \
  -d '{
    "query": "symbol testing results profitable unprofitable",
    "top_k": 10
  }'
```

**4. Development Rules and Workflows**
```bash
# "What workflows must I follow?"
curl -X POST http://localhost:8080/memory/query \
  -d '{
    "query": "development workflow rules best practices",
    "filters": {"type": "procedural"}
  }'
```

**5. Lessons Learned**
```bash
# "What insights have we discovered?"
curl -X POST http://localhost:8080/memory/query \
  -d '{
    "query": "strategy limitations insights lessons learned",
    "filters": {"type": "reflective"}
  }'
```

**6. Session-Specific Recall**
```bash
# "What happened in Session 8?"
curl -X POST http://localhost:8080/memory/query \
  -d '{
    "query": "Session 8 ORB validation empirical testing",
    "filters": {"session": 8}
  }'
```

---

## Part 5: Hybrid Workflow Integration

### Updated Session Start Routine

**Old Workflow (HANDOFF.md only):**
```
1. Read HANDOFF.md (focus on previous session)
2. Read CLAUDE.md (refresh rules)
3. Read System_Architecture_Reference.md (if needed)
4. Verify environment
5. Begin work
```

**New Hybrid Workflow:**
```
1. Read HANDOFF.md Session Summary (condensed narrative - 20 lines)
2. Query OpenMemory for specific context needed:
   - "What parameters validated in last session?"
   - "What symbols tested so far?"
   - "What are current performance metrics?"
3. Read CLAUDE.md (refresh rules)
4. Verify environment
5. Begin work with full context
```

### Updated Session End Routine

**Old Workflow:**
```
1. Update HANDOFF.md with comprehensive session summary
2. Commit changes
3. Create next session startup prompt
```

**New Hybrid Workflow:**
```
1. Write HANDOFF.md Session Summary (narrative only - 30 lines):
   - Objective
   - Key decisions made
   - Critical insights
   - Next steps
   - Reference: "Facts stored in OpenMemory with tag sessionN"

2. Populate OpenMemory with structured facts (10 minutes):
   - Episodic: What we did
   - Semantic: What we learned (facts/metrics)
   - Procedural: New rules established
   - Reflective: Insights discovered
   - Emotional: User preferences expressed

3. Commit changes

4. Create next session startup prompt (include OpenMemory queries)
```

### HANDOFF.md Template (Condensed for Hybrid Approach)

```markdown
### Session 9: PortfolioManager Integration (Oct 25, 2025)

**Objective:**
Implement Phase 4 multi-strategy orchestration with heat gating and risk management integration.

**Key Decisions:**
- Approved hybrid HANDOFF.md + OpenMemory approach for context management
- Decided to implement PortfolioManager with pre-trade gating pattern
- Deferred walk-forward validation until after Phase 4 complete

**Components Modified:**
- Created: core/portfolio_manager.py (385 lines)
- Modified: strategies/base_strategy.py (added heat integration)
- Created: tests/test_portfolio_manager.py (22 tests, all passing)

**Critical Insights:**
- Capital allocation requires heat coordination across strategies
- Pre-trade gating more effective than post-trade rejection
- PortfolioHeatManager integration point cleaner than expected

**Performance Results:**
Stored in OpenMemory with tag: session9_phase4

**Files Modified:**
See git commit 3a7b9e2 for complete changeset

**Test Status:**
- Previous: 66/66 PASSING
- Added: 22/22 PASSING
- Current: 88/88 PASSING (100%)

**Next Session:**
- Phase 4 integration testing (multi-strategy backtest)
- Gate 3 validation (performance metrics)
- Optimization if Sharpe/R:R targets not met

**OpenMemory Queries for Context:**
```bash
# Retrieve Session 9 facts
curl POST /memory/query -d '{"query": "session9 PortfolioManager findings", "filters": {"session": 9}}'

# Get Phase 4 architecture decisions
curl POST /memory/query -d '{"query": "Phase 4 architecture decisions gating pattern"}'
```
```

**HANDOFF.md Evolution:** Condensed format (30 lines vs 93 lines) reduces reading burden while preserving narrative. OpenMemory stores the detailed facts. This separation lets HANDOFF.md scale indefinitely - each session summary stays roughly the same length, while OpenMemory grows the queryable knowledge base.

---

## Part 6: MCP Integration with Claude Code

### Configure Claude Code to Use OpenMemory MCP

**IMPORTANT - Configuration File Location:**

On Windows, Claude Code does NOT read workspace `.claude/mcp.json`. Instead, edit the global configuration file:

**File Location:** `C:\Users\<USERNAME>\.claude.json`

Project-specific MCP servers go under: `projects['<PROJECT_PATH>']['mcpServers']`

**Edit `C:\Users\sheeh\.claude.json` and add to the ATLAS project section:**

```json
{
  "projects": {
    "C:\\Strat_Trading_Bot\\vectorbt-workspace": {
      "mcpServers": {
        "openmemory": {
          "type": "stdio",
          "command": "C:\\Dev\\openmemory\\node_modules\\.bin\\tsx.cmd",
          "args": ["C:\\Dev\\openmemory\\backend\\src\\mcp-stdio.ts"]
        }
      }
    }
  }
}
```

**Configuration Notes:**

1. **Absolute Paths Required**: Use absolute paths for both `command` and `args`. Relative paths fail on Windows due to working directory resolution issues.

2. **TypeScript Source Execution**: OpenMemory uses a Bun workspace with tsx for TypeScript execution. Call `tsx.cmd` directly from the root `node_modules/.bin/` directory.

3. **No npm/node wrapper**: Do NOT use `cmd /c npm` or `cmd /c node` wrappers. Windows batch file execution in MCP has path resolution issues.

4. **Entry Point**: `mcp-stdio.ts` is the correct entry point for Claude Code stdio integration (not the HTTP server).

**Restart Required:**

After modifying `.claude.json`, restart Claude Code completely (or restart VS Code if using Claude Code extension).

**Verification:**

After restart, OpenMemory MCP tools should appear:
- `mcp__openmemory__openmemory_query` - Semantic search
- `mcp__openmemory__openmemory_store` - Add memories
- `mcp__openmemory__openmemory_get` - Retrieve by ID
- `mcp__openmemory__openmemory_list` - List recent
- `mcp__openmemory__openmemory_reinforce` - Boost salience

### Using OpenMemory MCP Tools in Claude Code

Claude Code will now have access to OpenMemory tools:

```
- openmemory.query(query, filters, top_k)
- openmemory.store(content, type, metadata)
- openmemory.reinforce(memory_id)
- openmemory.list(filters, limit)
```

**Example Usage During Session:**
```
User: "What were the ORB validation results?"

Claude: [Uses openmemory.query("ORB validation results NVDA performance")]

Response:
- "NVDA ORB 5-min: 33 trades, 54.5% win rate, +2.51% return" (score: 0.94)
- "5-min range outperforms 30-min by 251x" (score: 0.91)
- "Volume 2.0x threshold superior to 1.5x" (score: 0.87)
```

---

## Part 7: Two-Week Experiment & Validation

### Week 1 Goals (Sessions 9-10)

**Setup Phase:**
- [x] Install OpenMemory
- [x] Configure MCP integration
- [x] Populate Session 8 facts
- [ ] Populate Session 9 facts (after completion)
- [ ] Populate Session 10 facts (after completion)

**Metrics to Track:**
- Time spent populating OpenMemory: _____ minutes
- Time spent maintaining server: _____ minutes
- Number of facts stored: _____
- Database size: _____ MB

**Don't Use Yet:** Just populate, don't query. Build the knowledge base first.

### Week 2 Goals (Sessions 11-12)

**Usage Phase:**
- [ ] Start Session 11 with OpenMemory queries instead of reading HANDOFF.md
- [ ] Track: How many queries used? _____
- [ ] Track: Did queries find relevant context missed by grep? Y/N
- [ ] Track: Time saved vs reading HANDOFF.md: _____ minutes

**Decision Criteria (Session 12 Retrospective):**

| Question | Answer | Score |
|----------|--------|-------|
| Did OpenMemory save time overall? | Y/N | +1/-1 |
| Did semantic search find context you would have missed? | Y/N | +2/0 |
| Was populating OpenMemory more burdensome than updating HANDOFF.md? | Y/N | -2/+1 |
| Did OpenMemory improve context quality for Claude Code? | Y/N | +2/0 |
| Would you continue using it after experiment? | Y/N | +3/0 |

**Scoring:**
- 5+: Keep both (hybrid approach validated)
- 2-4: Continue experiment 2 more weeks
- 0-1: Marginal benefit, evaluate cost
- <0: Overhead not worth it, revert to HANDOFF.md only

---

## Part 8: Troubleshooting & Common Issues

### Issue 1: OpenMemory Server Won't Start

**Symptom:** `Error: EADDRINUSE: address already in use`

**Solution:**
```bash
# Find process using port 8080
netstat -ano | findstr :8080

# Kill process by PID
taskkill /PID <process_id> /F

# Or change port in .env
PORT=8081
```

### Issue 2: MCP Server Not Appearing in Claude Code

**Symptom:** Claude Code doesn't show openmemory tools

**Solution:**
```bash
# Verify MCP server runs standalone
node C:\Dev\openmemory\backend\dist\mcp-server.js

# Check .claude/mcp.json syntax (must be valid JSON)
# Restart Claude Code completely
# Check Claude Code logs: Help > Developer Tools > Console
```

### Issue 3: Embedding API Costs Too High

**Symptom:** OpenAI bill exceeds budget

**Solution - Switch to Ollama (local, zero-cost):**

```bash
# Install Ollama for Windows
# Download from https://ollama.com/download

# Pull embedding model
ollama pull nomic-embed-text

# Update .env
EMBEDDING_PROVIDER=ollama
OLLAMA_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://localhost:11434

# Restart OpenMemory server
```

### Issue 4: Queries Return Irrelevant Results

**Symptom:** Semantic search returns wrong context

**Solutions:**
1. **Improve query specificity:**
   - Bad: "performance"
   - Good: "NVDA ORB performance metrics 6 months validation"

2. **Use metadata filters:**
   ```json
   {
     "query": "validation results",
     "filters": {
       "session": 8,
       "strategy": "ORB",
       "finding_type": "parameter_validation"
     }
   }
   ```

3. **Reinforce important memories:**
   ```bash
   # Increase salience of critical facts
   curl -X POST http://localhost:8080/memory/reinforce \
     -d '{"memory_id": "abc123"}'
   ```

---

## Part 9: Backup & Data Management

### Regular Backup Strategy

**Daily Backup (automated):**
```bash
# Add to cron/Task Scheduler
# Backup script: scripts/backup_openmemory.sh

#!/bin/bash
DATE=$(date +%Y%m%d)
SOURCE="C:/Dev/openmemory/data/atlas_memory.sqlite"
BACKUP_DIR="C:/Strat_Trading_Bot/vectorbt-workspace/backups/openmemory"

# Create backup directory if not exists
mkdir -p "$BACKUP_DIR"

# Copy database with date stamp
cp "$SOURCE" "$BACKUP_DIR/atlas_memory_$DATE.sqlite"

# Keep only last 7 days
find "$BACKUP_DIR" -name "atlas_memory_*.sqlite" -mtime +7 -delete

echo "OpenMemory backup completed: atlas_memory_$DATE.sqlite"
```

### Export to JSON (portable format)

```bash
# Export all memories to JSON
curl http://localhost:8080/memory/all > atlas_memory_export_$(date +%Y%m%d).json

# Import from JSON (if needed to restore)
# (implement custom script or use API to re-populate)
```

---

## Part 10: Cost Analysis

### Estimated Monthly Costs

**OpenAI Embeddings (Quality Option):**
- Embedding model: text-embedding-3-small ($0.02 per 1M tokens)
- Estimated usage: 100k tokens/month (aggressive ATLAS development)
- **Cost: $2/month**

**Ollama Local Embeddings (Zero-Cost Option):**
- Model: nomic-embed-text (runs on CPU)
- **Cost: $0/month**

**Hosting (if running 24/7):**
- Local machine: $0/month
- VPS (optional): $5-8/month (1GB RAM, 20GB storage)

**Total Monthly Cost:**
- OpenAI: $2/month
- Ollama: $0/month
- **Recommended: Start OpenAI, switch to Ollama if successful**

---

## Appendix A: File Structure

```
ATLAS_OpenMemory_Integration/
├── C:/Dev/openmemory/                    # OpenMemory repository
│   ├── backend/
│   │   ├── .env                          # Configuration
│   │   ├── dist/                         # Built files
│   │   │   └── mcp-server.js             # MCP integration
│   │   └── src/
│   └── data/
│       └── atlas_memory.sqlite           # Database
│
└── C:/Strat_Trading_Bot/vectorbt-workspace/
    ├── .claude/
    │   └── mcp.json                      # MCP configuration
    ├── scripts/
    │   ├── populate_openmemory.sh        # Population helper
    │   └── backup_openmemory.sh          # Backup automation
    ├── docs/
    │   ├── HANDOFF.md                    # Session narratives (condensed)
    │   ├── OpenMemory_Integration_Guide.md  # This document
    │   └── ...
    └── backups/
        └── openmemory/                   # Daily backups
```

---

## Appendix B: Sample Queries by Use Case

### Use Case 1: New Claude Session Startup

```bash
# Get overview of recent work
curl POST /memory/query -d '{
  "query": "last session objectives results key findings",
  "top_k": 10
}'

# Get current phase status
curl POST /memory/query -d '{
  "query": "Phase 4 PortfolioManager status progress",
  "filters": {"phase": "Phase4"}
}'

# Get development rules reminder
curl POST /memory/query -d '{
  "query": "mandatory workflows rules best practices",
  "filters": {"type": "procedural"},
  "top_k": 5
}'
```

### Use Case 2: Strategy Research

```bash
# What do we know about ORB strategy?
curl POST /memory/query -d '{
  "query": "ORB strategy performance limitations insights",
  "top_k": 15
}'

# What symbols have we tested?
curl POST /memory/query -d '{
  "query": "symbol testing results profitable unprofitable",
  "filters": {"strategy": "ORB"}
}'

# What parameters need optimization?
curl POST /memory/query -d '{
  "query": "parameters below target needs optimization",
  "filters": {"status": "below_target"}
}'
```

### Use Case 3: Debugging Reference

```bash
# How did we fix similar bugs before?
curl POST /memory/query -d '{
  "query": "bug fix solution zero trades filtering",
  "filters": {"type": "episodic"}
}'

# What VBT patterns have we verified?
curl POST /memory/query -d '{
  "query": "VBT verification workflow patterns validated",
  "filters": {"type": "procedural"}
}'
```

---

## Summary & Next Steps

**Immediate Actions:**
1. [ ] Install Node.js 20+ if not present
2. [ ] Clone OpenMemory repository
3. [ ] Configure `.env` with OpenAI or Ollama
4. [ ] Start server and verify health check
5. [ ] Add MCP configuration to `.claude/mcp.json`
6. [ ] Restart Claude Code
7. [ ] Populate Session 8 facts using provided script
8. [ ] Begin 2-week experiment

**Success Metrics:**
- Faster context retrieval in Sessions 11-12
- Discovery of relevant context missed by grep
- Sustainable 10-minute overhead per session
- Positive decision at Session 12 retrospective

**Fallback Plan:**
If experiment fails, simply stop using OpenMemory. HANDOFF.md remains unchanged as primary source of truth. No data loss, no wasted effort - the 2-week trial validates whether this scales ATLAS development.

---

**Implementation Philosophy:** This guide treats OpenMemory as an enhancement, not a dependency. ATLAS development continues normally even if OpenMemory fails. This low-risk approach lets you test whether semantic search genuinely improves workflow before committing to long-term maintenance.

**Document Status:** Ready for implementation
**Estimated Setup Time:** 2 hours (localhost) | 3-4 hours (VPS)
**Estimated Per-Session Overhead:** 10 minutes
**Expected Benefit Horizon:** Sessions 11+ (compound value)

---

## Part 11: Multi-Machine Deployment (VPS + Docker)

**NOTE:** This section is ONLY needed if you use MULTIPLE computers for ATLAS development (e.g., desktop at home + laptop at work). If you use ONE laptop everywhere, skip this - Parts 1-10 are sufficient.

---

### When You Need VPS Deployment

**Scenario:** You have two separate computers:
- Home desktop where you do most development
- Work laptop for testing during lunch breaks
- Both need access to the same OpenMemory database

**Problem:** Each computer has its own `localhost`:
- Home desktop: `localhost:8080` → Home computer only
- Work laptop: `localhost:8080` → Work computer only
- No way for work laptop to access home desktop's database

**Solution:** Deploy OpenMemory to cloud server (VPS) that both computers can reach.

---

### VPS Provider Recommendations

| Provider | Cost | RAM | Storage | Setup Difficulty |
|----------|------|-----|---------|------------------|
| DigitalOcean | $6/month | 1GB | 25GB | Easy |
| Linode | $5/month | 1GB | 25GB | Easy |
| Hetzner | $4/month | 2GB | 20GB | Medium |
| AWS Lightsail | $5/month | 1GB | 40GB | Medium |

**Recommendation:** DigitalOcean for simplicity and good documentation.

---

### Step-by-Step VPS Setup

#### 1. Create VPS Instance

```bash
# DigitalOcean Droplet Settings:
- Image: Ubuntu 22.04 LTS
- Plan: Basic ($6/month)
- RAM: 1GB
- Storage: 25GB
- Region: Closest to you (e.g., New York for East Coast)
- Authentication: SSH Key (generate if needed)

# After creation, note your VPS IP address: e.g., 203.0.113.50
```

#### 2. Initial Server Setup

```bash
# SSH into VPS
ssh root@203.0.113.50

# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
apt install docker-compose -y

# Verify installation
docker --version
docker-compose --version
```

#### 3. Install Firewall (Security Critical)

```bash
# Install UFW (Uncomplicated Firewall)
apt install ufw

# Allow SSH (CRITICAL - don't lock yourself out!)
ufw allow 22/tcp

# Allow OpenMemory API (only from your IPs)
# Replace with YOUR actual work/home IP addresses
ufw allow from 203.0.113.10 to any port 8080 comment 'Work IP'
ufw allow from 203.0.113.20 to any port 8080 comment 'Home IP'

# Enable firewall
ufw enable

# Verify rules
ufw status
```

**CRITICAL:** Get your IP addresses:
- Work IP: `curl ifconfig.me` (run from work computer)
- Home IP: `curl ifconfig.me` (run from home computer)

#### 4. Deploy OpenMemory with Docker

```bash
# Create directory
mkdir -p /opt/openmemory
cd /opt/openmemory

# Create docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  openmemory:
    image: node:20-alpine
    working_dir: /app
    command: sh -c "npm install && npm run build && npm start"
    ports:
      - "8080:8080"
    environment:
      - PORT=8080
      - EMBEDDING_PROVIDER=ollama
      - OLLAMA_BASE_URL=http://ollama:11434
      - OLLAMA_MODEL=nomic-embed-text
      - DB_PATH=/data/atlas_memory.sqlite
      - BEARER_TOKEN=${BEARER_TOKEN}
      - DECAY_LAMBDA=0.01
    volumes:
      - ./openmemory:/app
      - openmemory-data:/data
    restart: unless-stopped
    depends_on:
      - ollama

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama
    restart: unless-stopped

volumes:
  openmemory-data:
  ollama-data:
EOF

# Clone OpenMemory
git clone https://github.com/caviraoss/openmemory.git
mv openmemory/backend ./openmemory

# Generate secure bearer token
export BEARER_TOKEN=$(openssl rand -hex 32)
echo "BEARER_TOKEN=$BEARER_TOKEN" > .env
echo "Save this token for your local MCP configuration!"

# Start services
docker-compose up -d

# Pull Ollama embedding model
docker exec -it openmemory-ollama-1 ollama pull nomic-embed-text

# Check status
docker-compose ps
```

#### 5. Verify VPS Deployment

```bash
# From VPS itself
curl http://localhost:8080/health

# From your local machine (replace with your VPS IP)
curl http://203.0.113.50:8080/health
```

**Expected response:** `{"status":"ok","version":"1.2.0"}`

---

### Configure Local Machines to Use VPS

#### MCP Configuration (Home Computer + Work Laptop)

**Edit `.claude/mcp.json` on BOTH machines:**

```json
{
  "mcpServers": {
    "vectorbt-pro": {
      "command": "cmd",
      "args": [
        "/c",
        "uv",
        "--directory",
        "C:\\Strat_Trading_Bot\\vectorbt-workspace",
        "run",
        "mcp-server-vbt"
      ]
    },
    "playwright": {
      "command": "cmd",
      "args": [
        "/c",
        "node",
        "C:\\Users\\sheeh\\AppData\\Roaming\\npm\\node_modules\\@modelcontextprotocol\\server-playwright\\dist\\index.js"
      ]
    },
    "openmemory": {
      "command": "cmd",
      "args": [
        "/c",
        "node",
        "C:\\Dev\\openmemory-client\\mcp-client.js"
      ],
      "env": {
        "OPENMEMORY_URL": "http://203.0.113.50:8080",
        "OPENMEMORY_TOKEN": "your_bearer_token_from_vps_env_file"
      }
    }
  }
}
```

**Replace:**
- `203.0.113.50` with your actual VPS IP address
- `your_bearer_token_from_vps_env_file` with the token from VPS `.env`

#### Test From Both Machines

```bash
# Home computer
curl -X POST http://203.0.113.50:8080/memory/add \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_bearer_token" \
  -d '{"content": "Test from home computer"}'

# Work laptop
curl -X POST http://203.0.113.50:8080/memory/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test from home"}'

# Should return the memory added from home computer!
```

---

### VPS Maintenance

#### Automatic Updates

```bash
# Create update script
cat > /opt/openmemory/update.sh << 'EOF'
#!/bin/bash
cd /opt/openmemory
git -C openmemory pull
docker-compose down
docker-compose up -d --build
EOF

chmod +x /opt/openmemory/update.sh

# Run monthly
crontab -e
# Add: 0 2 1 * * /opt/openmemory/update.sh
```

#### Backup to Local Machine

```bash
# Run from local machine (not VPS)
# Backup script: scripts/backup_vps_openmemory.sh

#!/bin/bash
VPS_IP="203.0.113.50"
VPS_USER="root"
BACKUP_DIR="C:/Strat_Trading_Bot/vectorbt-workspace/backups/openmemory"
DATE=$(date +%Y%m%d)

# Download database from VPS
scp $VPS_USER@$VPS_IP:/opt/openmemory/data/atlas_memory.sqlite \
    $BACKUP_DIR/atlas_memory_vps_$DATE.sqlite

# Keep only last 7 days
find "$BACKUP_DIR" -name "atlas_memory_vps_*.sqlite" -mtime +7 -delete

echo "VPS backup completed: atlas_memory_vps_$DATE.sqlite"
```

#### Monitor VPS Health

```bash
# SSH into VPS
ssh root@203.0.113.50

# Check disk space
df -h

# Check memory usage
free -h

# Check Docker containers
docker-compose ps

# View OpenMemory logs
docker-compose logs openmemory --tail=100
```

---

### Security Best Practices for VPS

1. **Use SSH Keys (No Password Login):**
   ```bash
   # On VPS
   nano /etc/ssh/sshd_config
   # Set: PasswordAuthentication no
   systemctl restart sshd
   ```

2. **Regular Updates:**
   ```bash
   # Monthly security updates
   apt update && apt upgrade -y
   ```

3. **Monitor Failed Login Attempts:**
   ```bash
   # Install fail2ban
   apt install fail2ban -y
   systemctl enable fail2ban
   ```

4. **Backup Encryption:**
   ```bash
   # Encrypt backups before storing
   gpg --symmetric atlas_memory.sqlite
   ```

---

### Cost Comparison: Localhost vs VPS

| Aspect | Localhost | VPS |
|--------|-----------|-----|
| Monthly Cost | $0 | $5-10 |
| Setup Time | 1 hour | 3-4 hours |
| Maintenance | None | Monthly updates |
| Multi-Machine | No | Yes |
| Works Offline | Yes | No |
| Security Risk | Low | Medium (requires firewall) |

---

### When to Upgrade from Localhost to VPS

**Stay on Localhost If:**
- You use one laptop for all development
- You're okay with local-only access
- Cost is a concern
- You prefer simplicity

**Upgrade to VPS If:**
- You switch between desktop and laptop
- Team members need access
- You want access from mobile devices
- You're comfortable with server administration

---

**VPS Deployment Complete!** Your OpenMemory database is now accessible from any authorized machine.

---

**Document Status:** Ready for implementation
**Estimated Setup Time:** 2 hours (localhost) | 3-4 hours (VPS)
**Estimated Per-Session Overhead:** 10 minutes
**Expected Benefit Horizon:** Sessions 11+ (compound value)

---

**End of OpenMemory Integration Guide**
