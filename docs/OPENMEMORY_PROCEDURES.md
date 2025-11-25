# OpenMemory Procedures - ATLAS Trading System

Version: 1.0
Created: October 26, 2025
Purpose: Quick reference for OpenMemory operations

---

## System Overview

**Hybrid Context Management:**
- HANDOFF.md: Session narratives, decisions, next actions (human-curated)
- OpenMemory: Queryable facts database, semantic search (machine-indexed)

**Deployment Type:** Localhost (single machine)
- Server: http://localhost:8080
- Database: C:/Dev/openmemory/data/atlas_memory.sqlite
- MCP Integration: Enabled via Claude Code
- Embedding Provider: OpenAI (text-embedding-3-small)

---

## Daily Operations

### Starting OpenMemory Server

**Option 1: Manual Start (recommended for development)**
```bash
cd /c/Dev/openmemory/backend
npm run dev
```

Expected output:
```
OpenMemory server starting on port 8080
[MCP] Server started and transport connected
Server running on http://localhost:8080
```

**Option 2: Background Start**
```bash
cd /c/Dev/openmemory/backend
nohup npm run dev > openmemory.log 2>&1 &
```

### Checking Server Status

```bash
# Health check
curl http://localhost:8080/health

# Expected: {"ok":true,"version":"2.0-hsg","embedding":{...}}

# Check if process is running
ps aux | grep "tsx src/server/index.ts"
```

### Stopping OpenMemory Server

```bash
# Find the process ID
ps aux | grep "tsx src/server/index.ts"

# Kill the process (replace PID)
kill <PID>

# Or force kill if needed
kill -9 <PID>
```

---

## MCP Integration (Claude Code)

### Configuration

**IMPORTANT - Location:** `C:\Users\sheeh\.claude.json` (NOT `.claude/mcp.json`)

Claude Code reads from the global configuration file, not workspace `.claude/mcp.json` on Windows.

**Working Configuration (Windows):**

Add to `C:\Users\sheeh\.claude.json` under ATLAS project:

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

**Critical Requirements:**
- Use absolute paths for both command and args
- Call tsx.cmd from root node_modules (Bun workspace structure)
- Entry point: mcp-stdio.ts (stdio transport, not HTTP server)
- Restart Claude Code after configuration changes

**OpenMemory MCP Server:**
- Automatically starts when Claude Code launches
- Connects via stdio transport
- No manual server management needed when using Claude Code

**MCP Tools Available:**
- openmemory.query: Semantic search for facts
- openmemory.store: Add new memories
- openmemory.get: Retrieve specific memory by ID
- openmemory.list: List recent memories
- openmemory.reinforce: Boost memory salience

### Using MCP Tools in Claude Code

**Query memories:**
```
User: "What were the ORB validation results?"
Claude: [Automatically uses openmemory.query() to search]
```

**Store new facts (manual):**
```
User: "Store this finding: Volume threshold 3.0x tested on Session 9"
Claude: [Uses openmemory.store() to add memory]
```

### Restart Required

After modifying `.claude/mcp.json`, restart Claude Code to reload MCP servers.

---

## Session Workflow

### Session Start Routine

**1. Verify OpenMemory is running:**
```bash
curl -s http://localhost:8080/health | grep -q "ok" && echo "Running" || echo "Not running"
```

**2. If not running, start server:**
```bash
cd /c/Dev/openmemory/backend && npm run dev &
```

**3. Query for context (via Claude Code or curl):**
```bash
# Example: Get last session context
curl -s -X POST http://localhost:8080/memory/query \
  -H "Content-Type: application/json" \
  -d '{"query": "last session findings results", "k": 10}'
```

### Session End Routine

**1. Populate session facts:**
```bash
# Run population script (modify for current session first)
./scripts/populate_openmemory.sh
```

**2. Backup database (optional but recommended):**
```bash
./scripts/backup_openmemory.sh
```

**3. Verify memories stored:**
```bash
curl -s -X POST http://localhost:8080/memory/query \
  -H "Content-Type: application/json" \
  -d '{"query": "session <N> findings", "k": 5}'
```

---

## Maintenance

### Database Backup

**Manual backup:**
```bash
./scripts/backup_openmemory.sh
```

**Backup location:** `backups/openmemory/atlas_memory_YYYYMMDD.sqlite`

**Retention:** Indefinite (databases are small, ~1 MB per backup)
- Edit script to enable 90-day cleanup if needed

### Database Size Monitoring

```bash
# Check database size
du -h C:/Dev/openmemory/data/atlas_memory.sqlite

# Count memories
curl -s http://localhost:8080/sectors | python -m json.tool
```

### Cost Monitoring (OpenAI Embeddings)

**Current usage:**
- Model: text-embedding-3-small ($0.02 per 1M tokens)
- Estimated: ~$2/month for active development

**To switch to Ollama (zero-cost):**
1. Install Ollama for Windows
2. Pull model: `ollama pull nomic-embed-text`
3. Update `C:/Dev/openmemory/.env`:
   ```
   OM_EMBEDDINGS=ollama
   OLLAMA_URL=http://localhost:11434
   ```
4. Restart OpenMemory server

---

## Troubleshooting

### Server Won't Start

**Error: EADDRINUSE (port 8080 in use)**
```bash
# Find what's using port 8080
netstat -ano | findstr :8080

# Kill the process
taskkill /PID <process_id> /F

# Or change port in .env
# OM_PORT=8081
```

**Error: Cannot find module**
```bash
cd /c/Dev/openmemory/backend
npm install --legacy-peer-deps
```

### MCP Server Not Appearing in Claude Code

**CRITICAL - Windows Configuration Issue:**

If MCP server is not appearing, check the configuration file location first:

1. **File Location**: Edit `C:\Users\sheeh\.claude.json` NOT `.claude/mcp.json`
2. **Project Section**: Configuration must be under `projects['C:\\Strat_Trading_Bot\\vectorbt-workspace']['mcpServers']`
3. **Absolute Paths**: Use absolute paths for both command and args
4. **Correct Entry Point**: Use `mcp-stdio.ts` not `mcp-server.js` or `index.ts`

**Common Windows Issues:**

**Issue: ENOENT error when using npm wrapper**
```
Error: spawn npm ENOENT
```
Solution: Call tsx.cmd directly, not through npm or cmd wrapper

**Issue: Path resolution failure**
```
Error: Cannot find module 'C:\Strat_Trading_Bot\vectorbt-workspace\backend\src\mcp-stdio.ts'
```
Solution: Use absolute path for args, not relative path

**Issue: Working directory not respected**
```
Error: Cannot find package.json
```
Solution: Don't rely on cwd parameter, use absolute paths instead

**Verification Steps:**

1. Check `.claude.json` syntax (must be valid JSON)
2. Verify paths are correct (use absolute Windows paths with double backslashes)
3. Restart Claude Code completely (or restart computer for stubborn cases)
4. Check Claude Code logs: Help > Developer Tools > Console
5. Look for `[MCP-stdio] OpenMemory MCP server started via stdio` in logs

**Test MCP server standalone:**
```bash
cd /c/Dev/openmemory/backend
npm run mcp
# Should output: [MCP-stdio] OpenMemory MCP server started via stdio
# Press Ctrl+C to exit
```

**Working Configuration Example:**
```json
"openmemory": {
  "type": "stdio",
  "command": "C:\\Dev\\openmemory\\node_modules\\.bin\\tsx.cmd",
  "args": ["C:\\Dev\\openmemory\\backend\\src\\mcp-stdio.ts"]
}
```

### Query Returns No Results

**Check if memories exist:**
```bash
curl -s http://localhost:8080/sectors
```

**Repopulate if needed:**
```bash
./scripts/populate_openmemory.sh
```

**Improve query specificity:**
- Bad: "results"
- Good: "NVDA ORB validation results 5-min range Session 8"

### Database Corruption

**Restore from backup:**
```bash
# Find latest backup
ls -lt backups/openmemory/atlas_memory_*.sqlite | head -1

# Stop server
kill <PID>

# Restore backup
cp backups/openmemory/atlas_memory_YYYYMMDD.sqlite \
   C:/Dev/openmemory/data/atlas_memory.sqlite

# Restart server
cd /c/Dev/openmemory/backend && npm run dev
```

---

## Query Patterns

### Common Queries for ATLAS Development

**Strategy performance:**
```json
{
  "query": "strategy performance metrics results",
  "k": 10
}
```

**Symbol testing history:**
```json
{
  "query": "symbol testing NVDA TSLA AAPL profitable unprofitable",
  "k": 10
}
```

**Parameter validations:**
```json
{
  "query": "parameter validation optimal values tested",
  "k": 10
}
```

**Development rules:**
```json
{
  "query": "development workflow rules best practices mandatory",
  "k": 10
}
```

**Session-specific context:**
```json
{
  "query": "Session 8 ORB validation empirical testing",
  "k": 15
}
```

---

## File Locations

**OpenMemory Installation:**
- Repository: `C:/Dev/openmemory/`
- Backend: `C:/Dev/openmemory/backend/`
- Database: `C:/Dev/openmemory/data/atlas_memory.sqlite`
- Configuration: `C:/Dev/openmemory/.env`

**ATLAS Workspace:**
- MCP Config: `.claude/mcp.json`
- Population Script: `scripts/populate_openmemory.sh`
- Backup Script: `scripts/backup_openmemory.sh`
- Backups: `backups/openmemory/`
- Documentation: `docs/OpenMemory_Integration_Guide.md` (comprehensive)
- Procedures: `docs/OPENMEMORY_PROCEDURES.md` (this file - quick reference)

---

## Quick Command Reference

```bash
# Start server
cd /c/Dev/openmemory/backend && npm run dev

# Stop server
kill $(ps aux | grep "tsx src/server/index.ts" | grep -v grep | awk '{print $2}')

# Health check
curl http://localhost:8080/health

# Query memories
curl -X POST http://localhost:8080/memory/query \
  -H "Content-Type: application/json" \
  -d '{"query": "your search query", "k": 5}'

# Add memory
curl -X POST http://localhost:8080/memory/add \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer atlas_openmemory_dev_token_2025" \
  -d '{"content": "memory content", "metadata": {}}'

# Populate session facts
./scripts/populate_openmemory.sh

# Backup database
./scripts/backup_openmemory.sh

# View sectors stats
curl http://localhost:8080/sectors | python -m json.tool
```

---

## Status: Production Ready

OpenMemory integration is live and operational. Begin 2-week trial to evaluate effectiveness.

**Track metrics:**
- Time spent populating: _____ minutes per session
- Time saved querying vs reading HANDOFF.md: _____ minutes per session
- Facts stored: _____ total
- Database size: _____ MB
- Successful context retrievals: _____ / _____ queries

**Decision point:** Session 12 retrospective (2 weeks from now)

---

End of OpenMemory Procedures
