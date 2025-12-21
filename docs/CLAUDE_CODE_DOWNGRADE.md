# Claude Code Downgrade Instructions

## Will You Retain Data?

**YES** - Both will be retained:
- Chat history (stored in `~/.claude/`)
- MCP servers (stored in config files in `~/.claude/`)

The CLI binary is separate from your data.

---

## Step 1: Backup (Optional but Recommended)

```powershell
xcopy /E /I %USERPROFILE%\.claude %USERPROFILE%\.claude-backup
```

---

## Step 2: Uninstall Current Version

```powershell
npm uninstall -g @anthropic-ai/claude-code
```

---

## Step 3: Install Version 2.0.72

```powershell
npm install -g @anthropic-ai/claude-code@2.0.72
```

---

## Step 4: Verify

```powershell
claude --version
```

Should show `2.0.72`

---

## Session Notes

When you return, we identified a **significant bug** in how the scanner handles forming vs closed daily bars:

- The scanner uses today's FORMING bar as a setup bar
- Sets entry trigger at today's current intraday high
- Triggers when price touches that high again
- This caused the AAPL 3-2D-2U CALL to enter on Dec 18 before the pattern was complete

Fix needed: Exclude current calendar day/week/month from being a setup bar for daily/weekly/monthly timeframes.
