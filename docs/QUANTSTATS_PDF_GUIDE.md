# QuantStats HTML → PDF Conversion Guide for Claude Code

## Quick Reference

When you have QuantStats HTML reports that need PDF conversion with risk level headers, use this workflow:

### Prerequisites (one-time setup)
```bash
pip install playwright
playwright install chromium
```

### The Script Location
Save `quantstats_to_pdf.py` to your project directory or a tools folder.

---

## Usage Examples

### Single file with risk header:
```bash
python quantstats_to_pdf.py strat_options_SPY_2pct_risk.html --risk "2%"
```

### Batch convert multiple files:
```bash
python quantstats_to_pdf.py report_2pct.html --risk "2%" -o STRAT_2pct_Risk.pdf
python quantstats_to_pdf.py report_5pct.html --risk "5%" -o STRAT_5pct_Risk.pdf
python quantstats_to_pdf.py report_10pct.html --risk "10%" -o STRAT_10pct_Risk.pdf
```

### Without risk header (basic conversion):
```bash
python quantstats_to_pdf.py report.html
```

---

## Claude Code Prompt Template

Copy/paste this when asking Claude Code to convert your reports:

```
Convert these QuantStats HTML reports to PDF using the quantstats_to_pdf.py script:

Files to convert:
- strat_options_SPY_2pct_risk.html → 2% risk
- strat_options_SPY_5pct_risk.html → 5% risk  
- strat_options_SPY_10pct_risk.html → 10% risk

Use the --risk flag to add risk level headers to each PDF.
Output files should be named: STRAT_Options_SPY_[X]pct_Risk.pdf
```

---

## What the Script Does

1. **Fixes broken QuantStats JS** - Removes the `onload="save()"` that causes console errors
2. **Adds risk level banner** - Dark header bar with "RISK LEVEL: X%" badge
3. **Proper SVG rendering** - Uses Playwright/Chromium to correctly render all charts
4. **Clean PDF output** - Letter size, proper margins, background colors preserved

---

## Script Flags

| Flag | Description | Example |
|------|-------------|---------|
| `--risk` or `-r` | Add risk level header | `--risk "2%"` |
| `--output` or `-o` | Custom output filename | `-o MyReport.pdf` |
| `--scale` | Adjust PDF scale (default 0.85) | `--scale 0.9` |

---

## Troubleshooting

**Charts not rendering?**
- Make sure Chromium is installed: `playwright install chromium`

**Banner cut off?**
- Script already fixed this - uses contained margins, not edge-to-edge

**Need different banner text?**
- Edit the `prepare_html()` function in the script to customize the "ATLAS STRAT Options Backtest" text
