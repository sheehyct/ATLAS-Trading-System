# ATLAS VPS Operations Guide

**VPS Provider:** Hetzner Cloud (CPX21)
**IP Address:** 178.156.223.251
**OS:** Ubuntu 24.04
**User:** atlas
**Cost:** $8.99/month

---

## Quick Reference

### Connect to VPS
```bash
ssh atlas@178.156.223.251
```

### Most Common Commands
```bash
# Check if daemon is running
sudo systemctl status atlas-daemon

# Restart daemon (after git pull)
sudo systemctl restart atlas-daemon

# View live logs (Ctrl+C to exit)
sudo journalctl -u atlas-daemon -f

# Pull latest code
cd ~/vectorbt-workspace && git pull
```

---

## Daemon Management

### Start/Stop/Restart
```bash
sudo systemctl start atlas-daemon      # Start the daemon
sudo systemctl stop atlas-daemon       # Stop the daemon
sudo systemctl restart atlas-daemon    # Restart (use after git pull)
sudo systemctl status atlas-daemon     # Check status
```

### Enable/Disable Auto-Start on Boot
```bash
sudo systemctl enable atlas-daemon     # Start on boot (already enabled)
sudo systemctl disable atlas-daemon    # Don't start on boot
```

---

## Viewing Logs

### Live Logs (Real-Time)
```bash
sudo journalctl -u atlas-daemon -f                    # Follow live logs
sudo journalctl -u atlas-daemon -f | grep -i error    # Only errors
sudo journalctl -u atlas-daemon -f | grep -i trigger  # Only triggers
```

### Historical Logs
```bash
# Last N lines
sudo journalctl -u atlas-daemon -n 50                 # Last 50 lines
sudo journalctl -u atlas-daemon -n 100                # Last 100 lines

# Time-based
sudo journalctl -u atlas-daemon --since today         # Today's logs
sudo journalctl -u atlas-daemon --since "1 hour ago"  # Last hour
sudo journalctl -u atlas-daemon --since "9:30" --until "16:00"  # Market hours

# Specific date/time (use UTC - add 5 hours to EST)
sudo journalctl -u atlas-daemon --since "2025-12-12 14:30" --until "2025-12-12 21:00"
```

### Save Logs to File
```bash
sudo journalctl -u atlas-daemon --since today > /tmp/today_logs.txt
cat /tmp/today_logs.txt    # View the file
```

---

## Deploying Code Updates

### Standard Deployment
```bash
ssh atlas@178.156.223.251
cd ~/vectorbt-workspace && git pull
sudo systemctl restart atlas-daemon
sudo journalctl -u atlas-daemon -f    # Verify it started correctly
```

### One-Liner (from local machine)
```bash
ssh atlas@178.156.223.251 "cd ~/vectorbt-workspace && git pull && sudo systemctl restart atlas-daemon"
```

---

## Signal API (Dashboard Backend)

The Signal API serves signals to the Railway dashboard.

```bash
sudo systemctl status atlas-signal-api    # Check status
sudo systemctl restart atlas-signal-api   # Restart
sudo journalctl -u atlas-signal-api -f    # View logs
```

**Endpoint:** http://178.156.223.251:5000/signals

---

## System Health

### Check Server Resources
```bash
htop                    # Interactive process viewer (q to quit)
df -h                   # Disk space
free -h                 # Memory usage
uptime                  # System uptime and load
```

### Check Running Python Processes
```bash
ps aux | grep python    # All python processes
pgrep -a signal_daemon  # Daemon process specifically
```

### Network/Connectivity
```bash
ping -c 3 google.com    # Test internet
curl -s http://localhost:5000/health    # Test signal API locally
```

---

## Timezone Reference

The VPS runs in **UTC**. To convert:

| EST (Market Time) | UTC (VPS Time) |
|-------------------|----------------|
| 9:30 AM | 14:30 (2:30 PM) |
| 4:00 PM | 21:00 (9:00 PM) |
| 11:30 AM | 16:30 (4:30 PM) |

**Tip:** When querying logs with `--since`, use UTC times.

---

## Environment Variables

The daemon reads these from the environment. To change:

```bash
# Edit the service file
sudo nano /etc/systemd/system/atlas-daemon.service

# After editing, reload and restart
sudo systemctl daemon-reload
sudo systemctl restart atlas-daemon
```

### Key Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `SIGNAL_MIN_HOLD_SECONDS` | 300 | Minimum hold time before exit checks |
| `SIGNAL_MONITOR_INTERVAL` | 60 | Seconds between position checks |
| `SIGNAL_EXIT_DTE` | 3 | Close at this DTE or below |
| `SIGNAL_MAX_LOSS_PCT` | 0.50 | Max loss before exit (50%) |
| `SIGNAL_EXECUTION_ENABLED` | true | Enable trade execution |

---

## Troubleshooting

### Daemon Won't Start
```bash
# Check for errors
sudo journalctl -u atlas-daemon -n 50

# Check if port is in use
sudo lsof -i :5000

# Check Python environment
cd ~/vectorbt-workspace
source .venv/bin/activate
python -c "import vectorbtpro; print('OK')"
```

### Daemon Crashes/Restarts
```bash
# Check restart count
sudo systemctl show atlas-daemon | grep Restart

# View crash logs
sudo journalctl -u atlas-daemon --since "10 minutes ago"
```

### Git Pull Fails
```bash
cd ~/vectorbt-workspace
git status              # Check for local changes
git stash               # Stash local changes if needed
git pull                # Try again
git stash pop           # Restore stashed changes
```

### SSH Connection Refused
1. Check your internet connection
2. Verify VPS is running in Hetzner dashboard
3. Check if your IP changed (may need to update firewall)

---

## File Locations

| What | Path |
|------|------|
| Project root | `~/vectorbt-workspace` |
| Signal store | `~/vectorbt-workspace/data/signals/` |
| Execution history | `~/vectorbt-workspace/data/executions/` |
| Daemon service | `/etc/systemd/system/atlas-daemon.service` |
| Signal API service | `/etc/systemd/system/atlas-signal-api.service` |
| Logs | Via `journalctl` (systemd managed) |

---

## Useful Aliases

Add these to `~/.bashrc` on the VPS for shortcuts:

```bash
# Edit with: nano ~/.bashrc
# Then run: source ~/.bashrc

alias dstatus='sudo systemctl status atlas-daemon'
alias drestart='sudo systemctl restart atlas-daemon'
alias dlogs='sudo journalctl -u atlas-daemon -f'
alias dlast='sudo journalctl -u atlas-daemon -n 100'
alias deploy='cd ~/vectorbt-workspace && git pull && sudo systemctl restart atlas-daemon'
```

After adding, run `source ~/.bashrc` to activate.

---

## Emergency Procedures

### Stop All Trading Immediately
```bash
ssh atlas@178.156.223.251
sudo systemctl stop atlas-daemon
```

### Check Open Positions (via Alpaca)
Go to: https://app.alpaca.markets/paper/dashboard/overview

### Manual Position Close
Use Alpaca dashboard or API directly - the daemon isn't required.

---

## Monthly Maintenance

1. **Check disk space:** `df -h` (should be < 50% used)
2. **Update system:** `sudo apt update && sudo apt upgrade -y`
3. **Reboot if needed:** `sudo reboot` (daemon auto-starts)
4. **Review logs for errors:** Check for recurring issues

---

*Last Updated: December 12, 2025 (Session 83K-77)*
