# AWS Deployment Guide for Brother

## What You're Deploying
Single Python dashboard application (Plotly Dash framework).
Not separate frontend/backend - just one app.

## Requirements
- AWS Lightsail or EC2 instance (Ubuntu 20.04/22.04)
- $5/month Lightsail is more than enough
- 1GB RAM minimum, 10GB storage

## Quick Setup (5 minutes)

### Step 1: Create AWS Instance
1. Go to AWS Lightsail
2. Click "Create Instance"
3. Choose Ubuntu 22.04 LTS
4. Select $5/month plan (1GB RAM)
5. Name it: "atlas-dashboard"
6. Create instance

### Step 2: Run Setup Script
1. SSH into instance (use Lightsail browser SSH or terminal)
2. Copy-paste this ONE command:

```bash
curl -fsSL https://raw.githubusercontent.com/sheehyct/ATLAS-Algorithmic-Trading-System-V1/main/aws-setup.sh | bash
```

**That's it!** Script installs everything automatically.

### Step 3: Configure Firewall
In Lightsail Networking tab, add firewall rule:
- Application: Custom
- Protocol: TCP
- Port: 8050

### Step 4: Get IP Address
Note the instance's Public IP (e.g., 52.23.45.67)

### Step 5: Tell User
Send the IP address to account owner. They will:
1. SSH in and add their API credentials
2. Start the service

## After Setup - What Owner Does

Owner needs to SSH in once and run:

```bash
cd ~/ATLAS-Algorithmic-Trading-System-V1
nano .env
```

Add credentials:
```
ALPACA_API_KEY=their_key
ALPACA_SECRET_KEY=their_secret
```

Start service:
```bash
sudo systemctl start atlas-dashboard
sudo systemctl status atlas-dashboard
```

## Access Dashboard
http://PUBLIC_IP:8050

## Troubleshooting

**Check if running:**
```bash
sudo systemctl status atlas-dashboard
```

**View logs:**
```bash
sudo journalctl -u atlas-dashboard -f
```

**Restart service:**
```bash
sudo systemctl restart atlas-dashboard
```

**Check port 8050 is listening:**
```bash
sudo netstat -tulpn | grep 8050
```

## Optional: Add HTTPS (Later)

If owner wants secure access:
```bash
sudo apt install nginx certbot python3-certbot-nginx
# Configure nginx reverse proxy
# Get free SSL cert from Let's Encrypt
```

But for paper trading dashboard accessed from home, HTTP on port 8050 is fine.

## That's It!

Total setup time: 5 minutes
Literally just run the setup script and give owner the IP address.
