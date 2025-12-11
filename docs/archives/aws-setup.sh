#!/bin/bash
# ATLAS Dashboard - AWS Deployment Setup
# Copy-paste this entire script and run on Ubuntu Lightsail/EC2 instance

set -e  # Exit on error

echo "========================================="
echo "ATLAS Dashboard - AWS Setup"
echo "========================================="

# Update system
echo "[1/6] Updating system packages..."
sudo apt update
sudo apt install -y python3.10 python3-pip git curl

# Install uv (Python package manager)
echo "[2/6] Installing uv package manager..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Clone repository
echo "[3/6] Cloning ATLAS repository..."
cd $HOME
if [ -d "ATLAS-Algorithmic-Trading-System-V1" ]; then
    echo "Repository already exists, pulling latest..."
    cd ATLAS-Algorithmic-Trading-System-V1
    git pull
else
    git clone https://github.com/sheehyct/ATLAS-Algorithmic-Trading-System-V1.git
    cd ATLAS-Algorithmic-Trading-System-V1
fi

# Install dependencies using uv
echo "[4/6] Installing Python dependencies (this may take 2-3 minutes)..."
uv sync

# Create systemd service
echo "[5/6] Creating system service..."
sudo tee /etc/systemd/system/atlas-dashboard.service > /dev/null <<EOF
[Unit]
Description=ATLAS Trading Dashboard
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/ATLAS-Algorithmic-Trading-System-V1
EnvironmentFile=$HOME/ATLAS-Algorithmic-Trading-System-V1/.env
ExecStart=$HOME/.cargo/bin/uv run python -m dashboard.app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable atlas-dashboard

echo "[6/6] Setup complete!"
echo ""
echo "========================================="
echo "NEXT STEPS (for the account owner):"
echo "========================================="
echo "1. SSH into this server"
echo "2. Create .env file with credentials:"
echo "   cd ~/ATLAS-Algorithmic-Trading-System-V1"
echo "   nano .env"
echo ""
echo "3. Add these lines to .env:"
echo "   ALPACA_API_KEY=your_key_here"
echo "   ALPACA_SECRET_KEY=your_secret_here"
echo ""
echo "4. Start the service:"
echo "   sudo systemctl start atlas-dashboard"
echo "   sudo systemctl status atlas-dashboard"
echo ""
echo "5. Access dashboard at: http://$(curl -s ifconfig.me):8050"
echo "========================================="
