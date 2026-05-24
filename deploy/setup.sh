#!/usr/bin/env bash
# setup.sh — run once on the Bluehost VPS after SSH-ing in as root.
# Replace YOUR_REPO_URL with your actual GitHub repo URL before running.
set -e

echo "=== Installing system packages ==="
apt update && apt install -y python3-pip python3-venv git certbot python3-certbot-nginx

echo "=== Cloning repo ==="
cd /opt
git clone YOUR_REPO_URL baseball
cd baseball

echo "=== Creating virtualenv and installing dependencies ==="
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Creating data directory ==="
mkdir -p data
# Upload your data/ files via SFTP after this script finishes:
#   scp -r data/ root@YOUR_VPS_IP:/opt/baseball/

echo "=== Creating .env from template ==="
cp .env.example .env
echo ""
echo "ACTION REQUIRED: edit /opt/baseball/.env and fill in ODDS_API_KEY"
echo "  nano /opt/baseball/.env"

echo "=== Installing systemd services ==="
cp deploy/baseball-api.service /etc/systemd/system/
cp deploy/baseball-mcp.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable baseball-api baseball-mcp

echo "=== Installing nginx config ==="
cp deploy/nginx-baseball.conf /etc/nginx/sites-available/baseball
ln -sf /etc/nginx/sites-available/baseball /etc/nginx/sites-enabled/baseball
nginx -t && systemctl reload nginx

echo ""
echo "=== NEXT STEPS ==="
echo "1. Edit /opt/baseball/.env with your ODDS_API_KEY and real domain name"
echo "2. Upload data/ files via SFTP (parquets, model.pkl, run_meta.json)"
echo "3. Add DNS A records for api.trevormonroe.com and mcp.trevormonroe.com -> $(curl -s ifconfig.me)"
echo "4. Once DNS propagates, run: certbot --nginx -d api.trevormonroe.com -d mcp.trevormonroe.com"
echo "5. Start services: systemctl start baseball-api baseball-mcp"
echo "6. Add cron job: crontab -e  (paste contents of deploy/crontab.txt)"
