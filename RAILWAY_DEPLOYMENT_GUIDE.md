# Railway Deployment Guide (5 Minutes)

## Step 1: Push Railway Config to GitHub

Files already created:
- `Procfile` ✓
- `railway.toml` ✓

Commit and push:
```bash
git add Procfile railway.toml
git commit -m "add railway deployment configuration"
git push origin main
```

## Step 2: Deploy on Railway.app

### Via Web UI (Easiest):

1. Go to https://railway.app
2. Click "New Project"
3. Choose "Deploy from GitHub repo"
4. Select: `ATLAS-Algorithmic-Trading-System-V1`
5. Railway auto-detects Python and starts building

### Add Environment Variables:

1. In Railway project, click "Variables"
2. Add these:
   - `ALPACA_API_KEY` = your_paper_key
   - `ALPACA_SECRET_KEY` = your_paper_secret

3. Click "Deploy" (Railway auto-restarts with new env vars)

### Get Your URL:

1. Click "Settings" tab
2. Click "Generate Domain"
3. Railway gives you: `your-app-abc123.up.railway.app`

**Access dashboard:** https://your-app-abc123.up.railway.app

(Note: Railway uses HTTPS automatically, port 8050 not in URL)

## Step 3: Custom Domain (Optional)

If you have a domain:
1. Railway Settings → Domains
2. Add custom domain (e.g., `dashboard.yourdomain.com`)
3. Point CNAME to Railway URL
4. Done! Free SSL included

## Troubleshooting

### Build Fails?

**Check build logs in Railway dashboard**

Common issues:
- Missing dependencies (Railway auto-installs from pyproject.toml)
- Python version (Railway detects from pyproject.toml)

### App Crashes on Start?

**Check deployment logs**

Usually missing environment variables:
- Make sure `ALPACA_API_KEY` set
- Make sure `ALPACA_SECRET_KEY` set

### Port Issues?

Railway automatically assigns `PORT` environment variable.
Dashboard needs to listen on `0.0.0.0:8050` (already configured).

If issues, check `dashboard/config.py`:
```python
DASHBOARD_CONFIG = {
    'host': '0.0.0.0',  # Must be 0.0.0.0 for Railway
    'port': 8050,
    'debug': False  # Should be False for production
}
```

## Updating Dashboard

Whenever you push to GitHub:
```bash
git push origin main
```

Railway automatically:
1. Detects push
2. Rebuilds app
3. Deploys new version
4. Zero downtime!

## Cost Estimate

- First month: **FREE** ($5 credit)
- Ongoing: **~$5-7/month**
  - Base: $5/month
  - Data transfer: ~$0-2/month (dashboard is light)

## Monitoring

Railway dashboard shows:
- CPU usage
- Memory usage
- Request logs
- Error logs
- Deployment history

## Comparison: Railway vs AWS

| Feature | Railway | AWS Lightsail |
|---------|---------|---------------|
| Setup Time | 5 minutes | 30 minutes |
| HTTPS | Automatic | Manual (nginx + certbot) |
| Auto-deploy | Git push | Manual SSH + pull |
| Logs | Web dashboard | SSH + journalctl |
| Cost | $5-7/month | $5/month |
| Mobile-friendly | ✓ CDN | ✓ but slower |
| Custom domain | Free SSL | Need Let's Encrypt |

**Recommendation:** Use Railway as primary, AWS as backup.

## Railway Mobile Access

Your dashboard is already mobile-responsive (Bootstrap).
Just bookmark the Railway URL on your phone.

Works great on:
- iPhone Safari
- Android Chrome
- iPad
- Any mobile browser

Auto-refresh works on mobile too (updates every 30 seconds).

## That's It!

Railway deployment:
1. Push to GitHub
2. Connect Railway to repo
3. Add env vars
4. Access your URL

Total time: 5 minutes
