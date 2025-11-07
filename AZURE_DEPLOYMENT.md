# Azure App Service Deployment Guide

## 🔴 Issues Fixed

### Problem 1: Wrong Startup Command

**Error**: `gunicorn -k uvicorn.workers.UvicornWorker app:asgi_app`
**Fix**: Changed to `python app.py api`

### Problem 2: Dependency Conflict

**Error**: `ImportError: cannot import name 'Sentinel' from 'typing_extensions'`
**Fix**: Added `typing-extensions>=4.8.0` to requirements.txt

---

## ✅ What Was Changed

### 1. **requirements.txt**

- Added `typing-extensions>=4.8.0` to fix Azure's outdated version
- Added `protobuf>=4.25.0` for compatibility

### 2. **GitHub Actions Workflow** (`.github/workflows/main_voiceassistent.yml`)

- Updated to set correct startup command: `python app.py api`
- Added app settings configuration
- Improved build process with zip artifact

### 3. **startup.txt**

- Created file with correct command for Azure App Service

---

## 🚀 Deployment Steps

### Automatic Deployment (via GitHub Actions)

1. **Commit and Push Changes**:

```bash
git add .
git commit -m "Fix Azure deployment: startup command and dependencies"
git push origin main
```

2. **GitHub Actions will automatically**:

   - Build the app
   - Install dependencies
   - Deploy to Azure
   - Set startup command

3. **Wait 5-10 minutes** for deployment to complete

4. **Check Azure Logs**:
   - Go to: https://portal.azure.com
   - Navigate to your App Service: `VoiceAssistent`
   - Click "Log stream" to see startup logs

---

## 🔧 Manual Configuration (If Needed)

If GitHub Actions doesn't set the startup command automatically, do this manually:

### Option A: Via Azure Portal

1. Go to **Azure Portal** → **App Services** → **VoiceAssistent**
2. Click **Configuration** (left sidebar)
3. Under **General settings**:
   - **Startup Command**: `python app.py api`
4. Click **Save**
5. Click **Restart**

### Option B: Via Azure CLI

```bash
az webapp config set \
  --resource-group <your-resource-group> \
  --name VoiceAssistent \
  --startup-file "python app.py api"
```

---

## 📋 Required Environment Variables

Make sure these are set in Azure Portal → Configuration → Application settings:

```
LIVEKIT_URL=wss://your-livekit-server.livekit.cloud
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
SIP_OUTBOUND_TRUNK_ID=ST_your_trunk_id
AZURE_SPEECH_KEY=your_azure_speech_key
AZURE_SPEECH_REGION=your_region
OPENAI_API_KEY=your_openai_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key (optional)
LANGFUSE_SECRET_KEY=your_langfuse_secret_key (optional)
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com (optional)
API_PORT=8000
API_HOST=0.0.0.0
```

To add these:

1. Azure Portal → App Service → Configuration
2. Click **+ New application setting**
3. Add each variable
4. Click **Save** and **Restart**

---

## ✅ Verify Deployment

### 1. Check Health Endpoint

```bash
curl https://voiceassistent-d3f6hefvc5bfc4gz.canadacentral-01.azurewebsites.net/health
```

Expected response:

```json
{
  "status": "healthy",
  "agent_worker": "running",
  "timestamp": 1234567890.123
}
```

### 2. Check Status Endpoint

```bash
curl https://voiceassistent-d3f6hefvc5bfc4gz.canadacentral-01.azurewebsites.net/status
```

### 3. Dispatch Test Call

```bash
curl -X POST https://voiceassistent-d3f6hefvc5bfc4gz.canadacentral-01.azurewebsites.net/dispatch/call \
  -H "Content-Type: application/json" \
  -d '{"phone_number": "+91XXXXXXXXXX"}'
```

---

## 🐛 Troubleshooting

### Issue: App Still Won't Start

**Check Azure Logs**:

1. Azure Portal → App Service → Log stream
2. Look for errors in startup

**Common Issues**:

1. **Missing Environment Variables**

   - Error: `ValueError: SIP_OUTBOUND_TRUNK_ID is not set`
   - Fix: Add environment variables in Configuration

2. **typing_extensions Error** (should be fixed now)

   - If still occurring, manually SSH into Azure and run:

   ```bash
   pip install --upgrade typing-extensions
   ```

3. **Port Binding Issues**
   - The app should bind to `0.0.0.0:8000`
   - Azure sets `$PORT` automatically (usually 8000)

### Issue: Can't Connect to LiveKit

**Check**:

- LIVEKIT_URL is correct
- API keys are valid
- SIP trunk is configured
- Network allows outbound connections

### Issue: "Application Error" on Website

This is EXPECTED! The app runs in "API mode" which starts:

1. FastAPI server (for API endpoints)
2. LiveKit agent worker (for handling calls)

The root URL (`/`) returns JSON, not a web page.

**Access API Documentation**:
https://voiceassistent-d3f6hefvc5bfc4gz.canadacentral-01.azurewebsites.net/docs

---

## 📊 Monitoring

### View Logs

```bash
az webapp log tail --name VoiceAssistent --resource-group <your-rg>
```

### Check Metrics

- Azure Portal → App Service → Metrics
- Monitor: CPU, Memory, HTTP requests

### Application Insights (Optional)

Enable in Azure Portal for better observability

---

## 🔄 Update Deployment

To deploy updates:

1. **Make changes locally**
2. **Commit and push**:

```bash
git add .
git commit -m "Your update message"
git push origin main
```

3. **GitHub Actions deploys automatically**
4. **Wait 5-10 minutes**

---

## 📞 API Endpoints

Once deployed, these endpoints are available:

| Endpoint          | Method | Description                    |
| ----------------- | ------ | ------------------------------ |
| `/`               | GET    | Service information            |
| `/health`         | GET    | Health check                   |
| `/status`         | GET    | Agent status                   |
| `/dispatch/call`  | POST   | Dispatch single call           |
| `/dispatch/batch` | POST   | Dispatch multiple calls        |
| `/docs`           | GET    | API documentation (Swagger UI) |

---

## ⚠️ Important Notes

1. **This is NOT a regular web app** - it's a LiveKit agent with API endpoints
2. **Startup takes 30-60 seconds** - agent needs to connect to LiveKit
3. **Long-running process** - the app runs continuously, not per-request
4. **Always-on recommended** - Azure App Service should be set to "Always On"

### Enable Always On:

1. Azure Portal → App Service → Configuration
2. General settings → Always On: **On**
3. Save and restart

---

## 💰 Cost Considerations

- **Basic plan** (B1) minimum recommended for always-on agents
- **Free tier** will sleep after 20 minutes (not suitable)
- Outbound calls cost based on LiveKit SIP pricing

---

## ✅ Checklist

- [ ] Updated `requirements.txt` with `typing-extensions>=4.8.0`
- [ ] Updated GitHub Actions workflow
- [ ] Set environment variables in Azure Portal
- [ ] Enabled "Always On" in Azure App Service
- [ ] Startup command set to `python app.py api`
- [ ] Tested `/health` endpoint
- [ ] Tested `/dispatch/call` endpoint
- [ ] Verified agent connects to LiveKit
- [ ] Tested actual phone call

---

**Last Updated**: November 7, 2024
**Status**: Ready for deployment
