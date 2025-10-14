# 🚀 RiftRewind - Deployment Guide

## Current App
You have a beautiful Streamlit app with:
- ✅ Ekko video background
- ✅ Amazon Bedrock AI coaching
- ✅ Player analysis and insights
- ✅ Professional gaming UI

**Run locally:** `python3 -m streamlit run app.py`

---

## 🌐 Deploy to Streamlit Cloud (FREE & EASIEST)

### Step 1: Push to GitHub
```bash
git add -A
git commit -m "Clean up project, ready for deployment"
git push origin main
```

### Step 2: Deploy to Streamlit Cloud
1. Go to: **https://share.streamlit.io/**
2. Click **"New app"**
3. Connect your GitHub account (if not already)
4. Select:
   - Repository: `joshuaaachoo/solid-couscous`
   - Branch: `main`
   - Main file: `app.py`

### Step 3: Add Secrets
Click **"Advanced settings"** → **"Secrets"** → Add:

```toml
# AWS Credentials
AWS_ACCESS_KEY_ID = "your_access_key"
AWS_SECRET_ACCESS_KEY = "your_secret_key"
AWS_REGION = "us-east-1"

# Riot API
RIOT_API_KEY = "RGAPI-454b19d4-4f73-45f1-ac43-b15269530962"
```

### Step 4: Deploy!
Click **"Deploy"** - Takes 2-3 minutes

Your app will be live at: `https://your-app-name.streamlit.app`

---

## 🎯 For Hackathon Judges

Your app showcases:
1. **Amazon Bedrock** - AI-powered coaching insights
2. **Riot Games API** - Real-time player data
3. **Beautiful UI** - Gaming-themed with video background
4. **Streamlit** - Modern Python web framework

---

## 💰 Cost
- **Streamlit Cloud:** FREE (Community plan)
- **AWS Bedrock:** ~$0.01-0.05 per analysis (Free Tier eligible)
- **Total:** Essentially FREE for hackathon

---

## 🔧 Troubleshooting

### "Module not found" errors
Make sure `requirements.txt` is in repo root with all dependencies.

### Bedrock errors
Ensure AWS credentials in secrets have Bedrock permissions:
- `bedrock:InvokeModel`
- `bedrock:InvokeModelWithResponseStream`

### Video not showing
Streamlit Cloud sometimes has iframe restrictions. The app will still work without video.

---

## 📊 Next Steps (If You Want More AWS Services)

### Add DynamoDB Caching
- Cache API responses to reduce Riot API calls
- Store player analysis results

### Add Amazon Comprehend
- Toxicity detection in match chat
- Sentiment analysis of player behavior

### Add CloudWatch Dashboard
- Monitor API usage
- Track app performance
- Display metrics to judges

---

**Ready to deploy?** Follow Step 1-4 above! 🚀
