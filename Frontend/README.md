# FairSight — AI Bias Detection Platform
### Solution Challenge 2026 · Unbiased AI Decision Track

> Detect, explain, and fix hidden bias in any AI/ML dataset — powered by Gemini AI and deployed on Google Cloud Run.

---

## 🎯 What It Does

FairSight is a full-stack AI fairness auditing platform that:
1. **Accepts any CSV dataset** (hiring, loans, healthcare, etc.)
2. **Auto-detects sensitive attributes** (gender, race, age, etc.)
3. **Trains a classifier** and measures bias using Fairlearn metrics
4. **Computes fairness scores** using Demographic Parity + Equalized Odds
5. **Uses Gemini 1.5 Pro** to explain bias in plain English
6. **Suggests mitigation strategies** with code snippets

---

## 🏗️ Architecture

```
Frontend (HTML/JS)  →  FastAPI Backend  →  Fairlearn Engine
                              ↓
                    Gemini 1.5 Pro API
                              ↓
                    Google Cloud Run
```

---

## 🚀 Quick Start (Local)

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
export GEMINI_API_KEY=your_gemini_api_key_here
python main.py
# API runs at http://localhost:8080
```

### 2. Frontend

```bash
# Just open in browser — no build needed!
open frontend/index.html
# Or serve it:
cd frontend && python -m http.server 3000
```

---

## ☁️ Deploy to Google Cloud

### Prerequisites
- Google Cloud account with billing enabled
- `gcloud` CLI installed and authenticated
- Gemini API key from [Google AI Studio](https://aistudio.google.com)

### Deploy

```bash
export GCP_PROJECT_ID=your-project-id
export GEMINI_API_KEY=your-gemini-key
chmod +x deploy/deploy.sh
./deploy/deploy.sh
```

### Frontend Hosting (Firebase)

```bash
npm install -g firebase-tools
firebase login
firebase init hosting  # select frontend/ as public dir
firebase deploy
```

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/analysis/preview` | Preview CSV columns |
| POST | `/api/analysis/upload` | Run full bias analysis |
| POST | `/api/explain/bias-summary` | Gemini bias explanation |
| POST | `/api/explain/mitigation` | Gemini mitigation strategies |

---

## 🧪 Sample Datasets

The frontend includes built-in sample datasets:
- **Hiring Dataset** — 300 rows with intentional gender + race bias
- **Loan Approval** — 300 rows with income + race + gender bias

---

## 📝 Hackathon Submission Answers

**Challenge:** `[Unbiased AI Decision] Ensuring Fairness and Detecting Bias in Automated Decisions`

**Solution Overview (copy-paste):**
> FairSight is an AI-powered bias detection platform that helps organizations audit their machine learning systems for fairness. Users upload any tabular dataset, select sensitive attributes (gender, race, age), and the platform automatically trains a classifier, computes industry-standard fairness metrics (Demographic Parity, Equalized Odds), and generates a comprehensive fairness score. Gemini 1.5 Pro explains the bias findings in plain English and suggests concrete mitigation strategies with code. The platform is deployed on Google Cloud Run, making it accessible to any team needing to ensure their AI decisions are fair and unbiased.

**Google AI Model/Service:**
> Gemini 1.5 Pro (natural language bias explanations and mitigation strategy generation), Google Cloud Run (serverless deployment), Cloud Build (CI/CD)

**Google Cloud Deployed:** Yes ✅

---

## 🛠 Tech Stack

- **Frontend:** Vanilla HTML/CSS/JS (zero build step, instant deploy)
- **Backend:** Python FastAPI
- **Bias Detection:** Fairlearn + scikit-learn
- **AI:** Google Gemini 1.5 Pro
- **Cloud:** Google Cloud Run + Cloud Build
- **Containerization:** Docker

---

## 📁 Project Structure

```
fairsight/
├── backend/
│   ├── main.py              # FastAPI app entry
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── routes/
│   │   ├── analysis.py      # CSV upload + bias analysis
│   │   └── gemini_explain.py # Gemini AI explanations
│   └── services/
│       └── bias_engine.py   # Core fairness metrics
├── frontend/
│   └── index.html           # Full single-file React-less frontend
└── deploy/
    └── deploy.sh            # Cloud Run deploy script
```

---

## 🎥 Demo Video Script

1. Open FairSight homepage
2. Click "Hiring Dataset" sample → dataset loads
3. Show auto-detected sensitive columns (gender, race, age)
4. Click "Run Fairness Audit"
5. Show fairness scores dashboard
6. Highlight Gemini AI explanation card
7. Show bias bar charts per attribute
8. Click Mitigation Strategies tab
9. Show code snippets for fixing bias
10. Mention Google Cloud Run deployment

**Recommended:** Record with Loom or OBS, ~2 min max.

---

Built with ❤️ for Solution Challenge 2026
