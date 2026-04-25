# FairSight

> Detect bias. Understand impact. Build fairer AI.

![FairSight Demo](./screenshot-placeholder.png)

FairSight is an AI-powered bias detection and fairness auditing platform built for rapid prototyping and hackathon demos. It combines FastAPI, scikit-learn, Fairlearn-style metrics, and Google Gemini to convert raw CSV datasets into actionable fairness insights.

---

## Architecture

```
CSV Upload
    |
    v
FastAPI Backend (Backend/main.py)
    |
    v
bias_engine.py (Backend/services/bias_engine.py)
    |
    +--> Demographic Parity
    |    Equalized Odds
    |    Fairness Score
    |
    +--> Gemini Explain / Mitigation
    v
Frontend (Frontend/index.html)
```

---

## Tech Stack

- Backend: FastAPI with CORS middleware
- ML: `scikit-learn` `LogisticRegression`
- Fairness metrics: `fairlearn` concepts
- Vector storage: `ChromaDB`
- AI explanations: Google Gemini 1.5 Flash
- Frontend: Vanilla JavaScript + Tailwind-style CSS
- Data processing: `pandas`, `numpy`

---

## Core Features

1. CSV dataset upload with automatic sensitive column detection
   - Detects `gender`, `race`, `age`, `religion`, `nationality`, `disability`, and related columns
2. Demographic Parity computation — outcome rate per group
3. Equalized Odds computation — TPR and FPR per group
4. Composite Fairness Score (0-100)
   - `score = (dp_score * 0.4 + tpr_score * 0.3 + fpr_score * 0.3) * 100`
5. Severity classification
   - `High` (<50), `Medium` (<75), `Low` (75+)
6. Gemini AI generates executive summary, real-world impact, urgency level
7. Gemini AI suggests 4 mitigation strategies with code hints
8. Fallback responses when the Gemini API key is not set
9. Numeric columns like `age` are auto-binned into readable ranges

---

## Live Demo Results

- Achieved **79% model accuracy** on the hiring dataset
- Detected **25% disparity gap** between `Male` / `Female` / `Non-binary` groups
- Overall fairness score of **27/100** flagged as **Critical Priority**
- Gemini correctly identified **gender** as the most critical bias attribute
- Frontend shows:
  - Demographic Parity bars
  - Equalized Odds TPR/FPR charts
  - Mitigation Strategies cards

---

## API Endpoints

### `GET /health`

Check service health.

**Response**

```json
{
  "status": "ok",
  "service": "FairSight API"
}
```

---

### `POST /api/analysis/preview`

Preview a CSV before analysis. Returns columns, dataset shape, detected sensitive columns, and sample rows.

**Request**

- `file`: CSV upload

**Response**

```json
{
  "columns": ["gender", "age", "race", "experience_years", "education", "hired"],
  "shape": {"rows": 300, "cols": 6},
  "detected_sensitive": ["gender", "race", "age"],
  "sample": [ ... ]
}
```

---

### `POST /api/analysis/upload`

Runs the full bias audit pipeline.

**Request**

- `file`: CSV upload
- `label_col`: target column name
- `sensitive_cols`: comma-separated sensitive attributes

**Response**

```json
{
  "model_accuracy": 0.79,
  "total_samples": 300,
  "test_samples": 90,
  "label_col": "hired",
  "sensitive_cols": ["gender", "race", "age"],
  "bias_analysis": {
    "gender": {
      "fairness_score": 27,
      "severity": "High",
      "demographic_parity": {
        "group_rates": {"Male": 0.62, "Female": 0.37, "Non-binary": 0.40},
        "disparity": 0.25,
        "max_group": "Male",
        "min_group": "Female"
      },
      "equalized_odds": {
        "tpr_per_group": {"Male": 0.72, "Female": 0.48, "Non-binary": 0.44},
        "fpr_per_group": {"Male": 0.12, "Female": 0.20, "Non-binary": 0.18},
        "tpr_disparity": 0.28,
        "fpr_disparity": 0.08
      }
    }
  },
  "visualizations": {
    "overall_fairness": "<base64_png>",
    "dp_gender": "<base64_png>",
    "eo_gender": "<base64_png>"
  }
}
```

---

### `POST /api/explain/bias-summary`

Returns a Gemini-generated executive summary and urgency ranking.

**Request**

```json
{
  "bias_results": { ... },
  "dataset_name": "hiring.csv"
}
```

**Response**

```json
{
  "source": "gemini",
  "explanation": {
    "executive_summary": "...",
    "attribute_explanations": {
      "gender": "...",
      "race": "..."
    },
    "real_world_impact": "...",
    "urgency_level": "Critical",
    "key_finding": "..."
  }
}
```

---

### `POST /api/explain/mitigation`

Returns 4 mitigation strategy suggestions with code hints.

**Request**

```json
{
  "sensitive_col": "gender",
  "fairness_score": 27,
  "demographic_parity": { ... },
  "equalized_odds": { ... }
}
```

**Response**

```json
{
  "source": "gemini",
  "strategies": [
    {
      "name": "Reweighing",
      "description": "...",
      "difficulty": "Easy",
      "impact": "High",
      "code_hint": "from fairlearn.preprocessing import Reweighing; ...",
      "tradeoff": "No accuracy loss expected; minimal runtime cost."
    }
  ]
}
```

---

## Fairness Metrics Explained

### Demographic Parity
Demographic Parity measures whether each demographic group receives the positive outcome at the same rate. In FairSight, it computes the positive outcome percentage per sensitive group and reports the gap between the highest and lowest rates.

### Equalized Odds
Equalized Odds checks whether the model's True Positive Rate (TPR) and False Positive Rate (FPR) are aligned across groups. It is a stronger fairness condition that measures performance consistency conditioned on the true label.

### Composite Fairness Score
FairSight combines normalized disparity measures into a single score from 0 to 100:

```python
score = (dp_score * 0.4 + tpr_score * 0.3 + fpr_score * 0.3) * 100
```

- `dp_score` rewards low demographic parity gap
- `tpr_score` rewards similar TPR across groups
- `fpr_score` rewards similar FPR across groups

### Severity Classification
- `High`: fairness score below 50
- `Medium`: fairness score below 75
- `Low`: fairness score 75 or above

---

## Mitigation Strategies

### Reweighing
Assigns sample weights to balance representation across sensitive groups. This is a pre-processing approach that keeps your model architecture unchanged.

### ThresholdOptimizer
A post-processing method that adjusts decision thresholds per group to equalize TPR and FPR. Useful when you want to enforce group fairness without retraining the model.

### ExponentiatedGradient
An in-processing technique that trains the model while satisfying fairness constraints. It often provides a better accuracy-fairness tradeoff at the cost of more compute.

### Feature Removal
Dropping or masking a sensitive feature like `gender` can reduce direct discrimination, but proxy variables may still carry bias.

---

## Real-world Use Cases

| Use Case | Why it matters |
|---|---|
| Hiring | Prevent biased hiring decisions across gender, race, or age groups |
| Loan approval | Reduce disparate impact in credit decisions |
| Healthcare | Ensure fair diagnosis and treatment recommendations |
| Education | Audit admissions and scholarship decisions for equity |

---

## How to Run

### Backend

```bash
cd Backend
source venv/bin/activate
export GEMINI_API_KEY=your_key_here
python main.py
```

The backend runs on `http://localhost:8080`.

### Frontend

```bash
cd Frontend
python3 -m http.server 3000
```

The frontend runs on `http://localhost:3000` and talks to the FastAPI backend.

---

## Generate a Test CSV Quickly

Create a hiring dataset with a Python one-liner:

```bash
python - <<'PY'
import csv
rows = [
    ("Male", 28, "White", 5, "Bachelor", 1),
    ("Female", 32, "Black", 7, "Master", 0),
    ("Non-binary", 24, "Asian", 3, "Bachelor", 0),
]
with open('hiring.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['gender', 'age', 'race', 'experience_years', 'education', 'hired'])
    writer.writerows(rows * 20)
PY
```

This generates a demo `hiring.csv` with at least 60 rows.

---

## Known Limitations

- The current pipeline uses a single `LogisticRegression` model and may not capture all real-world complexity.
- Automatic sensitive-column detection is rule-based and may miss custom or non-standard column names.
- Gemini explanations depend on the configured API key; fallback responses are used when the key is missing.
- Numeric features are binned into only a few categories, which may simplify complex distributions.
- The demo frontend is a static HTML page and not a full production React app.

---

## Future Improvements

- Add support for additional model families and hyperparameter tuning
- Extend sensitive attribute detection with user-defined categories
- Build a full React/Next.js UI with interactive charts and dashboards
- Add async batch processing for larger datasets
- Add exportable PDF reports for fairness audits
- Add a dedicated ChromaDB-backed explanation search experience

---

## Notes for Hackathon Judges

FairSight is designed to be both technically robust and demo-ready. It combines model-level fairness auditing with human-readable Gemini explanations so judges can quickly see both the algorithmic gap and the real-world implications.

---

## Screenshot Placeholder

![Screenshot placeholder](./screenshot-placeholder.png)

Update this section with your own UI screenshot once the demo is polished.
