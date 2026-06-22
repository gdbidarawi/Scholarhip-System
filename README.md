# ScholarAI — Scholarship Eligibility Prediction System
> A full-stack ML application: RandomForest model + Flask REST API + HTML/JS frontend

---

## Project Structure

```
scholarship_system/
├── backend/
│   ├── train_model.py        # Dataset generation + model training
│   ├── app.py                # Flask REST API
│   └── model/                # Saved model + metadata (auto-generated)
│       ├── scholarship_model.pkl
│       └── model_metadata.pkl
├── frontend/
│   └── index.html            # Full-featured UI (no build tools needed)
├── requirements.txt
├── run.sh                    # One-click launcher
└── README.md
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
cd backend
python3 train_model.py

# 3. Start the API
python3 app.py

# 4. Open frontend/index.html in your browser
```

Or just run:
```bash
bash run.sh
```

---

## ML Model Details

| Attribute         | Value                         |
|-------------------|-------------------------------|
| Algorithm         | RandomForest (300 estimators) |
| Accuracy          | 92.00%                        |
| ROC-AUC           | 0.9585                        |
| Cross-Val AUC     | 0.9538 ± 0.0051               |
| Training samples  | 2,400                         |
| Test samples      | 600                           |

### Evaluation Dimensions (mirrors document rubric)

| Dimension                  | Weight |
|----------------------------|--------|
| Academic Excellence        | 40%    |
| Financial Need             | 30%    |
| Leadership & Community     | 20%    |
| Resilience & Background    | 10%    |

### Features Used

- `gpa` — Cumulative GPA (0–4.0)
- `sat_score` — SAT score (400–1600)
- `annual_income` — Household income (USD)
- `community_hours` — Volunteer/service hours
- `leadership_roles` — Number of leadership positions
- `first_gen` — First-generation student (0/1)
- `extracurriculars` — Number of activities
- `essay_score` — Internal essay review score (0–100)
- `region_index` — 0=Urban, 1=Semi-Urban, 2=Rural

---

## API Endpoints

### `POST /api/predict`
Evaluate a single applicant profile.

**Request body:**
```json
{
  "applicant_name": "Yohannes Alemayehu",
  "gpa": 3.89,
  "sat_score": 1320,
  "annual_income": 12500,
  "community_hours": 140,
  "leadership_roles": 2,
  "first_gen": 1,
  "extracurriculars": 3,
  "essay_score": 82,
  "region_index": 1
}
```

**Response:**
```json
{
  "applicant_name": "Yohannes Alemayehu",
  "eligibility_status": "ELIGIBLE",
  "confidence_score": "91.20%",
  "score_breakdown": {
    "academic_score": "37.80%",
    "financial_need_score": "29.00%",
    "impact_score": "15.60%",
    "background_score": "8.80%"
  },
  "key_drivers": [
    "Strong academic record with a GPA of 3.89/4.00.",
    "High financial need — household income of $12,500 is well below threshold.",
    "140 volunteer hours demonstrate strong community commitment."
  ],
  "justification_summary": "..."
}
```

### `GET /api/model-info`
Returns model metadata, accuracy, and AUC.

### `POST /api/batch`
Evaluate multiple applicants in one request.

**Request body:**
```json
{ "applicants": [ { ...profile1 }, { ...profile2 } ] }
```

---

## Architecture: Two-Stage Pipeline (from source document)

```
Stage 1: ML Filter (scikit-learn)
  └── Fast numerical screening of basic profiles
         ↓
Stage 2: LLM Agent (optional)
  └── Deep analysis of essays, letters, narratives
      (Use the system prompt from the source document)
```

---

## Frontend Features

- Live eligibility score with animated gauge bar
- Dimension-by-dimension score breakdown
- Key decision drivers with justifications
- Offline fallback (runs approximation in browser if API is down)
- Demo profile pre-fill (Yohannes Alemayehu)
- Clean, responsive design — no build tools required

---

*ScholarAI · RandomForest · scikit-learn · Flask · Vanilla JS*
