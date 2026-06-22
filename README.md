# ScholarAI — Scholarship Eligibility Prediction System
> A full-stack ML application: RandomForest model + Flask REST API + HTML/JS frontend



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


Response:
json
GET /api/model-info`
Returns model metadata, accuracy, and AUC.

POST /api/batch`
Evaluate multiple applicants in one request.

Request body:
json

Frontend Features
 Live eligibility score with animated gauge bar
 Dimension-by-dimension score breakdown
 Key decision drivers with justifications
 Offline fallback (runs approximation in browser if API is down)
 Demo profile pre-fill (Yohannes Alemayehu)
 Clean, responsive design — no build tools required



ScholarAI · RandomForest · scikit-learn · Flask · Vanilla JS
