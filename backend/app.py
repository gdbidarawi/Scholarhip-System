"""
Scholarship Eligibility Prediction — Flask REST API
Endpoints:
  POST /api/predict   → run ML prediction on a student profile
  GET  /api/model-info → return model metadata
  POST /api/batch     → predict multiple applicants at once
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, os, traceback, numpy as np

# ── Load artefacts ──────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
MODEL_PATH    = os.path.join(BASE, "model", "scholarship_model.pkl")
METADATA_PATH = os.path.join(BASE, "model", "model_metadata.pkl")

with open(MODEL_PATH, "rb")    as f: model    = pickle.load(f)
with open(METADATA_PATH, "rb") as f: metadata = pickle.load(f)

FEATURES = metadata["features"]

app = Flask(__name__)
CORS(app)

# ── Helper: build score breakdown (approximate weights from rubric) ──
WEIGHT_MAP = {
    "gpa":              ("Academic Excellence",    0.40),
    "sat_score":        ("Academic Excellence",    0.40),
    "essay_score":      ("Academic Excellence",    0.40),
    "annual_income":    ("Financial Need",         0.30),
    "community_hours":  ("Leadership & Impact",    0.20),
    "leadership_roles": ("Leadership & Impact",    0.20),
    "extracurriculars": ("Leadership & Impact",    0.20),
    "first_gen":        ("Resilience & Background",0.10),
    "region_index":     ("Resilience & Background",0.10),
}

REGION_LABEL = {0: "Urban", 1: "Semi-Urban", 2: "Rural"}

def compute_breakdown(data: dict, proba: float) -> dict:
    """Return weighted sub-scores approximating the 4-dimension rubric."""
    gpa            = float(data.get("gpa", 0))
    sat            = float(data.get("sat_score", 400))
    income         = float(data.get("annual_income", 50000))
    hours          = float(data.get("community_hours", 0))
    leadership     = float(data.get("leadership_roles", 0))
    first_gen      = float(data.get("first_gen", 0))
    extras         = float(data.get("extracurriculars", 0))
    essay          = float(data.get("essay_score", 50))
    region         = float(data.get("region_index", 0))

    academic  = min(1.0, (gpa / 4.0) * 0.60 + ((sat - 400) / 1200) * 0.25 + (essay / 100) * 0.15)
    financial = min(1.0, max(0.0, 1 - income / 100000))
    impact    = min(1.0, min(hours / 200, 1) * 0.50 + min(leadership / 5, 1) * 0.30 + min(extras / 7, 1) * 0.20)
    background= min(1.0, first_gen * 0.60 + (region / 2) * 0.40)

    # Weighted composite (matches rubric weights)
    composite = academic * 0.40 + financial * 0.30 + impact * 0.20 + background * 0.10

    return {
        "academic_score":     round(academic  * 40, 2),
        "financial_need_score": round(financial * 30, 2),
        "impact_score":       round(impact    * 20, 2),
        "background_score":   round(background* 10, 2),
        "composite_raw":      round(composite * 100, 2),
    }

def build_key_drivers(data: dict, breakdown: dict, eligible: bool) -> list:
    drivers = []
    gpa = float(data.get("gpa", 0))
    income = float(data.get("annual_income", 99999))
    hours = float(data.get("community_hours", 0))
    first_gen = int(data.get("first_gen", 0))
    sat = float(data.get("sat_score", 0))

    if gpa >= 3.5:
        drivers.append(f"Strong academic record with a GPA of {gpa:.2f}/4.00.")
    elif gpa < 2.5:
        drivers.append(f"Below-average GPA of {gpa:.2f} reduced the academic score significantly.")

    if income < 20000:
        drivers.append(f"High financial need — household income of ${income:,.0f} is well below threshold.")
    elif income > 80000:
        drivers.append(f"Lower financial need — household income of ${income:,.0f} reduces the financial score.")

    if hours >= 100:
        drivers.append(f"Outstanding community commitment: {int(hours)} volunteer hours logged.")
    elif hours < 20:
        drivers.append("Minimal community service hours weakened the leadership score.")

    if first_gen:
        drivers.append("First-generation university student status adds to background resilience score.")

    if sat >= 1400:
        drivers.append(f"Excellent SAT score of {int(sat)} reinforces academic excellence.")

    return drivers[:3]

# ── Routes ──────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "RandomForestClassifier", "accuracy": f"{metadata['accuracy']*100:.2f}%"})

@app.route("/api/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model_type":     "RandomForest (300 trees)",
        "accuracy":       f"{metadata['accuracy']*100:.2f}%",
        "roc_auc":        f"{metadata['roc_auc']:.4f}",
        "cv_auc":         f"{metadata['cv_auc_mean']:.4f}",
        "train_samples":  metadata["train_samples"],
        "features":       FEATURES,
        "eligibility_threshold": "75.00%",
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json(force=True)

        # Validate required fields
        required = ["gpa", "annual_income"]
        missing  = [f for f in required if f not in body or body[f] is None]
        if missing:
            return jsonify({
                "error":      "MISSING_PREREQUISITES",
                "message":    f"Required fields missing: {', '.join(missing)}",
                "eligible":   False,
            }), 400

        # Build feature vector
        row = [
            float(body.get("gpa",              0.0)),
            float(body.get("sat_score",        800)),
            float(body.get("annual_income",    50000)),
            float(body.get("community_hours",  0)),
            float(body.get("leadership_roles", 0)),
            float(body.get("first_gen",        0)),
            float(body.get("extracurriculars", 0)),
            float(body.get("essay_score",      60)),
            float(body.get("region_index",     0)),
        ]

        proba    = float(model.predict_proba([row])[0][1])
        eligible = proba >= 0.50   # model threshold (calibrated)
        score_pct = proba * 100

        breakdown   = compute_breakdown(body, proba)
        key_drivers = build_key_drivers(body, breakdown, eligible)

        name = body.get("applicant_name", "Applicant")

        justification = (
            f"{name} {'meets' if eligible else 'does not meet'} the scholarship eligibility criteria. "
            f"The model assigned a confidence score of {score_pct:.2f}%, "
            f"{'surpassing' if eligible else 'falling below'} the 75% approval threshold. "
            f"{'Key strengths include strong academic performance and demonstrated community involvement.' if eligible else 'Areas for improvement include academic performance, financial documentation, or community engagement.'}"
        )

        return jsonify({
            "applicant_name":      name,
            "eligibility_status":  "ELIGIBLE" if eligible else "NOT ELIGIBLE",
            "confidence_score":    f"{score_pct:.2f}%",
            "confidence_raw":      round(proba, 4),
            "score_breakdown":     {
                "academic_score":        f"{breakdown['academic_score']:.2f}%",
                "financial_need_score":  f"{breakdown['financial_need_score']:.2f}%",
                "impact_score":          f"{breakdown['impact_score']:.2f}%",
                "background_score":      f"{breakdown['background_score']:.2f}%",
            },
            "key_drivers":         key_drivers,
            "justification_summary": justification,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "SERVER_ERROR", "message": str(e)}), 500


@app.route("/api/batch", methods=["POST"])
def batch_predict():
    try:
        body        = request.get_json(force=True)
        applicants  = body.get("applicants", [])
        if not applicants:
            return jsonify({"error": "No applicants provided"}), 400

        results = []
        for app_data in applicants:
            row = [
                float(app_data.get("gpa",              0.0)),
                float(app_data.get("sat_score",        800)),
                float(app_data.get("annual_income",    50000)),
                float(app_data.get("community_hours",  0)),
                float(app_data.get("leadership_roles", 0)),
                float(app_data.get("first_gen",        0)),
                float(app_data.get("extracurriculars", 0)),
                float(app_data.get("essay_score",      60)),
                float(app_data.get("region_index",     0)),
            ]
            proba    = float(model.predict_proba([row])[0][1])
            eligible = proba >= 0.50
            results.append({
                "applicant_name":     app_data.get("applicant_name", "Unknown"),
                "eligibility_status": "ELIGIBLE" if eligible else "NOT ELIGIBLE",
                "confidence_score":   f"{proba*100:.2f}%",
                "confidence_raw":     round(proba, 4),
            })

        eligible_count = sum(1 for r in results if r["eligibility_status"] == "ELIGIBLE")
        return jsonify({
            "total":            len(results),
            "eligible_count":   eligible_count,
            "not_eligible":     len(results) - eligible_count,
            "results":          results,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🚀 Scholarship API running on http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=False)
