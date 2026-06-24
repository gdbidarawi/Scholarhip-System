"""
Scholarship Eligibility Prediction — Model Training
Uses a RandomForest Classifier trained on a synthetic dataset.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import pickle
import os

# ──────────────────────────────────────────────
# 1. SYNTHETIC DATASET GENERATION
# ──────────────────────────────────────────────
np.random.seed(42)
N = 3000

def generate_dataset(n):
    gpa                = np.clip(np.random.normal(3.2, 0.55, n), 0.0, 4.0)
    sat_score          = np.clip(np.random.normal(1150, 200, n), 400, 1600).astype(int)
    annual_income      = np.clip(np.random.exponential(35000, n), 5000, 200000).astype(int)
    community_hours    = np.clip(np.random.exponential(60, n), 0, 500).astype(int)
    leadership_roles   = np.random.randint(0, 6, n)
    first_gen          = np.random.binomial(1, 0.38, n)
    extracurriculars   = np.random.randint(0, 8, n)
    essay_score        = np.clip(np.random.normal(70, 18, n), 0, 100)
    region_index       = np.random.randint(0, 3, n)   # 0=urban,1=semi-urban,2=rural

    # ── Eligibility score formula (mirrors the rubric from the document) ──
    # Academic (40%)
    academic = (
        (gpa / 4.0) * 0.25 +
        ((sat_score - 400) / 1200) * 0.10 +
        (essay_score / 100) * 0.05
    )
    # Financial need (30%)  – lower income ⟹ higher need
    financial = (1 - np.clip(annual_income / 100000, 0, 1)) * 0.30
    # Leadership / community (20%)
    impact = (
        np.clip(community_hours / 200, 0, 1) * 0.10 +
        np.clip(leadership_roles / 5, 0, 1) * 0.06 +
        np.clip(extracurriculars / 7, 0, 1) * 0.04
    )
    # Background (10%)
    background = (
        first_gen * 0.06 +
        (region_index / 2) * 0.04
    )

    raw_score = academic + financial + impact + background
    noise     = np.random.normal(0, 0.04, n)
    final     = np.clip(raw_score + noise, 0, 1)
    eligible  = (final >= 0.75).astype(int)

    df = pd.DataFrame({
        "gpa":              gpa,
        "sat_score":        sat_score,
        "annual_income":    annual_income,
        "community_hours":  community_hours,
        "leadership_roles": leadership_roles,
        "first_gen":        first_gen,
        "extracurriculars": extracurriculars,
        "essay_score":      essay_score,
        "region_index":     region_index,
        "eligible":         eligible,
        "raw_score":        final,
    })
    return df

df = generate_dataset(N)

FEATURES = [
    "gpa", "sat_score", "annual_income", "community_hours",
    "leadership_roles", "first_gen", "extracurriculars",
    "essay_score", "region_index",
]

X = df[FEATURES]
y = df["eligible"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ──────────────────────────────────────────────
# 2. MODEL TRAINING
# ──────────────────────────────────────────────
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])

pipeline.fit(X_train, y_train)

# ──────────────────────────────────────────────
# 3. EVALUATION
# ──────────────────────────────────────────────
y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

acc     = accuracy_score(y_test, y_pred)
auc     = roc_auc_score(y_test, y_proba)
cv      = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")

print("=" * 55)
print("  SCHOLARSHIP ML MODEL — EVALUATION REPORT")
print("=" * 55)
print(f"  Accuracy :  {acc*100:.2f}%")
print(f"  ROC-AUC  :  {auc:.4f}")
print(f"  CV AUC   :  {cv.mean():.4f} ± {cv.std():.4f}")
print("-" * 55)
print(classification_report(y_test, y_pred, target_names=["NOT ELIGIBLE", "ELIGIBLE"]))

# Feature importances
rf     = pipeline.named_steps["clf"]
importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("Feature Importances:")
for feat, imp in importances.items():
    print(f"  {feat:<22}: {imp:.4f}")
print("=" * 55)

# ──────────────────────────────────────────────
# 4. PERSIST MODEL & METADATA
# ──────────────────────────────────────────────
os.makedirs("model", exist_ok=True)

with open("model/scholarship_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

metadata = {
    "features":      FEATURES,
    "threshold":     0.75,
    "accuracy":      float(acc),
    "roc_auc":       float(auc),
    "cv_auc_mean":   float(cv.mean()),
    "train_samples": len(X_train),
    "test_samples":  len(X_test),
}

with open("model/model_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("\n✅  Model saved → model/scholarship_model.pkl")
print("✅  Metadata saved → model/model_metadata.pkl")
