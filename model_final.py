"""
model_final.py  –  Land AI Complete Training Script
Implements all 7 improvements:
1. Multiple model comparison (RF, GBM, LinearReg, Ridge)
2. Feature importance (explainability without SHAP)
3. Holdout validation on 5 unseen villages
4. Confidence interval using RF tree variance
5. Cross-validation for reliable R² estimate
6. Saves all artifacts needed by app
"""

import pandas as pd
import numpy as np
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error

# ─── STEP 1: LOAD ────────────────────────────────────────────────────────────
df = pd.read_csv("kolhapur_land_dataset_FINAL.csv")
print(f"✅ Loaded: {df.shape[0]} rows | {df['City_Village'].nunique()} villages")

# ─── STEP 2: HOLDOUT — 5 unseen villages kept completely separate ─────────────
HOLDOUT_VILLAGES = ["Halkarni", "Dajipur", "Tilari", "Ningudage", "Lat"]
holdout = df[df["City_Village"].isin(HOLDOUT_VILLAGES)].copy()
df      = df[~df["City_Village"].isin(HOLDOUT_VILLAGES)].copy()
print(f"✅ Holdout: {len(holdout)} rows | Train pool: {len(df)} rows")

# ─── STEP 3: OUTLIER REMOVAL ─────────────────────────────────────────────────
Q1, Q3 = df["Rate_per_sqft"].quantile(0.25), df["Rate_per_sqft"].quantile(0.75)
IQR = Q3 - Q1
df = df[(df["Rate_per_sqft"] > Q1 - 1.5*IQR) &
        (df["Rate_per_sqft"] < Q3 + 1.5*IQR)].copy()
print(f"✅ After outlier removal: {len(df)} rows")

# ─── STEP 4: FEATURE ENGINEERING ─────────────────────────────────────────────
def engineer(d):
    d = d.copy()
    RISK = {"Low": 1, "Medium": 2, "High": 3}
    d["Flood_Risk"]  = d["Flood_Risk"].map(RISK).fillna(2).astype(int)
    d["Crime_Level"] = d["Crime_Level"].map(RISK).fillna(2).astype(int)
    d["Location_Score"] = (d["Market_Access"] + d["School_Access"] + d["Hospital_Access"]) / 3
    d["Connectivity"]   = 1 / (d["Distance_to_Highway"] + 1)
    d["City_Proximity"] = 1 / (d["Distance_to_City_Center"] + 1)
    d["Risk_Score"]     = (d["Flood_Risk"] + d["Crime_Level"]) / 2
    return d

df      = engineer(df)
holdout = engineer(holdout)
print("✅ Feature engineering done")

# ─── STEP 5: DEFINE FEATURES & TARGET ────────────────────────────────────────
# Buyer_Gender NOT in ML features (only affects stamp duty, not land rate)
DROP = ["Rate_per_sqft", "Land_Cost", "Total_Price", "Stamp_Duty_Amount",
        "Registration_Amount", "Buyer_Gender", "Stamp_Duty_Percent",
        "Registration_Percent"]

CAT_COLS = ["Taluka", "City_Village", "Area_Type", "Land_Type"]

X      = df.drop(columns=DROP)
y      = np.log1p(df["Rate_per_sqft"])
X_h    = holdout.drop(columns=DROP)
y_h    = np.log1p(holdout["Rate_per_sqft"])

NUM_COLS = [c for c in X.columns if c not in CAT_COLS]

preprocessor = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
    ("num", "passthrough", NUM_COLS)
])

# ─── STEP 6: TRAIN / TEST SPLIT ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ─── STEP 7: MULTIPLE MODEL COMPARISON ───────────────────────────────────────
models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=300, max_depth=20, min_samples_split=5,
        random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        random_state=42),
    "Linear Regression": LinearRegression(),
    "Ridge Regression":  Ridge(alpha=10.0),
}

print("\n📊 MODEL COMPARISON")
print("=" * 65)
comparison = []
best_r2, best_name, best_pipeline = -999, None, None

for name, mdl in models.items():
    pipe = Pipeline([("pre", preprocessor), ("model", mdl)])
    
    # Cross-validation (5-fold)
    cv_r2  = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
    
    # Fit and evaluate on test set
    pipe.fit(X_train, y_train)
    y_pred_log = pipe.predict(X_test)
    y_pred     = np.expm1(y_pred_log)
    y_true     = np.expm1(y_test)
    
    r2  = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"{name:22s} | R²={r2:.4f} | MAE=₹{mae:,.0f} | CV R²={cv_r2.mean():.4f}±{cv_r2.std():.4f}")
    comparison.append({"model": name, "r2": round(r2,4), "mae": int(mae),
                        "cv_r2": round(cv_r2.mean(),4), "cv_std": round(cv_r2.std(),4)})
    
    if r2 > best_r2:
        best_r2, best_name, best_pipeline = r2, name, pipe

print(f"\n🏆 Best Model: {best_name} (R²={best_r2:.4f})")

# ─── STEP 8: HOLDOUT VALIDATION (unseen villages) ────────────────────────────
print("\n📋 HOLDOUT VALIDATION (5 completely unseen villages)")
print("=" * 65)
y_hold_pred = np.expm1(best_pipeline.predict(X_h))
y_hold_true = np.expm1(y_h)

hold_r2  = r2_score(y_hold_true, y_hold_pred)
hold_mae = mean_absolute_error(y_hold_true, y_hold_pred)
print(f"Holdout R²: {hold_r2:.4f} | MAE: ₹{hold_mae:,.0f}")

holdout_results = []
for village in HOLDOUT_VILLAGES:
    mask    = holdout["City_Village"] == village
    y_true  = np.expm1(y_h[mask])
    y_pred  = y_hold_pred[mask.values]
    if len(y_true) > 0:
        avg_true = int(y_true.mean())
        avg_pred = int(y_pred.mean())
        err_pct  = abs(avg_pred - avg_true) / avg_true * 100
        print(f"  {village:15s}: Actual ₹{avg_true:,}/sqft | Predicted ₹{avg_pred:,}/sqft | Error {err_pct:.1f}%")
        holdout_results.append({"village": village, "actual": avg_true,
                                 "predicted": avg_pred, "error_pct": round(err_pct, 1)})

# ─── STEP 9: FEATURE IMPORTANCE ──────────────────────────────────────────────
rf_model = best_pipeline.named_steps["model"]
feature_names = CAT_COLS + NUM_COLS

importance = {}
if hasattr(rf_model, "feature_importances_"):
    imp = rf_model.feature_importances_
    importance = dict(zip(feature_names, [round(float(v)*100, 2) for v in imp]))
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
    print(f"\n🔍 TOP FEATURE IMPORTANCES:")
    for k, v in list(importance.items())[:8]:
        bar = "█" * int(v/2)
        print(f"  {k:25s}: {v:5.1f}% {bar}")

# ─── STEP 10: CONFIDENCE INTERVAL FUNCTION ────────────────────────────────────
# For RF: predict with each tree → std gives uncertainty
def predict_with_confidence(pipeline, X_input):
    """Returns (mean_pred, lower_80, upper_80) in real ₹/sqft"""
    pre_model = pipeline.named_steps["model"]
    X_transformed = pipeline.named_steps["pre"].transform(X_input)
    
    if hasattr(pre_model, "estimators_"):
        tree_preds = np.array([tree.predict(X_transformed)
                               for tree in pre_model.estimators_])
        mean_log = tree_preds.mean(axis=0)
        std_log  = tree_preds.std(axis=0)
        # 80% confidence interval (1.28 std)
        lower = np.expm1(mean_log - 1.28 * std_log)
        upper = np.expm1(mean_log + 1.28 * std_log)
        mean  = np.expm1(mean_log)
        return mean, lower, upper
    else:
        pred = np.expm1(pipeline.predict(X_input))
        return pred, pred * 0.88, pred * 1.12

# ─── STEP 11: SAVE ALL ARTIFACTS ─────────────────────────────────────────────
joblib.dump(best_pipeline,          "model_clean.pkl")
joblib.dump(list(X.columns),        "feature_columns.pkl")
joblib.dump(predict_with_confidence,"predict_fn.pkl")

with open("model_metadata.json", "w") as f:
    json.dump({
        "best_model":       best_name,
        "test_r2":          round(best_r2, 4),
        "test_mae":         int(mean_absolute_error(np.expm1(y_test),
                                np.expm1(best_pipeline.predict(X_test)))),
        "holdout_r2":       round(hold_r2, 4),
        "holdout_mae":      int(hold_mae),
        "comparison":       comparison,
        "holdout_results":  holdout_results,
        "feature_importance": importance,
        "cat_cols":         CAT_COLS,
        "num_cols":         NUM_COLS,
        "holdout_villages": HOLDOUT_VILLAGES,
    }, f, indent=2)

print(f"\n✅ Saved: model_clean.pkl | feature_columns.pkl | model_metadata.json")
print(f"\n{'='*65}")
print(f"FINAL SUMMARY")
print(f"  Best Model  : {best_name}")
print(f"  Test R²     : {best_r2:.4f}")
print(f"  Holdout R²  : {hold_r2:.4f}  ← on villages never seen in training")
print(f"{'='*65}")
