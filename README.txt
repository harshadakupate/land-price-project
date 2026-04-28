============================================================
  LAND AI — Kolhapur Land Price Prediction System
  Final AIML Project | 3rd Year
============================================================

SETUP INSTRUCTIONS
------------------
1. Install dependencies:
   pip install flask pandas numpy scikit-learn joblib openpyxl

2. Train the model (generates model_clean.pkl):
   python model_final.py

3. Run the app:
   python app.py

4. Open browser: http://127.0.0.1:5000
   Login: admin / 1234

FILES INCLUDED
--------------
app.py                          - Main Flask application
model_final.py                  - Training script (run this first)
kolhapur_land_dataset_FINAL.csv - Dataset (3080 rows, 76 villages)
model_clean.pkl                 - Trained model (pre-generated)
model_metadata.json             - Model comparison results
feature_columns.pkl             - Feature column list
requirements.txt                - Python dependencies

templates/
  login.html                    - Login page
  dashboard.html                - Dashboard with model stats
  model_report.html             - AI Model Performance Report
  section1_input/output.html    - Land Price Prediction
  section2_input/output.html    - Smart Recommendation + Map
  section3_input/output.html    - Location Comparison
  error.html                    - Error page

static/css/style.css            - Master stylesheet
static/js/script.js             - Master JS

FEATURES IMPLEMENTED
--------------------
1. Multiple Model Comparison  (RF, GBM, Linear, Ridge)
2. Feature Importance Chart   (in Section 1 + Model Report)
3. Holdout Validation         (5 unseen villages tested)
4. Leaflet Map                (Section 2 live map)
5. Confidence Interval        (±range shown in Section 1 & 3)
6. Investment Signal          (BUY NOW / HOLD / WAIT / CAUTION)
7. Gender-based Stamp Duty    (Maharashtra real rules)

DATASET
-------
- 76 real Kolhapur district villages across 12 talukas
- 3080 rows (40 rows/village with realistic variation)
- Stamp duty: Urban M=6% F=5% | Semi-Urban M=4% F=3% | Rural M=3% F=2%
- Registration always 1%, capped at Rs.30,000

MODEL PERFORMANCE
-----------------
Best Model: Gradient Boosting
Test R2:    0.9858
Holdout R2: 0.7292 (on 5 completely unseen villages)
MAE:        Rs.77/sqft
============================================================
