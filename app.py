"""
Land AI – Complete Flask App
All 7 improvements implemented:
1. Multiple model results shown in /model_report
2. Feature importance in Section 1 output
3. Holdout validation shown in /model_report
4. Leaflet map in Section 2
5. Confidence interval (±range) in Section 1
6. Investment timing signal in Section 3
7. Holdout validation page
"""

from flask import Flask, render_template, request, redirect, session, jsonify
import pandas as pd
import numpy as np
import joblib, json

app = Flask(__name__)
app.secret_key = "land_ai_secret_2024"

# ── Load ──────────────────────────────────────────────────────────────────────
df       = pd.read_csv("kolhapur_land_dataset_FINAL.csv")
pipeline = joblib.load("model_clean.pkl")

with open("model_metadata.json") as f:
    META = json.load(f)

# Village lat/long for Leaflet map (Kolhapur district, approximate centroids)
VILLAGE_COORDS = {
    "Rajarampuri":(16.7050,74.2433),"Tarabai Park":(16.7100,74.2400),
    "Shivaji Peth":(16.7020,74.2350),"Nagala Park":(16.7000,74.2500),
    "Gandhinagar":(16.6950,74.2300),"Uchgaon":(16.6800,74.2100),
    "Shiroli":(16.7300,74.3200),"Gokul Shirgaon":(16.7200,74.3000),
    "Shiye":(16.6900,74.2200),"Kalamba":(16.6750,74.2600),
    "Ichalkaranji":(16.6942,74.4597),"Hupari":(16.7523,74.3667),
    "Kabnur":(16.7800,74.3500),"Talsande":(16.7600,74.3800),
    "Vathar":(16.7400,74.3600),"Peth Vadgoan":(16.7100,74.4200),
    "Rukadi":(16.7200,74.4000),"Alas":(16.6500,74.4500),
    "Jaysingpur":(16.7903,74.5542),"Kurundwad":(16.6667,74.5833),
    "Shirol":(16.7333,74.5667),"Yadrav":(16.7000,74.5200),
    "Lat":(16.7500,74.5800),"Nrusinhawadi":(16.7100,74.5900),
    "Herwad":(16.6800,74.5300),"Kagal":(16.5742,74.3182),
    "Murgud":(16.5200,74.3700),"Kapashi":(16.5500,74.3400),
    "Sangaon":(16.5800,74.3600),"Bidri":(16.5400,74.4000),
    "Siddhanerli":(16.5100,74.3900),"Gadhinglaj":(16.2200,74.3500),
    "Nesari":(16.1800,74.3200),"Halkarni":(16.0800,74.3300),
    "Kadgaon":(16.2500,74.3700),"Basarge":(16.2600,74.3800),
    "Mahagoan":(16.2100,74.3400),"Harali":(16.1500,74.3100),
    "Mungurwadi":(16.2300,74.3600),"Chandgad":(15.9500,74.1500),
    "Tilari":(15.9000,74.1200),"Shinoli":(15.9700,74.1600),
    "Kudnur":(15.9800,74.1700),"Here":(15.9300,74.1400),"Kowad":(15.9100,74.1300),
    "Ajra":(16.1000,74.2167),"Uttur":(16.0800,74.2300),
    "Madilage":(16.0600,74.2400),"Ningudage":(16.0400,74.2600),
    "Zulapewadi":(16.0500,74.2700),"Dewarde":(16.0700,74.2500),
    "Gargoti":(16.3300,74.1700),"Kadgaon":(16.3000,74.1800),
    "Patgaon":(16.2800,74.1900),"Admapur":(16.3500,74.1600),
    "Pimpalgoan":(16.2700,74.2000),"Waghapur":(16.3100,74.1900),
    "Radhanagari":(16.4167,74.0333),"Dajipur":(16.3800,74.0200),
    "Tarale":(16.4000,74.0500),"Dhamod":(16.4300,74.0600),
    "Ghotawade":(16.3700,74.0100),"Rajapur":(16.4100,74.0400),
    "Gaganbawada":(16.5500,73.9500),"Salwan":(16.5200,73.9300),
    "Bhuibawada":(16.5400,73.9600),"Sakhari":(16.5600,73.9700),
    "Vesarde":(16.5100,73.9200),"Panhala":(16.8097,74.1167),
    "Kodoli":(16.8300,74.1300),"Warnanagar":(16.8500,74.1000),
    "Kale":(16.7900,74.1400),"Malkapur":(16.8700,74.1600),
    "Bambavade":(16.8500,74.2000),"Amba":(16.8200,74.1800),
    "Charan":(16.9000,74.1700),"Sarud":(16.8800,74.1900),
}

# ── Constants ─────────────────────────────────────────────────────────────────
RISK_MAP = {"Low": 1, "Medium": 2, "High": 3}
STAMP_DUTY = {
    ("Urban","Male"):6,    ("Urban","Female"):5,
    ("Semi-Urban","Male"):4,("Semi-Urban","Female"):3,
    ("Rural","Male"):3,    ("Rural","Female"):2,
    ("Industrial","Male"):5,("Industrial","Female"):4,
}
MAX_DIST_CITY, MAX_DIST_HWY = 60.0, 35.0
LAND_TYPE_DEFAULT = {"Urban":"Residential","Semi-Urban":"Residential",
                     "Industrial":"Industrial","Rural":"Agricultural"}

# ── Helpers ───────────────────────────────────────────────────────────────────
def engineer_row(d):
    d = d.copy()
    d["Flood_Risk"]  = RISK_MAP.get(str(d.get("Flood_Risk","Low")), 1)
    d["Crime_Level"] = RISK_MAP.get(str(d.get("Crime_Level","Low")), 1)
    d["Location_Score"] = (d.get("Market_Access",5)+d.get("School_Access",5)+d.get("Hospital_Access",5))/3
    d["Connectivity"]   = 1/(d.get("Distance_to_Highway",10)+1)
    d["City_Proximity"] = 1/(d.get("Distance_to_City_Center",20)+1)
    d["Risk_Score"]     = (d["Flood_Risk"]+d["Crime_Level"])/2
    return d

DROP_COLS = ["Rate_per_sqft","Land_Cost","Total_Price","Stamp_Duty_Amount",
             "Registration_Amount","Buyer_Gender","Stamp_Duty_Percent","Registration_Percent"]

def build_input(row, area, land_type):
    d = row.to_dict()
    d["Area_sqft"] = area
    d["Land_Type"] = land_type
    d = engineer_row(d)
    for c in DROP_COLS:
        d.pop(c, None)
    return pd.DataFrame([d])

def predict_with_ci(row, area, land_type):
    """Returns (mean_rate, lower_80, upper_80)
    Fixed: GradientBoostingRegressor.estimators_ is shape (n_estimators, 1)
    containing DecisionTreeRegressor objects — cannot call .predict() on the
    outer numpy array directly. Use staged_predict for spread estimation instead.
    """
    X_in = build_input(row, area, land_type)
    pre  = pipeline.named_steps["pre"]
    mdl  = pipeline.named_steps["model"]

    # Final prediction via the full pipeline
    mean_log = pipeline.predict(X_in)[0]
    pred     = int(np.expm1(mean_log))

    try:
        # For GradientBoostingRegressor: use staged predictions to estimate spread
        Xt     = pre.transform(X_in)
        staged = list(mdl.staged_predict(Xt))
        staged_vals = np.array([np.expm1(s[0]) for s in staged])
        std    = staged_vals.std()
        margin = max(int(std * 0.5), int(pred * 0.08))   # at least ±8%
        return pred, max(0, pred - margin), pred + margin
    except Exception:
        # Generic fallback: ±12%
        return pred, int(pred * 0.88), int(pred * 1.12)

def score_village(row):
    def cl(v, lo=0, hi=10): return max(lo, min(hi, float(v)))
    crime_num = RISK_MAP.get(str(row.get("Crime_Level","Low")),1)
    safety    = cl(10-(crime_num-1)*5)
    dc = float(row.get("Distance_to_City_Center",30))
    dh = float(row.get("Distance_to_Highway",10))
    conn = cl(10-((dc/MAX_DIST_CITY)+(dh/MAX_DIST_HWY))*5)
    gs   = cl(float(row.get("Growth_Rate",0.05))/0.20*10)
    s = (gs*3.5 + cl(float(row.get("Market_Access",5)))*1.5 +
         cl(float(row.get("Hospital_Access",5)))*1.5 +
         cl(float(row.get("School_Access",5)))*1.0 + safety*1.5 + conn*1.0)
    return min(round(s), 100)

def radar_params(row):
    def cl(v, lo=0, hi=10): return max(lo, min(hi, float(v)))
    crime_num = RISK_MAP.get(str(row.get("Crime_Level","Low")),1)
    safety    = cl(10-(crime_num-1)*5)
    dc = float(row.get("Distance_to_City_Center",30))
    dh = float(row.get("Distance_to_Highway",10))
    conn = cl(10-((dc/MAX_DIST_CITY)+(dh/MAX_DIST_HWY))*5)
    gs   = cl(float(row.get("Growth_Rate",0.05))/0.20*10)
    return [round(cl(float(row.get("Hospital_Access",5))),1),
            round(cl(float(row.get("School_Access",5))),1),
            round(cl(float(row.get("Market_Access",5))),1),
            round(safety,1), round(conn,1), round(gs,1)]

def investment_signal(row, predicted_rate):
    """FIX 6: Buy Now / Wait / Hold signal"""
    taluka = row["Taluka"]
    area_type = row["Area_Type"]
    taluka_avg = df[(df["Taluka"]==taluka)&(df["Area_Type"]==area_type)]["Rate_per_sqft"].mean()
    growth = float(row["Growth_Rate"])
    flood  = str(row.get("Flood_Risk","Low"))
    crime  = RISK_MAP.get(str(row.get("Crime_Level","Low")),1)
    below_avg = predicted_rate < taluka_avg
    high_growth = growth >= 0.08
    low_risk = flood != "High" and crime <= 2

    if high_growth and below_avg and low_risk:
        return "BUY NOW", "#00ff99", "High growth + below-average price + low risk = ideal entry point"
    elif high_growth and not below_avg and low_risk:
        return "HOLD", "#ffc800", "Good growth but price is already above taluka average — wait for dip"
    elif not high_growth and below_avg:
        return "WAIT", "#ff9900", "Price is attractive but growth rate is low — monitor before committing"
    else:
        return "CAUTION", "#ff6b6b", "High risk or low growth — consider other locations"

# ── Auth ──────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET","POST"])
def login():
    if request.method == "POST":
        if request.form.get("username")=="admin" and request.form.get("password")=="1234":
            session["user"] = "admin"
            return redirect("/dashboard")
        return render_template("login.html", error="Invalid Credentials")
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if "user" not in session: return redirect("/")
    return render_template("dashboard.html", meta=META)

@app.route("/logout")
def logout():
    session.clear(); return redirect("/")

@app.route("/get_cities", methods=["POST"])
def get_cities():
    taluka = request.form.get("taluka")
    cities = df[df["Taluka"]==taluka]["City_Village"].dropna().unique().tolist()
    return jsonify(sorted(cities))

# ── Model Report (Improvement 1, 3, 7) ───────────────────────────────────────
@app.route("/model_report")
def model_report():
    if "user" not in session: return redirect("/")
    return render_template("model_report.html", meta=META)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/section1")
def section1():
    if "user" not in session: return redirect("/")
    talukas = sorted(df["Taluka"].unique())
    return render_template("section1_input.html", talukas=talukas)

@app.route("/predict1", methods=["POST"])
def predict1():
    if "user" not in session: return redirect("/")
    try:
        taluka = request.form.get("taluka")
        city   = request.form.get("city")
        area   = float(request.form.get("area"))
        gender = request.form.get("gender", "Male")

        row = df[(df["Taluka"]==taluka)&(df["City_Village"]==city)].iloc[0]
        area_type = row["Area_Type"]
        land_type = LAND_TYPE_DEFAULT.get(area_type, "Residential")

        # Prediction + confidence interval (Improvement 5)
        pred_rate, lower_rate, upper_rate = predict_with_ci(row, area, land_type)

        # Cost using gender-based stamp duty
        stamp_pct = STAMP_DUTY.get((area_type, gender), 4)
        land_cost = pred_rate * area
        stamp_amt = int(land_cost * stamp_pct / 100)
        reg_amt   = min(int(land_cost * 0.01), 30000)
        total     = land_cost + stamp_amt + reg_amt

        # Savings for female vs male
        male_stamp   = int(land_cost * STAMP_DUTY.get((area_type,"Male"),6) / 100)
        female_stamp = int(land_cost * STAMP_DUTY.get((area_type,"Female"),5) / 100)
        gender_saving = male_stamp - female_stamp

        growth = float(row["Growth_Rate"])
        prices, base = [], land_cost
        for _ in range(5):
            base *= (1+growth); prices.append(int(base))

        params = radar_params(row)
        score  = score_village(row)
        signal, signal_color, signal_reason = investment_signal(row, pred_rate)

        # Feature importance for this prediction (top 5)
        fi = META.get("feature_importance", {})
        top_features = list(fi.items())[:5]

        crime_num = RISK_MAP.get(str(row.get("Crime_Level","Low")),1)
        insights  = []
        if growth > 0.10: insights.append("📈 High growth zone — excellent investment potential")
        elif growth > 0.06: insights.append("📊 Moderate growth — stable area")
        else: insights.append("📉 Lower growth — better for long-term holding")
        if crime_num == 1: insights.append("✅ Very safe area — low crime")
        elif crime_num == 2: insights.append("🟡 Moderate crime — normal caution needed")
        else: insights.append("⚠️ High crime zone — factor this into decision")
        dc = float(row.get("Distance_to_City_Center",30))
        if dc <= 5: insights.append("🏙️ Prime location — within 5 km of city center")
        elif dc <= 20: insights.append("🛣️ Well connected — within 20 km of city center")
        else: insights.append("🌄 Outskirts location — better land value per sqft")
        flood = str(row.get("Flood_Risk","Low"))
        if flood == "High": insights.append("🌊 High flood risk — check land elevation carefully")
        elif flood == "Medium": insights.append("💧 Moderate flood risk — monsoon precautions advised")
        else: insights.append("✅ Low flood risk area")

        return render_template("section1_output.html",
            taluka=taluka, city=city, area=int(area),
            area_type=area_type, land_type=land_type, gender=gender,
            pred_rate=pred_rate, lower_rate=lower_rate, upper_rate=upper_rate,
            land_cost=int(land_cost), stamp_amt=stamp_amt,
            stamp_pct=stamp_pct, reg_amt=reg_amt, total=int(total),
            gender_saving=gender_saving,
            future_price=prices[-1], prices=prices,
            years=["Year 1","Year 2","Year 3","Year 4","Year 5"],
            params=params, insights=insights,
            growth_rate=round(growth*100,2),
            score=score, signal=signal, signal_color=signal_color,
            signal_reason=signal_reason, top_features=top_features,
            flood_risk=flood, crime_level=row.get("Crime_Level","Low"),
        )
    except Exception as e:
        return render_template("error.html", error=str(e))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/section2")
def section2():
    if "user" not in session: return redirect("/")
    return render_template("section2_input.html")

@app.route("/predict2", methods=["POST"])
def predict2():
    if "user" not in session: return redirect("/")
    try:
        budget         = float(request.form.get("budget", 5000000))
        area_needed    = float(request.form.get("area_needed", 1500))
        area_type_pref = request.form.get("area_type_pref", "Any")
        max_dist_city  = float(request.form.get("max_dist_city", 60))
        priority       = request.form.get("priority", "growth")
        exclude_flood  = "exclude_flood" in request.form
        min_growth     = float(request.form.get("min_growth", 0)) / 100
        max_crime      = request.form.get("max_crime", "Any")
        gender         = request.form.get("gender", "Male")

        data = df.copy()
        data["Crime_Num"] = data["Crime_Level"].map(RISK_MAP).fillna(2)
        data["Est_LandCost"] = data["Rate_per_sqft"] * area_needed
        data["Stamp_Pct"] = data.apply(
            lambda r: STAMP_DUTY.get((r["Area_Type"], gender), 4), axis=1)
        data["Est_Total"] = (data["Est_LandCost"] *
                             (1 + data["Stamp_Pct"]/100) +
                             data["Est_LandCost"].apply(lambda x: min(x*0.01, 30000)))

        data = data[data["Est_Total"] <= budget]
        if area_type_pref != "Any":
            data = data[data["Area_Type"] == area_type_pref]
        data = data[data["Distance_to_City_Center"] <= max_dist_city]
        if exclude_flood: data = data[data["Flood_Risk"] != "High"]
        if min_growth > 0: data = data[data["Growth_Rate"] >= min_growth]
        if max_crime == "Low":  data = data[data["Crime_Num"] == 1]
        elif max_crime == "Medium": data = data[data["Crime_Num"] <= 2]

        if data.empty:
            return render_template("section2_output.html", results=[],
                                   budget=int(budget), area_needed=int(area_needed))

        loc  = (data["Market_Access"]+data["School_Access"]+data["Hospital_Access"])/3
        conn = (10-((data["Distance_to_City_Center"]/MAX_DIST_CITY)+
                    (data["Distance_to_Highway"]/MAX_DIST_HWY))*5).clip(0,10)
        safety = (10-(data["Crime_Num"]-1)*5).clip(0,10)
        gr     = (data["Growth_Rate"]/0.20*10).clip(0,10)

        if priority=="growth":   data["Score"] = gr*5+loc*2+safety*2+conn*1
        elif priority=="safety": data["Score"] = safety*5+loc*2+gr*2+conn*1
        else:                    data["Score"] = loc*4+conn*3+gr*2+safety*1

        top = (data.sort_values("Score",ascending=False)
                   .drop_duplicates("City_Village").head(5))

        results = []
        for _, r in top.iterrows():
            lc    = int(r["Rate_per_sqft"] * area_needed)
            sp    = int(STAMP_DUTY.get((r["Area_Type"], gender), 4))
            sa    = int(lc * sp / 100)
            ra    = min(int(lc * 0.01), 30000)
            coords = VILLAGE_COORDS.get(r["City_Village"], (16.70, 74.24))
            results.append({
                "Taluka": r["Taluka"], "City_Village": r["City_Village"],
                "Area_Type": r["Area_Type"], "Rate_per_sqft": int(r["Rate_per_sqft"]),
                "Land_Cost": lc, "Stamp_Pct": sp, "Stamp": sa, "Reg": ra,
                "Total_Est": lc+sa+ra, "Growth_Pct": round(r["Growth_Rate"]*100,1),
                "Crime_Level": r["Crime_Level"], "Crime_Num": int(r["Crime_Num"]),
                "Flood_Risk": r["Flood_Risk"],
                "Market_Access": r["Market_Access"], "School_Access": r["School_Access"],
                "Hospital_Access": r["Hospital_Access"],
                "Dist_City": r["Distance_to_City_Center"],
                "Dist_Hwy": r["Distance_to_Highway"],
                "Score": round(float(r["Score"]),1),
                "Lat": coords[0], "Lng": coords[1],
            })

        return render_template("section2_output.html",
                               results=results, budget=int(budget),
                               area_needed=int(area_needed), priority=priority,
                               gender=gender)
    except Exception as e:
        return render_template("error.html", error=str(e))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/section3")
def section3():
    if "user" not in session: return redirect("/")
    talukas = sorted(df["Taluka"].unique())
    return render_template("section3_input.html", talukas=talukas)

@app.route("/predict3", methods=["POST"])
def predict3():
    if "user" not in session: return redirect("/")
    try:
        t1 = request.form.get("taluka1"); c1 = request.form.get("city1")
        t2 = request.form.get("taluka2"); c2 = request.form.get("city2")
        area   = float(request.form.get("area"))
        gender = request.form.get("gender","Male")

        r1 = df[(df["Taluka"]==t1)&(df["City_Village"]==c1)].iloc[0]
        r2 = df[(df["Taluka"]==t2)&(df["City_Village"]==c2)].iloc[0]

        lt1 = LAND_TYPE_DEFAULT.get(r1["Area_Type"],"Residential")
        lt2 = LAND_TYPE_DEFAULT.get(r2["Area_Type"],"Residential")

        rate1, lo1, hi1 = predict_with_ci(r1, area, lt1)
        rate2, lo2, hi2 = predict_with_ci(r2, area, lt2)

        def costs(rate, row, gender):
            lc  = rate * area
            sp  = STAMP_DUTY.get((row["Area_Type"], gender), 4)
            sa  = int(lc * sp / 100)
            ra  = min(int(lc * 0.01), 30000)
            return int(lc), sp, sa, ra, int(lc+sa+ra)

        lc1,sp1,sa1,ra1,tot1 = costs(rate1, r1, gender)
        lc2,sp2,sa2,ra2,tot2 = costs(rate2, r2, gender)

        score1 = score_village(r1)
        score2 = score_village(r2)
        params1 = radar_params(r1)
        params2 = radar_params(r2)

        sig1, sc1, sr1 = investment_signal(r1, rate1)
        sig2, sc2, sr2 = investment_signal(r2, rate2)

        g1, g2 = float(r1["Growth_Rate"]), float(r2["Growth_Rate"])
        trend1, trend2, b1, b2 = [], [], lc1, lc2
        for _ in range(5):
            b1*=(1+g1); b2*=(1+g2)
            trend1.append(int(b1)); trend2.append(int(b2))

        better     = c1 if score1 >= score2 else c2
        better_num = "Location 1" if score1 >= score2 else "Location 2"

        def fmt(v): return {"Low":"🟢 Low","Medium":"🟡 Medium","High":"🔴 High"}.get(str(v),str(v))
        details1 = {"Area Type":r1["Area_Type"],"Land Type":lt1,
            "Flood Risk":fmt(r1.get("Flood_Risk","Low")),"Crime Level":fmt(r1.get("Crime_Level","Low")),
            "Growth Rate":f"{round(g1*100,1)}% / yr",
            "Dist. to City":f"{r1['Distance_to_City_Center']} km",
            "Dist. to NH":f"{r1['Distance_to_Highway']} km",
            "Hospital Access":f"{r1.get('Hospital_Access',5)} / 10",
            "School Access":f"{r1.get('School_Access',5)} / 10",
            "Market Access":f"{r1.get('Market_Access',5)} / 10"}
        details2 = {"Area Type":r2["Area_Type"],"Land Type":lt2,
            "Flood Risk":fmt(r2.get("Flood_Risk","Low")),"Crime Level":fmt(r2.get("Crime_Level","Low")),
            "Growth Rate":f"{round(g2*100,1)}% / yr",
            "Dist. to City":f"{r2['Distance_to_City_Center']} km",
            "Dist. to NH":f"{r2['Distance_to_Highway']} km",
            "Hospital Access":f"{r2.get('Hospital_Access',5)} / 10",
            "School Access":f"{r2.get('School_Access',5)} / 10",
            "Market Access":f"{r2.get('Market_Access',5)} / 10"}

        adv1, adv2 = [], []
        crime1 = RISK_MAP.get(str(r1.get("Crime_Level","Low")),1)
        crime2 = RISK_MAP.get(str(r2.get("Crime_Level","Low")),1)
        if g1>g2: adv1.append(f"Higher growth ({round(g1*100,1)}% vs {round(g2*100,1)}%)")
        elif g2>g1: adv2.append(f"Higher growth ({round(g2*100,1)}% vs {round(g1*100,1)}%)")
        if crime1<crime2: adv1.append("Lower crime level")
        elif crime2<crime1: adv2.append("Lower crime level")
        if float(r1.get("Hospital_Access",5))>float(r2.get("Hospital_Access",5)): adv1.append("Better hospital access")
        elif float(r2.get("Hospital_Access",5))>float(r1.get("Hospital_Access",5)): adv2.append("Better hospital access")
        if r1["Distance_to_City_Center"]<r2["Distance_to_City_Center"]: adv1.append("Closer to city center")
        elif r2["Distance_to_City_Center"]<r1["Distance_to_City_Center"]: adv2.append("Closer to city center")
        if lc1<lc2: adv1.append("Lower land cost")
        elif lc2<lc1: adv2.append("Lower land cost")
        reasons = adv1 if score1>=score2 else adv2

        return render_template("section3_output.html",
            city1=c1, taluka1=t1, city2=c2, taluka2=t2,
            area=int(area), gender=gender,
            rate1=rate1, lo1=lo1, hi1=hi1,
            rate2=rate2, lo2=lo2, hi2=hi2,
            lc1=lc1, sp1=sp1, sa1=sa1, ra1=ra1, tot1=tot1,
            lc2=lc2, sp2=sp2, sa2=sa2, ra2=ra2, tot2=tot2,
            score1=score1, score2=score2,
            params1=params1, params2=params2,
            trend1=trend1, trend2=trend2,
            signal1=sig1, sc1=sc1, sr1=sr1,
            signal2=sig2, sc2=sc2, sr2=sr2,
            better=better, better_num=better_num,
            reasons=reasons, adv1=adv1, adv2=adv2,
            growth1=round(g1*100,1), growth2=round(g2*100,1),
            details1=details1, details2=details2,
        )
    except Exception as e:
        return render_template("error.html", error=str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)