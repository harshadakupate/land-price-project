# 🌍 Land Price Prediction System

A Machine Learning based web application that predicts land prices in Kolhapur district based on user inputs like location, area, and land type.

---

## 🚀 Features

* 🔍 Predict land prices using ML model
* 📊 Interactive dashboard
* 🧠 Model report and insights
* 🧾 Multiple input sections for different predictions
* 🌐 Web-based interface using Flask

---

## 🛠️ Tech Stack

* **Frontend:** HTML, CSS, JavaScript
* **Backend:** Python (Flask)
* **Machine Learning:** Scikit-learn
* **Data Processing:** Pandas, NumPy

---

## 📁 Project Structure

```
land_final/
│
├── app.py
├── model_clean.pkl
├── kolhapur_land_dataset_FINAL.csv
├── requirements.txt
├── Procfile
│
├── static/
│   ├── css/
│   └── js/
│
├── templates/
│   ├── login.html
│   ├── dashboard.html
│   ├── section1_input.html
│   ├── section1_output.html
│   ├── section2_input.html
│   ├── section2_output.html
│   ├── section3_input.html
│   ├── section3_output.html
│   ├── model_report.html
│   └── error.html
```

---

## ⚙️ Installation (Run Locally)

1. Clone the repository:

```
git clone https://github.com/harshadakupate/land-price-project.git
```

2. Navigate to project folder:

```
cd land-price-project
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Run the application:

```
python app.py
```

5. Open in browser:

```
http://127.0.0.1:5000/
```

---

## 🌐 Deployment

This project is deployed using Render.

Steps:

* Push code to GitHub
* Connect GitHub repo to Render
* Set build command: `pip install -r requirements.txt`
* Set start command: `gunicorn app:app`

---

## 📊 Dataset

* Contains land data from Kolhapur district
* Includes features like:

  * Location
  * Area type
  * Land size
  * Price

---

## 🧠 Machine Learning Model

* Algorithm used: Regression Model
* Trained using Scikit-learn
* Model file: `model_clean.pkl`

---

## 👩‍💻 Author

**Harshada Kupate**
AIML Student | Full Stack Learner

---

## 📌 Future Improvements

* Add user authentication system
* Improve UI/UX design
* Add more accurate prediction models
* Integrate Power BI dashboard

---

## ⭐ Show Your Support

If you like this project, give it a ⭐ on GitHub!
