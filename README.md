# 🩸 Diabetes Risk ML App  
🩸 This project is for educational purposes only. Predictions are not intended for real medical diagnosis or treatment.
A complete **FastAPI + Machine Learning web application** that predicts diabetes risk using the **Scikit-Learn Diabetes Dataset**.  
It includes:
- A **web UI** built with HTML + CSS (Jinja2 templates)  
- A **REST API** (`/predict`, `/model/info`, `/health`)  
- A trained ML model pipeline (`train.py`)  
- Fully isolated Python environment (`.venv`)  

> ⚠️ *Educational demo only — not for medical use.*

---

## 🧠 Project Overview
This project demonstrates how to build, train, and deploy a machine learning model as a web service using **FastAPI**.  
The model predicts the **risk of diabetes** based on selected medical parameters.

Technologies used:
- **Python 3.10+**
- **FastAPI**
- **Scikit-Learn**
- **Uvicorn**
- **Jinja2 (templates)**
- **HTML + CSS**

---

## ⚙️ 1. Clone the Repository

```bash
git clone https://github.com/lakshminarasimha6802/Diabetes-Risk-ML-App.git
cd Diabetes-Risk-ML-App

## ⚙️ 2. Set Up Virtual Environment (.venv)

It’s best practice to isolate dependencies for each project.

Create .venv (Windows PowerShell)
python -m venv .venv

Activate it
.venv\Scripts\activate


You’ll see (.venv) appear in the terminal — meaning you’re now inside your project’s virtual environment.

Deactivate when done
deactivate

📦 3. Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt


📁 This installs all necessary libraries inside your .venv, not globally.
A hidden folder called __pycache__ will also appear automatically when Python compiles files — this is normal.

🚀 4. Train the ML Model

Before running the app, train the model once to generate model.pkl:

python train.py


This script:

Loads the Scikit-Learn Diabetes dataset

Trains a regression/classification model

Saves the trained model to model.pkl

You’ll see a success message like:

✅ Model saved to model.pkl

🌐 5. Run the FastAPI Web Application

Start the web server:

uvicorn app:app --reload


You’ll see something like:

INFO:     Uvicorn running on http://127.0.0.1:8000


Now open your browser:

🏠 Home Page: http://127.0.0.1:8000

🩸 API Docs: http://127.0.0.1:8000/docs

🧠 Model Info: http://127.0.0.1:8000/model/info

Press CTRL + C to stop the server.

🖼️ 6. Project Structure
Diabetes-Risk-ML-App/
│
├── app.py               # FastAPI app entry point
├── train.py             # Model training script
├── model.pkl            # Trained model file
├── requirements.txt     # Dependencies
├── README.md            # Project documentation
├── .gitignore           # Ignore cache, venv, etc.
│
├── static/              # CSS & static assets
│   └── style.css
│
└── templates/           # Frontend pages
    ├── base.html
    ├── home.html
    ├── form.html
    └── result.html

⚡ Faster Dependency Setup (optional)

You can speed up installation using a local pip cache:

mkdir C:\pip_cache
pip install -r requirements.txt --cache-dir C:\pip_cache --trusted-host pypi.org --trusted-host files.pythonhosted.org --timeout 120


Next time you install, pip will reuse cached packages — no need to download again.

📸 7. Preview