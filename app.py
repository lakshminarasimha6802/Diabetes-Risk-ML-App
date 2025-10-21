from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.responses import HTMLResponse

from pydantic import BaseModel, Field
import numpy as np
import joblib
import os

# --------- Load model once at startup ---------
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
ARTIFACT = joblib.load(MODEL_PATH)
MODEL = ARTIFACT["model"]
META = ARTIFACT["meta"]

# --------- FastAPI & templating setup ---------
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
app = FastAPI(
    title="Diabetes Risk ML App",
    version="1.0.0",
    description="Educational demo: UI + API for diabetes risk prediction."
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --------- API schema (kept for JSON clients) ---------
class DiabetesFeatures(BaseModel):
    age: float = Field(..., description="Normalized age")
    sex: float = Field(..., description="Normalized sex")
    bmi: float = Field(..., description="Body mass index (normalized)")
    bp: float = Field(..., description="Mean blood pressure (normalized)")
    s1: float = Field(..., description="TC (normalized)")
    s2: float = Field(..., description="LDL (normalized)")
    s3: float = Field(..., description="HDL (normalized)")
    s4: float = Field(..., description="TCH (normalized)")
    s5: float = Field(..., description="LTG (normalized)")
    s6: float = Field(..., description="GLU (normalized)")

# --------- JSON API endpoints ---------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/model/info")
def model_info():
    return META

@app.post("/predict")
def predict(item: DiabetesFeatures):
    try:
        order = ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"]
        X = np.array([[getattr(item, f) for f in order]], dtype=float)
        prob_high = float(MODEL.predict_proba(X)[0][1])
        label = "high_risk" if prob_high >= 0.5 else "low_risk"
        return {
            "label": label,
            "probabilities": {
                "low_risk": round(1.0 - prob_high, 4),
                "high_risk": round(prob_high, 4)
            },
            "meta": {"classes": META["classes"], "threshold_note": "Binary labels from 70th percentile target"}
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --------- UI pages ---------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/start", response_class=HTMLResponse)
def start(request: Request, name: str = Form(...), age_display: int = Form(...)):
    return templates.TemplateResponse(
        "form.html",
        {"request": request, "name": name, "age_display": age_display}
    )

@app.post("/predict-ui", response_class=HTMLResponse)
def predict_ui(
    request: Request,
    name: str = Form(...),
    age_display: int = Form(...),
    age: float = Form(...),
    sex: float = Form(...),
    bmi: float = Form(...),
    bp: float = Form(...),
    s1: float = Form(...),
    s2: float = Form(...),
    s3: float = Form(...),
    s4: float = Form(...),
    s5: float = Form(...),
    s6: float = Form(...)
):
    try:
        order = ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"]
        values = [age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]
        X = np.array([values], dtype=float)
        prob_high = float(MODEL.predict_proba(X)[0][1])
        label = "High Risk" if prob_high >= 0.5 else "Low Risk"
        probs = {
            "Low Risk": round(1.0 - prob_high, 4),
            "High Risk": round(prob_high, 4)
        }
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "name": name,
                "age_display": age_display,
                "label": label,
                "probs": probs,
                "features": dict(zip(order, values))
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "name": name,
                "age_display": age_display,
                "label": "Error",
                "error": str(e),
                "probs": {},
                "features": {}
            },
            status_code=400
        )
