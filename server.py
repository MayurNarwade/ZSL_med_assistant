from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import sys

# ----------------------------
# PATH FIX
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, SRC_DIR)

# ----------------------------
# Import Symptom Pipeline
# ----------------------------
try:
    from pipeline import infer
except ImportError:
    from src.pipeline import infer


# ----------------------------
# Lazy load disease model
# ----------------------------
def load_disease_model():
    try:
        from pipeline_disease import predict_disease
    except ImportError:
        from src.pipeline_disease import predict_disease
    return predict_disease


app = FastAPI(title="AI Health Assistant API")

# ----------------------------
# CORS
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# Request Models
# ----------------------------
class PredictRequest(BaseModel):
    symptoms: str


class DiseaseRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {"message": "AI Health Assistant Backend Running"}


# ------------------------------------------------------
# MAIN ENDPOINT — MERGED PREDICTION (SYMBOLS + DISEASES)
# ------------------------------------------------------
@app.post("/predict")
def predict(req: PredictRequest):
    symptoms = req.symptoms.strip()
    if not symptoms:
        return {"error": "Symptoms required"}

    # -------------------------
    # 1) Symptom predictions
    # -------------------------
    base = infer(symptoms)

    # Always force fields (safety)
    base.setdefault("predictions", [])
    base.setdefault("triage", "low")
    base.setdefault("normalized_text", symptoms)

    # -------------------------
    # 2) Disease predictions
    # -------------------------
    diseases = []
    try:
        predict_disease = load_disease_model()
        dis_out = predict_disease(symptoms)

        # disease_out has format: {"predictions":[ ... ]}
        diseases_raw = dis_out.get("predictions", [])

        # normalize structure
        for d in diseases_raw:
            diseases.append({
                "label": d.get("label", ""),
                "score": d.get("score", 0),
                "description": d.get("desc", d.get("description", "")),
                "specialists": d.get("specialists", []),
                "recommended_tests": d.get("recommended_tests", [])
            })

    except Exception as e:
        diseases = []
        base["disease_error"] = str(e)

    # -------------------------
    # 3) REMOVE DUPLICATE labels
    # -------------------------
    symptom_labels = set(p.get("label", "").lower() for p in base["predictions"])
    diseases = [d for d in diseases if d.get("label", "").lower() not in symptom_labels]

    # -------------------------
    # 4) Attach cleaned disease list
    # -------------------------
    base["diseases"] = diseases

    # -------------------------
    # 5) Final clean response
    # -------------------------
    base.setdefault("disclaimer", "Informational only; not a medical diagnosis.")

    return base


# ---------------------------------------------
# OPTIONAL direct disease endpoint
# ---------------------------------------------
@app.post("/predict_disease")
def predict_disease_endpoint(req: DiseaseRequest):
    txt = req.text.strip()
    if not txt:
        return {"error": "Text required"}

    predict_disease = load_disease_model()
    return predict_disease(txt)


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
