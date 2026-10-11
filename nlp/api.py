"""SentinelAI backend: serves the toxicity model to edusafe_toxicity_detector.html

Run from the folder that contains this file:
    python -m uvicorn api:app --port 8000
    or
    python main.py
"""
import os
import shutil
from pathlib import Path

import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "custom_toxicity_model.keras"
ZIP_PATH = BASE / "custom_toxicity_model.keras.zip"
HTML_PATH = BASE / "edusafe_toxicity_detector.html"
LABELS = ["hate_speech", "offensive_language", "neither"]
THRESHOLD = 50.0

# A .keras file is already a zip archive. If only the ".keras.zip" copy exists,
# use it under the correct name instead of unzipping it.
if not MODEL_PATH.exists() and ZIP_PATH.exists():
    shutil.copy(ZIP_PATH, MODEL_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found. Put custom_toxicity_model.keras (or the .keras.zip) in {BASE}"
    )

model = tf.keras.models.load_model(MODEL_PATH)

app = FastAPI(title="SentinelAI Toxicity API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Post(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
def serve_home():
    if HTML_PATH.exists():
        return FileResponse(HTML_PATH)
    return HTMLResponse("<h1>SentinelAI Moderator</h1><p>Frontend file edusafe_toxicity_detector.html not found.</p>")


@app.get("/health")
def health():
    return {
        "status": "online",
        "model_loaded": model is not None,
        "labels": LABELS
    }


@app.post("/analyze")
def analyze(post: Post):
    text = post.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty")

    scores = model.predict(tf.constant([text]), verbose=0)[0]
    by_label = {name: round(float(s) * 100.0, 1) for name, s in zip(LABELS, scores)}
    
    # Toxicity percentage is hate_speech + offensive_language (or 100 - neither)
    toxic_score = round(float(scores[0] + scores[1]) * 100.0, 1)
    is_toxic = toxic_score >= THRESHOLD

    action = (
        "Post blocked and submitted to the school administration for manual review."
        if is_toxic
        else "Post approved automatically and published to the educational forum."
    )

    return {
        "text": text,
        "is_toxic": is_toxic,
        "confidence_score": toxic_score,
        "categories": by_label,
        "scores": {name: float(s) for name, s in zip(LABELS, scores)},
        "action": action,
        "details": {
            "neural_network_status": "Active (Bi-LSTM Softmax)",
            "model_path": str(MODEL_PATH.name)
        }
    }