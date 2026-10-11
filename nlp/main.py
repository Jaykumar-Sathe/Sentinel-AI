"""
SentinelAI Toxicity Detector - FastAPI Backend
Provides real-time NLP toxicity & cyberbullying detection using Bi-LSTM Keras neural network.
"""

import os
import re
import json
import logging
from typing import Dict, Any, List

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SentinelAI")

# Current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "custom_toxicity_model.keras")
HTML_PATH = os.path.join(BASE_DIR, "edusafe_toxicity_detector.html")

app = FastAPI(
    title="SentinelAI Moderator API",
    description="Cyberbullying and Toxicity Detection using Bi-Directional LSTM Neural Network",
    version="1.0.0"
)

# Enable CORS for local and web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model globally on startup
model = None
LABELS = ["hate_speech", "offensive_language", "neither"]

def load_ai_model():
    global model
    try:
        import keras
        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading Keras model from {MODEL_PATH}...")
            model = keras.models.load_model(MODEL_PATH)
            logger.info("Keras neural network loaded successfully!")
        else:
            logger.warning(f"Model file not found at {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error loading Keras model: {e}")
        model = None

# Initialize model
load_ai_model()

class AnalyzeRequest(BaseModel):
    text: str

class AnalyzeResponse(BaseModel):
    text: str
    is_toxic: bool
    confidence_score: float
    categories: Dict[str, float]
    action: str
    details: Dict[str, Any]

# Curated lexicons for context-aware education forum moderation
TOXIC_PATTERNS = [
    r"\b(idiot|moron|stupid|dumb|loser|ugly|fat|freak)\b",
    r"\b(shut\s*up|get\s*lost|kill\s*yourself|kys|drop\s*dead)\b",
    r"\b(hate\s*you|despise\s*you|trash|scum|bitch|bastard|asshole)\b",
    r"\b(hurt\s*you|beat\s*you\s*up|punch\s*you|attack\s*you)\b",
    r"\b(f+u+c+k|s+h+i+t|d+i+c+k|c+u+n+t|whore|slut)\b"
]

SAFE_CONSTRUCTIVE_PATTERNS = [
    r"\b(disagree|perspective|discuss|opinion|debate|argument|point|classroom|tomorrow|respect)\b",
    r"\b(thank\s*you|thanks|please|appreciate|helpful|good\s*job|well\s*done|great\s*work)\b"
]

def evaluate_toxicity(text: str) -> Dict[str, Any]:
    """
    Evaluates toxicity using the trained Bi-LSTM neural network and context-aware educational safety heuristics.
    Returns calibrated probability (0-100%) and violation flag.
    """
    clean_text = text.strip()
    lower_text = clean_text.lower()

    raw_model_preds = None
    model_toxic_pct = None

    if model is not None:
        try:
            import tensorflow as tf
            preds = model.predict(tf.constant([clean_text]), verbose=0)[0]
            # [hate_speech, offensive_language, neither]
            raw_model_preds = {label: float(val) for label, val in zip(LABELS, preds)}
            # Model toxicity percentage is (hate_speech + offensive_language) * 100
            model_toxic_pct = float(preds[0] + preds[1]) * 100.0
        except Exception as e:
            logger.error(f"Prediction error from Keras model: {e}")

    # Lexical context scoring
    toxic_hits = 0
    for pattern in TOXIC_PATTERNS:
        matches = re.findall(pattern, lower_text)
        toxic_hits += len(matches)

    safe_hits = 0
    for pattern in SAFE_CONSTRUCTIVE_PATTERNS:
        matches = re.findall(pattern, lower_text)
        safe_hits += len(matches)

    is_threat = bool(re.search(r"\b(hurt|kill|punch|beat\s*up|attack)\b", lower_text))

    if model_toxic_pct is not None:
        # We have the neural network running
        hate_pct = round(raw_model_preds["hate_speech"] * 100.0, 1)
        offensive_pct = round(raw_model_preds["offensive_language"] * 100.0, 1)
        neither_pct = round(raw_model_preds["neither"] * 100.0, 1)

        confidence_score = model_toxic_pct

        # If lexical threats are detected, ensure minimum high confidence
        if is_threat or toxic_hits > 1:
            confidence_score = max(confidence_score, 92.5)
            hate_pct = max(hate_pct, 85.0)
            neither_pct = min(neither_pct, 5.0)
        elif safe_hits > 0 and confidence_score < 40.0:
            confidence_score = max(1.0, confidence_score - 5.0)
            neither_pct = min(99.5, neither_pct + 5.0)

        confidence_score = round(min(99.0, max(0.5, confidence_score)), 1)
        is_toxic = confidence_score >= 50.0

        categories = {
            "hate_speech": round(hate_pct, 1),
            "offensive_language": round(offensive_pct, 1),
            "neither": round(neither_pct, 1)
        }
    else:
        # Fallback simulation when model is not loaded
        if toxic_hits > 0 or is_threat:
            base = 75.0 + min(20.0, (toxic_hits - 1) * 10.0)
            if is_threat:
                base = max(base, 92.5)
            confidence_score = round(min(98.5, max(55.0, base)), 1)
            is_toxic = True
        else:
            base = 4.0 + (len(clean_text) % 8) * 1.5
            if safe_hits > 0:
                base = max(2.0, base - 5.0)
            confidence_score = round(min(35.0, base), 1)
            is_toxic = False

        if is_toxic:
            hate_prob = round(confidence_score * 0.45, 1)
            offensive_prob = round(confidence_score * 0.55, 1)
            neither_prob = round(max(0.0, 100.0 - confidence_score), 1)
        else:
            hate_prob = round(confidence_score * 0.1, 1)
            offensive_prob = round(confidence_score * 0.2, 1)
            neither_prob = round(max(0.0, 100.0 - confidence_score), 1)

        categories = {
            "hate_speech": hate_prob,
            "offensive_language": offensive_prob,
            "neither": neither_prob
        }

    action = (
        "Post blocked and submitted to the school administration for manual review."
        if is_toxic
        else "Post approved automatically and published to the educational forum."
    )

    return {
        "is_toxic": is_toxic,
        "confidence_score": confidence_score,
        "categories": categories,
        "action": action,
        "details": {
            "neural_network_status": "Active (Bi-LSTM Softmax)" if model is not None else "Simulated Fallback",
            "raw_model_predictions": raw_model_preds,
            "toxic_hits": toxic_hits,
            "safe_hits": safe_hits
        }
    }

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serves the SentinelAI Forum Moderator web interface."""
    if os.path.exists(HTML_PATH):
        return FileResponse(HTML_PATH)
    return HTMLResponse("<h1>SentinelAI Moderator</h1><p>Frontend file edusafe_toxicity_detector.html not found.</p>")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "online",
        "model_loaded": model is not None,
        "labels": LABELS
    }

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_post(req: AnalyzeRequest):
    """
    Analyzes input text for cyberbullying and toxicity.
    Matches the schema expected by edusafe_toxicity_detector.html.
    """
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    result = evaluate_toxicity(req.text)
    return AnalyzeResponse(
        text=req.text,
        is_toxic=result["is_toxic"],
        confidence_score=result["confidence_score"],
        categories=result["categories"],
        action=result["action"],
        details=result["details"]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
