import os
from dotenv import load_dotenv

load_dotenv()

os.environ["MODEL_VALIDATION_DISABLED"] = "True"

import cv2
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from inference import get_model


# -----------------------------
# Configuration
# -----------------------------

API_KEY = os.getenv("ROBOFLOW_API_KEY")
MODEL_ID = "road-damage-detection-i40w8/3"


# -----------------------------
# FastAPI
# -----------------------------

app = FastAPI(title="Road Damage Detection API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Model
# -----------------------------

model = None


def get_roboflow_model():

    global model

    if model is None:

        print("Loading Roboflow model...")

        if not API_KEY:
            raise RuntimeError(
                "ROBOFLOW_API_KEY is not configured in the environment."
            )

        model = get_model(
            model_id=MODEL_ID,
            api_key=API_KEY
        )

        print("Roboflow model loaded successfully.")

    return model


# -----------------------------
# Health check
# -----------------------------

@app.get("/health")
async def health():

    return {
        "status": "ok",
        "message": "Road Damage Detection API is running"
    }


# -----------------------------
# Homepage
# -----------------------------

@app.get("/")
async def serve_index():

    index_path = os.path.join(
        os.path.dirname(__file__),
        "index.html"
    )

    if os.path.exists(index_path):

        return FileResponse(index_path)

    return {
        "message": "Road Damage Detection API is running."
    }


# -----------------------------
# Prediction
# -----------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        # Load model only when prediction is requested
        detection_model = get_roboflow_model()

        # Read uploaded image
        contents = await file.read()

        npimg = np.frombuffer(
            contents,
            np.uint8
        )

        image = cv2.imdecode(
            npimg,
            cv2.IMREAD_COLOR
        )

        if image is None:

            return {
                "detections": []
            }

        # Run Roboflow inference
        results = detection_model.infer(
            image,
            confidence=0.4
        )[0]

        detections = []

        for pred in results.predictions:

            x1 = pred.x - pred.width / 2
            y1 = pred.y - pred.height / 2

            x2 = pred.x + pred.width / 2
            y2 = pred.y + pred.height / 2

            detections.append({
                "bbox": [
                    x1,
                    y1,
                    x2,
                    y2
                ],
                "class": pred.class_name,
                "confidence": float(pred.confidence)
            })

        return {
            "detections": detections
        }

    except Exception as e:

        print("Prediction error:", str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
