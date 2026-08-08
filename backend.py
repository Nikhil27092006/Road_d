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

API_KEY = os.getenv("ROBOFLOW_API_KEY")
MODEL_ID = "road-damage-detection-i40w8/3"   # replace with YOUR verified model ID


app = FastAPI(title="Road Damage Detection API")

# Allow the frontend (opened as a file or on any localhost port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading model...")
model = get_model(model_id=MODEL_ID, api_key=API_KEY)
print("Model loaded.")


@app.get("/")
async def serve_index():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Road Damage Detection API is running. Frontend index.html not found."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if image is None:
            return {"detections": []}

        results = model.infer(image, confidence=0.4)[0]

        detections = []
        for pred in results.predictions:
            # Roboflow returns center x,y + width,height -> convert to x1,y1,x2,y2
            x1 = pred.x - pred.width / 2
            y1 = pred.y - pred.height / 2
            x2 = pred.x + pred.width / 2
            y2 = pred.y + pred.height / 2

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "class": pred.class_name,
                "confidence": float(pred.confidence)
            })

        return {"detections": detections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))