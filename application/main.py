"""FastAPI backend serving the fruit quality/ripeness/defect CNN classifier."""

import io
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

MODEL_PATH = Path(__file__).parent / "model" / "fruit_quality_model.keras"
IMG_SIZE = (160, 160)

# Must match the merged class order used in the training notebook.
CLASS_NAMES = [
    "Bad_Apple", "Bad_Banana", "Bad_Guava", "Bad_Lime", "Bad_Orange", "Bad_Pomegranate",
    "Good_Apple", "Good_Banana", "Good_Guava", "Good_Lime", "Good_Orange", "Good_Pomegranate",
]

model_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. "
            "Download fruit_quality_model.keras from Colab/Drive and place it there."
        )
    model_state["model"] = tf.keras.models.load_model(MODEL_PATH)
    yield
    model_state.clear()


app = FastAPI(title="Fruit Quality Classifier", lifespan=lifespan)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    array = np.asarray(image, dtype=np.float32)
    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
    return np.expand_dims(array, axis=0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    image_bytes = await file.read()
    try:
        input_array = preprocess_image(image_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")

    probs = model_state["model"].predict(input_array, verbose=0)[0]
    pred_id = int(np.argmax(probs))

    return {
        "predicted_class": CLASS_NAMES[pred_id],
        "confidence": float(probs[pred_id]),
        "all_probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, probs)},
    }
