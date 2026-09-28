# Fruit Quality Classifier — Application

FastAPI backend + Streamlit UI serving the MobileNetV2 model trained in `fruit_quality_classifier.ipynb`.

### Dataset

The model was trained using the **Fruit Quality Classification and Detection** dataset from Kaggle.

[Download Dataset](https://www.kaggle.com/datasets/abrars2/fruit-quality-classificaltion-and-detection/data)

## Setup

1. Trained model `fruit_quality_model.keras` is placed at:
   ```
   application/model/fruit_quality_model.keras
   ```
2. Dependencies are already installed via `uv` in `.venv`. If you need to reinstall:
   ```
   uv sync
   ```

## Run

Start the API (from the `application/` directory):
```
uv run uvicorn main:app --reload
```

In a separate terminal, start the UI:
```
uv run streamlit run app.py
```

Streamlit will open in your browser; upload a fruit photo and click "Classify". The UI calls the FastAPI backend at `http://127.0.0.1:8000`, so the backend must be running first.
