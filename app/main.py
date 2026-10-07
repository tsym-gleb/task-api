from contextlib import asynccontextmanager
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd
from fastapi import FastAPI, Request

from app.schemas.predict import PredictRequest, PredictResponse


MODEL_PATH = Path("models/wine_model.pkl")
CLASS_NAMES = ["class_0", "class_1", "class_2"]
FEATURE_ORDER = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols",
    "proanthocyanins", "color_intensity", "hue",
    "od280/od315_of_diluted_wines", "proline",
]


def _build_gradio_demo(model) -> gr.Interface:
    def predict_wine(*features):
        row = pd.DataFrame([dict(zip(FEATURE_ORDER, features))])[FEATURE_ORDER]
        proba = model.predict_proba(row)[0]
        return {CLASS_NAMES[i]: float(p) for i, p in enumerate(proba)}

    return gr.Interface(
        fn=predict_wine,
        inputs=[gr.Number(label=name, value=v) for name, v in zip(
            FEATURE_ORDER,
            [13.5, 1.8, 2.4, 18.0, 105.0, 2.7, 2.9, 0.28, 1.85, 5.5, 1.05, 3.2, 1180.0],
        )],
        outputs=gr.Label(label="Распределение вероятностей"),
        title="🍷 Wine classifier",
        description="Введите 13 показателей лабораторного анализа партии",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request) -> PredictResponse:
    model = request.app.state.model
    row = pd.DataFrame([req.model_dump(by_alias=True)])[FEATURE_ORDER]
    proba = model.predict_proba(row)[0]
    pred_idx = int(proba.argmax())
    return PredictResponse(
        predicted_class=CLASS_NAMES[pred_idx],
        probabilities={CLASS_NAMES[i]: float(round(p, 4)) for i, p in enumerate(proba)},
    )


_model = joblib.load(MODEL_PATH)
demo = _build_gradio_demo(_model)
app = gr.mount_gradio_app(app, demo, path="/")