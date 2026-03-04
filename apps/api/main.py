from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from typing import Any, Callable

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict


ALLOWED_STAGES = {"Staging", "Production", "Archived", "None"}


class TranslateRequest(BaseModel):
    text: str


class TranslateResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    translation: str
    model_name: str
    model_stage: str
    model_uri: str


def _default_loader(model_uri: str, tracking_uri: str | None) -> Any:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    return mlflow.pyfunc.load_model(model_uri)


def _get_stage_from_env() -> str:
    stage = os.getenv("MODEL_STAGE", "Staging")
    if stage not in ALLOWED_STAGES:
        allowed = ", ".join(sorted(ALLOWED_STAGES))
        raise RuntimeError(
            f"Invalid MODEL_STAGE='{stage}'. Use exact MLflow stage name: {allowed}"
        )
    return stage


def _extract_translation(prediction: Any) -> str:
    if isinstance(prediction, list):
        if not prediction:
            return ""
        return str(prediction[0])

    if isinstance(prediction, pd.Series):
        if prediction.empty:
            return ""
        return str(prediction.iloc[0])

    if isinstance(prediction, pd.DataFrame):
        if prediction.empty:
            return ""
        return str(prediction.iloc[0, 0])

    return str(prediction)


def create_app(load_model_func: Callable[[str, str | None], Any] = _default_loader) -> FastAPI:
    model_load_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        app.state.model_name = os.getenv("MODEL_NAME", "fr2en-translator")
        app.state.model_stage = _get_stage_from_env()
        app.state.model_uri = f"models:/{app.state.model_name}/{app.state.model_stage}"
        app.state.model = None

        yield

    app = FastAPI(title="Registry-first Translation API", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str | bool]:
        return {"status": "ok", "model_loaded": app.state.model is not None}

    def _ensure_model_loaded() -> Any:
        if app.state.model is not None:
            return app.state.model

        with model_load_lock:
            if app.state.model is not None:
                return app.state.model
            app.state.model = load_model_func(app.state.model_uri, app.state.tracking_uri)

        return app.state.model

    @app.post("/translate", response_model=TranslateResponse)
    def translate(payload: TranslateRequest) -> TranslateResponse:
        if not payload.text or not payload.text.strip():
            raise HTTPException(status_code=400, detail="text must not be empty")

        try:
            model = _ensure_model_loaded()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"model load failed: {exc}") from exc

        model_input = pd.DataFrame({"text": [payload.text]})
        prediction = model.predict(model_input)
        translation = _extract_translation(prediction)

        return TranslateResponse(
            translation=translation,
            model_name=app.state.model_name,
            model_stage=app.state.model_stage,
            model_uri=app.state.model_uri,
        )

    return app


app = create_app()
