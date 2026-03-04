from __future__ import annotations

import os
from pathlib import Path

import mlflow.pyfunc
import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer


class HFMarianTranslatorPyfunc(mlflow.pyfunc.PythonModel):
    def __init__(self) -> None:
        self.tokenizer: MarianTokenizer | None = None
        self.model: MarianMTModel | None = None
        self.device = "cpu"

    def _resolve_local_model_dir(self, context: mlflow.pyfunc.PythonModelContext) -> Path | None:
        model_dir = context.artifacts.get("hf_model_dir") if context.artifacts else None
        if not model_dir:
            return None
        resolved = Path(model_dir)
        if resolved.exists() and resolved.is_dir():
            return resolved
        return None

    def _resolve_model_source(self, context: mlflow.pyfunc.PythonModelContext) -> str | Path:
        local_dir = self._resolve_local_model_dir(context)
        if local_dir is not None:
            return local_dir

        model_id = os.getenv("HF_MODEL_ID", "").strip()
        if model_id:
            return model_id

        return "Helsinki-NLP/opus-mt-fr-en"

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        torch.set_num_threads(1)
        source = self._resolve_model_source(context)

        self.tokenizer = MarianTokenizer.from_pretrained(source)
        self.model = MarianMTModel.from_pretrained(source)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame) -> list[str]:
        if "text" not in model_input.columns:
            raise ValueError("Input DataFrame must contain a 'text' column")

        if self.tokenizer is None or self.model is None:
            raise RuntimeError("HF Marian model is not loaded")

        texts = ["" if value is None else str(value) for value in model_input["text"].tolist()]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=64)

        return self.tokenizer.batch_decode(output, skip_special_tokens=True)
