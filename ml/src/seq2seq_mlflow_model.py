from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import pandas as pd
import sentencepiece as spm


class TranslatorPyfunc(mlflow.pyfunc.PythonModel):
    def __init__(self) -> None:
        self.model: Any = None
        self.tokenizer: spm.SentencePieceProcessor | None = None
        self.device = "cpu"

    def _is_truthy_env(self, name: str) -> bool:
        value = os.getenv(name, "").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _translate_mode(self) -> str:
        mode = os.getenv("TRANSLATE_MODE", "full").strip().lower()
        if self._is_truthy_env("TOKENIZER_ONLY_MODE"):
            return "stub"
        if mode not in {"full", "stub"}:
            return "full"
        return mode

    def _resolve_artifact_path(self, raw_path: str) -> Path:
        normalized_path = Path(raw_path.replace("\\", "/"))
        if normalized_path.exists():
            return normalized_path

        direct_path = Path(raw_path)
        if direct_path.exists():
            return direct_path

        filename = normalized_path.name
        search_roots = [normalized_path.parent, direct_path.parent]

        for root in search_roots:
            if not root or not root.exists():
                continue
            candidate = root / filename
            if candidate.exists():
                return candidate
            artifacts_candidate = root / "artifacts" / filename
            if artifacts_candidate.exists():
                return artifacts_candidate

        raise FileNotFoundError(f"Artifact not found: {raw_path} (normalized={normalized_path})")

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        ckpt_path = self._resolve_artifact_path(context.artifacts["checkpoint"])
        tokenizer_path = self._resolve_artifact_path(context.artifacts["tokenizer"])

        if tokenizer_path.is_dir():
            tokenizer_candidates = list(tokenizer_path.glob("*.model"))
            if not tokenizer_candidates:
                raise FileNotFoundError("No tokenizer .model found in artifact directory")
            tokenizer_path = tokenizer_candidates[0]

        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path.as_posix())

        if self._translate_mode() == "stub":
            self.model = None
            return

        loaded_checkpoint: Any = None
        import torch

        try:
            loaded_checkpoint = torch.load(ckpt_path, map_location=self.device)
        except Exception:
            loaded_checkpoint = torch.jit.load(str(ckpt_path), map_location=self.device)

        if isinstance(loaded_checkpoint, torch.nn.Module):
            self.model = loaded_checkpoint
        elif isinstance(loaded_checkpoint, dict) and isinstance(loaded_checkpoint.get("model"), torch.nn.Module):
            self.model = loaded_checkpoint["model"]
        else:
            self.model = loaded_checkpoint

        if hasattr(self.model, "eval"):
            self.model.eval()

    def _decode_tokens(self, token_ids: list[int], fallback_text: str) -> str:
        if not token_ids or self.tokenizer is None:
            return fallback_text

        safe_ids = [token_id for token_id in token_ids if isinstance(token_id, int) and token_id >= 0]
        if not safe_ids:
            return fallback_text

        try:
            decoded = self.tokenizer.decode(safe_ids)
            return decoded if decoded else fallback_text
        except Exception:
            return fallback_text

    def _translate_with_model(self, text: str, token_ids: list[int]) -> str | None:
        if self.model is None:
            return None

        import torch

        try:
            if hasattr(self.model, "translate"):
                output = self.model.translate(text)
                if isinstance(output, str):
                    return output
                if isinstance(output, (list, tuple)):
                    return self._decode_tokens(list(output), text)

            if hasattr(self.model, "predict"):
                output = self.model.predict([text])
                if isinstance(output, (list, tuple)) and output:
                    first = output[0]
                    if isinstance(first, str):
                        return first
                    if isinstance(first, (list, tuple)):
                        return self._decode_tokens(list(first), text)

            if callable(self.model):
                input_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)
                output = self.model(input_tensor)
                if isinstance(output, torch.Tensor):
                    if output.ndim == 3:
                        pred_ids = output.argmax(dim=-1).squeeze(0).tolist()
                        return self._decode_tokens(pred_ids, text)
                    if output.ndim == 2:
                        pred_ids = output.argmax(dim=-1).tolist()
                        return self._decode_tokens(pred_ids, text)
        except Exception:
            return None

        return None

    def _translate_single(self, text: str) -> str:
        if self.tokenizer is None:
            return text

        token_ids = self.tokenizer.encode(text, out_type=int)
        translated = self._translate_with_model(text, token_ids)
        if translated:
            return translated

        return self._decode_tokens(token_ids, text)

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame) -> list[str]:
        if "text" not in model_input.columns:
            raise ValueError("Input DataFrame must contain a 'text' column")

        outputs: list[str] = []
        for raw_text in model_input["text"].tolist():
            text = "" if raw_text is None else str(raw_text)
            outputs.append(self._translate_single(text))
        return outputs
