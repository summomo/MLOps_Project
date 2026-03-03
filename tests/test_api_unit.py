from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.main import create_app


class DummyModel:
    def predict(self, model_input):
        return ["dummy"]


def test_translate_empty_text_returns_400():
    app = create_app(load_model_func=lambda *_: DummyModel())
    with TestClient(app) as client:
        response = client.post("/translate", json={"text": "   "})

        assert response.status_code == 400
