from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.main import create_app

class MockTranslatorModel:
    def predict(self, model_input):
        text = str(model_input.iloc[0]["text"])
        return [f"translated:{text}"]


def test_translate_returns_translation_field(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "fr2en-translator")
    monkeypatch.setenv("MODEL_STAGE", "Staging")

    loaded = {}

    def fake_loader(model_uri: str, tracking_uri: str | None):
        loaded["model_uri"] = model_uri
        loaded["tracking_uri"] = tracking_uri
        return MockTranslatorModel()

    app = create_app(load_model_func=fake_loader)
    with TestClient(app) as client:
        response = client.post("/translate", json={"text": "bonjour"})

        assert response.status_code == 200
        body = response.json()
        assert body["translation"] == "translated:bonjour"
        assert body["model_name"] == "fr2en-translator"
        assert body["model_stage"] == "Staging"
        assert body["model_uri"] == "models:/fr2en-translator/Staging"

    assert loaded["model_uri"] == "models:/fr2en-translator/Staging"
