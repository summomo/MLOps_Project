# MLOps Final Project

## Registry-first (Plan A) Guide

### 1) Install dependencies

```powershell
pip install -r requirements.txt
```

### 2) Set environment variables (Windows PowerShell)

> Do not copy model files from the external project into this repository; inject absolute paths via environment variables.

```powershell
$env:MODEL_CKPT_PATH="C:\absolute\path\to\best.pt"
$env:TOKENIZER_PATH="C:\absolute\path\to\spm_joint_fr2en_v3.model"
$env:MODEL_NAME="fr2en-translator"
$env:MODEL_STAGE="Staging"
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
```

`MODEL_STAGE` must use MLflow stage enums with exact case: `Staging`, `Production`, `Archived`, `None`.
For production, strictly set: `MODEL_STAGE="Production"` (to avoid loading a non-production stage).

If authentication is required, set the additional environment variables expected by your MLflow server (for example username, password, or token).

### 3) Package and register to MLflow Registry

```powershell
python ml/src/export_mlflow_model.py
```

The script will:
- Read `MODEL_CKPT_PATH` and `TOKENIZER_PATH`
- Package checkpoint + tokenizer as artifacts into `mlflow.pyfunc`
- Log params: `git_commit`, `dvc_data_rev`, `ckpt_source_path`, `tokenizer_source_path`
- Register under `MODEL_NAME` (default `fr2en-translator`)
- Attempt automatic transition to `Staging`; if permissions are missing, it prints manual commands

### 4) Start API (loads only from Registry stage)

```powershell
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
```

At startup, the service executes:

`mlflow.pyfunc.load_model("models:/{MODEL_NAME}/{MODEL_STAGE}")`

It does not load a local `best.pt` path via `torch.load`.

### 5) Call the endpoint

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/translate" -ContentType "application/json" -Body '{"text":"bonjour le monde"}'
```

Example response:

```json
{
  "translation": "...",
  "model_name": "fr2en-translator",
  "model_stage": "Staging",
  "model_uri": "models:/fr2en-translator/Staging"
}
```

### 6) Inference logic notes

No reusable seq2seq beam-search inference entrypoint was found in this repository, so `ml/src/seq2seq_mlflow_model.py` includes a minimal runnable inference path:
- It first tries available checkpoint interfaces such as `translate`, `predict`, or `forward`
- If the real model structure cannot be matched, it falls back to a minimal tokenizer encode/decode stub so the service remains runnable

⚠️ The current stub is primarily for validating the **MLOps workflow (packaging, registry, stage-based loading, online serving)** and does not represent real translation quality.

Replace `TranslatorPyfunc._translate_with_model` with your real greedy/beam decode logic later (file: `ml/src/seq2seq_mlflow_model.py`).
