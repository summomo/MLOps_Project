# MLOps Final Project

## Registry-first (Plan A) Usage Guide

### 1) Install dependencies

```powershell
pip install -r requirements.txt
```

### 2) Set environment variables (Windows PowerShell)

> Important: do not copy model files from the external project into this repository; provide absolute paths via environment variables.

```powershell
$env:MODEL_CKPT_PATH="C:\absolute\path\to\best.pt"
$env:TOKENIZER_PATH="C:\absolute\path\to\spm_joint_fr2en_v3.model"
$env:MODEL_NAME="fr2en-translator"
$env:MODEL_STAGE="Staging"
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
$env:HF_MODEL_ID="Helsinki-NLP/opus-mt-fr-en"  # optional: enables HF export mode
```

`MODEL_STAGE` must use MLflow fixed enum values with exact case: `Staging`, `Production`, `Archived`, `None`.
For production environments, strictly use: `MODEL_STAGE="Production"` (to avoid loading a non-production stage by mistake).

If authentication is required, set the additional environment variables expected by your MLflow server (for example username/password or token).

### 3) Package and register to MLflow Registry

```powershell
python ml/src/export_mlflow_model.py
```

The script will:
- If `HF_MODEL_ID` is set: export a Hugging Face Marian FR->EN model (`Helsinki-NLP/opus-mt-fr-en`) via `ml/src/hf_marian_mlflow_model.py`
- Otherwise: read `MODEL_CKPT_PATH` and `TOKENIZER_PATH` and export the legacy checkpoint wrapper
- Log params: `git_commit`, `dvc_data_rev`, plus mode-specific source params (`hf_model_id` or checkpoint/tokenizer paths)
- Register under `MODEL_NAME` (default: `fr2en-translator`)
- Try to transition automatically to `Staging`; if permissions are insufficient, it prints manual commands

HF mode also logs `hf_model_id` and stores downloaded HF files as MLflow artifacts for reproducible serving.

### 4) Start the API (loads only from Registry stage)

```powershell
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
```

At startup, the service executes:

`mlflow.pyfunc.load_model("models:/{MODEL_NAME}/{MODEL_STAGE}")`

No local `torch.load(best.pt)` path-based loading is used.

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

No reusable seq2seq beam-search inference entrypoint was found in the current repository, so `ml/src/seq2seq_mlflow_model.py` implements a minimal runnable inference path:
- First it tries available checkpoint methods such as `translate` / `predict` / `forward`
- If the real model structure cannot be matched, it falls back to a minimal tokenizer encode/decode stub to keep the service runnable

⚠️ The current stub is mainly for validating the **MLOps workflow (packaging, registration, stage-based loading, and online serving)**, and does not represent real translation quality.

Later, replace `TranslatorPyfunc._translate_with_model` with the real greedy/beam decode logic (file: `ml/src/seq2seq_mlflow_model.py`).

CI check registration

## 7) Render deployment and promotion gates

### Render services (Blueprint)

This repository includes `render.yaml` to create two web services from the same codebase:

- `mlops-api-staging` tracking branch `staging`
- `mlops-api-prod` tracking branch `main`

Both services start with:

`uvicorn apps.api.main:app --host 0.0.0.0 --port $PORT`

Health check path is:

`/health`

### Required Render environment variables

For both services:

- `MODEL_NAME=fr2en-translator`
- `MLFLOW_TRACKING_URI` (set in Render dashboard)
- Optional auth vars if needed by your tracking backend:
  - `MLFLOW_TRACKING_USERNAME`
  - `MLFLOW_TRACKING_PASSWORD`

Stage-specific values:

- Staging service: `MODEL_STAGE=Staging`
- Production service: `MODEL_STAGE=Production`

`MODEL_STAGE` is case-sensitive and must match MLflow stage enums exactly.

### Gates for staging to main promotion

`ml/src/gates.py` runs smoke and latency gates against `STAGING_URL`:

- Sends 3 requests to `POST {STAGING_URL}/translate`
- Smoke gate: each response must be HTTP 200 with a non-empty `translation`
- Latency gate: p95 latency must be within threshold (default `1.0s`)

When successful, it prints `GATES PASSED` and exits with code `0`; otherwise exits with code `1`.

### GitHub Actions gate on main PRs

`/.github/workflows/gates-on-main.yml` runs on pull requests targeting `main` and executes:

`python ml/src/gates.py`

Create this GitHub secret before using the workflow:

- Repository -> Settings -> Secrets and variables -> Actions -> New repository secret
- Name: `STAGING_URL`
- Value: your staging service base URL, for example `https://mlops-api-staging.onrender.com`

Use the resulting `gates` workflow status check in branch rulesets so that merges to `main` require successful gates.

## 8) Local validation (HF model on Registry-first flow)

1. Configure MLflow tracking credentials via environment variables (do not hardcode tokens).
2. Set:

```powershell
$env:HF_MODEL_ID="Helsinki-NLP/opus-mt-fr-en"
```

3. Register model to registry stage:

```powershell
python ml/src/export_mlflow_model.py
```

4. Ensure Render (or local API) uses only:

- `MODEL_NAME=fr2en-translator`
- `MODEL_STAGE=Staging` (or `Production`)

5. Verify:

- `GET /health` returns `200`
- `POST /translate` with `{"text":"bonjour"}` returns non-empty English output
