from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import time
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from ml.src.hf_marian_mlflow_model import HFMarianTranslatorPyfunc
from ml.src.seq2seq_mlflow_model import TranslatorPyfunc


def _get_required_path(env_name: str) -> Path:
    raw_value = os.getenv(env_name)
    if not raw_value:
        raise ValueError(f"Missing required environment variable: {env_name}")
    path = Path(raw_value).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{env_name} does not point to an existing file: {path}")
    return path


def _git_commit_sha(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        sha = result.stdout.strip()
        return sha or "unknown"
    except Exception:
        return "unknown"


def _dvc_data_rev(repo_root: Path) -> str:
    dvc_dir = repo_root / ".dvc"
    dvc_lock = repo_root / "dvc.lock"
    dvc_yaml = repo_root / "dvc.yaml"

    if not dvc_dir.exists() and not dvc_lock.exists() and not dvc_yaml.exists():
        return "not_available"

    source_file = dvc_lock if dvc_lock.exists() else dvc_yaml
    if source_file.exists():
        digest = hashlib.sha256(source_file.read_bytes()).hexdigest()
        return f"{source_file.name}:{digest[:16]}"

    return "dvc_present_but_revision_unknown"


def _find_registered_version(client: MlflowClient, model_name: str, run_id: str, retries: int = 10) -> str | None:
    for _ in range(retries):
        versions = client.search_model_versions(f"name='{model_name}'")
        matching = [version for version in versions if version.run_id == run_id]
        if matching:
            latest = max(matching, key=lambda item: int(item.version))
            return latest.version
        time.sleep(1)
    return None


def _normalize_stage(stage: str) -> str:
    normalized = stage.strip().lower()
    if normalized == "staging":
        return "Staging"
    if normalized == "production":
        return "Production"
    return stage


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    hf_model_id = os.getenv("HF_MODEL_ID", "").strip()
    model_name = os.getenv("MODEL_NAME", "fr2en-translator")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    git_commit = _git_commit_sha(repo_root)
    dvc_data_rev = _dvc_data_rev(repo_root)

    with mlflow.start_run() as run:
        mlflow.log_param("git_commit", git_commit)
        mlflow.log_param("dvc_data_rev", dvc_data_rev)

        if hf_model_id:
            mlflow.log_param("hf_model_id", hf_model_id)

            from transformers import MarianMTModel, MarianTokenizer

            with tempfile.TemporaryDirectory() as tmp_dir:
                local_model_dir = Path(tmp_dir) / "hf_model"
                local_model_dir.mkdir(parents=True, exist_ok=True)

                tokenizer = MarianTokenizer.from_pretrained(hf_model_id)
                model = MarianMTModel.from_pretrained(hf_model_id)
                tokenizer.save_pretrained(local_model_dir)
                model.save_pretrained(local_model_dir)

                model_info = mlflow.pyfunc.log_model(
                    artifact_path="seq2seq_translator",
                    python_model=HFMarianTranslatorPyfunc(),
                    artifacts={"hf_model_dir": local_model_dir.as_posix()},
                    registered_model_name=model_name,
                )
        else:
            checkpoint_path = _get_required_path("MODEL_CKPT_PATH")
            tokenizer_path = _get_required_path("TOKENIZER_PATH")

            mlflow.log_param("ckpt_source_path", str(checkpoint_path))
            mlflow.log_param("tokenizer_source_path", str(tokenizer_path))

            model_info = mlflow.pyfunc.log_model(
                artifact_path="seq2seq_translator",
                python_model=TranslatorPyfunc(),
                artifacts={
                    "checkpoint": checkpoint_path.as_posix(),
                    "tokenizer": tokenizer_path.as_posix(),
                },
                registered_model_name=model_name,
            )

        print(f"Logged model artifact URI: {model_info.model_uri}")

        client = MlflowClient()
        model_version = _find_registered_version(client, model_name, run.info.run_id)

        if not model_version:
            print("Could not resolve newly registered model version automatically.")
            print(f"Run ID: {run.info.run_id}")
            print(f"Model name: {model_name}")
            return

        print(f"Registered model: models:/{model_name}/{model_version}")

        try:
            stage = _normalize_stage("staging")
            client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage=stage,
                archive_existing_versions=False,
            )
            print(f"Transitioned model version {model_version} to stage {stage}.")
        except Exception as exc:
            print("Automatic stage transition failed (possibly due to permissions).")
            print(f"Error: {exc}")
            print(f"Next step: in DagsHub/MLflow UI, transition version {model_version} to 'Staging' (or 'Production' for prod).")
            print("Or run this command manually:")
            print(
                f"mlflow models transition-stage -m \"{model_name}\" -v \"{model_version}\" -s \"Staging\""
            )
            print(
                f"mlflow models transition-stage -m \"{model_name}\" -v \"{model_version}\" -s \"Production\""
            )


if __name__ == "__main__":
    main()
