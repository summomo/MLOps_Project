# MLOps Final Project

## Registry-first (方案 A) 使用说明

### 1) 安装依赖

```powershell
pip install -r requirements.txt
```

### 2) 设置环境变量（Windows PowerShell）

> 注意：不要把外部 project 的模型文件复制到本仓库；通过绝对路径注入。

```powershell
$env:MODEL_CKPT_PATH="C:\absolute\path\to\best.pt"
$env:TOKENIZER_PATH="C:\absolute\path\to\spm_joint_fr2en_v3.model"
$env:MODEL_NAME="fr2en-translator"
$env:MODEL_STAGE="Staging"
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
```

`MODEL_STAGE` 必须使用 MLflow 固定枚举且大小写严格一致：`Staging`、`Production`、`Archived`、`None`。
生产环境请严格设置为：`MODEL_STAGE="Production"`（避免误加载非生产 stage）。

如需认证，请额外设置你 MLflow Server 所需的环境变量（例如用户名、密码或 token）。

### 3) 打包并注册到 MLflow Registry

```powershell
python ml/src/export_mlflow_model.py
```

脚本会：
- 读取 `MODEL_CKPT_PATH` 和 `TOKENIZER_PATH`
- 将 checkpoint + tokenizer 作为 artifacts 一起打包到 `mlflow.pyfunc`
- 记录参数：`git_commit`、`dvc_data_rev`、`ckpt_source_path`、`tokenizer_source_path`
- 注册到 `MODEL_NAME`（默认 `fr2en-translator`）
- 尝试自动切换到 `Staging`，如果权限不足会打印手动命令

### 4) 启动 API（仅从 Registry Stage 加载）

```powershell
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
```

服务启动时会执行：

`mlflow.pyfunc.load_model("models:/{MODEL_NAME}/{MODEL_STAGE}")`

不使用本地 `torch.load(best.pt)` 路径加载。

### 5) 调用接口

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/translate" -ContentType "application/json" -Body '{"text":"bonjour le monde"}'
```

返回示例：

```json
{
  "translation": "...",
  "model_name": "fr2en-translator",
  "model_stage": "Staging",
  "model_uri": "models:/fr2en-translator/Staging"
}
```

### 6) 推理逻辑说明

当前仓库未发现可复用的现成 seq2seq beam-search 推理入口，因此 `ml/src/seq2seq_mlflow_model.py` 中实现了最小可运行推理：
- 优先尝试调用 checkpoint 内可用的 `translate` / `predict` / `forward`
- 若无法匹配真实模型结构，则回退到 tokenizer 编码后解码的最小 stub，保证服务可运行

⚠️ 当前 stub 主要用于验证 **MLOps 流程（打包、注册、按 stage 加载、在线服务）**，不代表真实翻译质量。

后续请将真实 greedy/beam decode 逻辑替换到 `TranslatorPyfunc._translate_with_model`（文件：`ml/src/seq2seq_mlflow_model.py`）。
