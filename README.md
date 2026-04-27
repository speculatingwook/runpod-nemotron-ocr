# RunPod Nemotron OCR Worker

[![Runpod](https://api.runpod.io/badge/speculatingwook/runpod-nemotron-ocr)](https://console.runpod.io/hub/speculatingwook/runpod-nemotron-ocr)

RunPod Serverless worker for NVIDIA Nemotron OCR v2. The container installs the
Nemotron OCR package from the Hugging Face repository, renders PDF pages with
PyMuPDF, and keeps the OCR model in a process-level cache so warm workers do not
reload the model for every request.

## Files

- `handler.py` - RunPod entrypoint and OCR orchestration.
- `Dockerfile` - CUDA/PyTorch base image plus Nemotron OCR v2 install.
- `requirements.txt` - runtime dependencies for the worker.
- `scripts/preload_model.py` - optional runtime model-load smoke check.
- `test_input.json` - initial RunPod request body.
- `.runpod/hub.json` - RunPod Hub listing metadata and deploy defaults.
- `.runpod/tests.json` - RunPod Hub release validation test.
- `.github/workflows/build-image.yml` - Docker Hub linux/amd64 image build.

## Local Checks

Local unit tests avoid importing RunPod, PyMuPDF, or Nemotron:

```bash
uv run --python 3.12 --with pytest pytest -q
```

## Build And Publish

Use GitHub Actions for the first build because the image targets linux/amd64 and
compiles a CUDA extension against the PyTorch/CUDA environment.

1. Push this project to GitHub.
2. Create a Docker Hub repository named `runpod-nemotron-ocr`.
3. Add these GitHub Actions secrets:

```text
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN
```

You can add them in GitHub under **Settings -> Secrets and variables -> Actions**,
or with the GitHub CLI:

```bash
gh secret set DOCKERHUB_USERNAME --body "<dockerhub-username>"
gh secret set DOCKERHUB_TOKEN
```

`DOCKERHUB_TOKEN` should be a Docker Hub access token, not your password.

4. Run **Build Docker Hub image** from the Actions tab.
5. Use the published image:

```text
docker.io/<dockerhub-username>/runpod-nemotron-ocr:0.1.0
```

## RunPod Settings

Create a custom Serverless endpoint from Docker registry/template:

```text
Endpoint type: Queue
GPU: RTX 4090, L40S, A100, or A40 fallback
Allowed CUDA version: 13.0
Min workers: 0
Max workers: 1
Idle timeout: 60-120s
Execution timeout: 10-20min
Container disk: 40GB+
Network volume: recommended
```

Recommended environment variables:

```text
HF_HOME=/runpod-volume/hf
TRANSFORMERS_CACHE=/runpod-volume/hf
TORCH_HOME=/runpod-volume/torch
NEMOTRON_LANG=multi
NEMOTRON_MODEL_DIR=/opt/nemotron-ocr-v2/v2_multilingual
```

## Request Shape

```json
{
  "input": {
    "pdf_url": "https://example.com/03-book5-part1-intro-missing-p031-036.pdf",
    "pages": [[1, 6]],
    "dpi": 200,
    "merge_level": "paragraph"
  }
}
```

Use exactly one of `pdf_url` or `pdf_base64`. Page numbers are 1-based.
