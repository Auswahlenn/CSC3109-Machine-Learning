# EfficientNet-B0 deployment

The deployment exposes:

- `GET /` - browser upload interface.
- `GET /health` - model readiness and class-order response.
- `POST /predict` - multipart image inference returning the predicted label,
  confidence, and all class scores.
- `GET /docs` - generated OpenAPI interface.

## Run locally in WSL2

```bash
bash scripts/wsl_tensorflow.sh -m pip install -r deployment/requirements.txt
bash scripts/wsl_tensorflow.sh -m uvicorn deployment.app:app \
  --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

## Build and run Docker

From the repository root:

```powershell
docker build -f deployment/Dockerfile -t csc3109-efficientnet-b0:1.0 .
docker run --rm -p 8000:8000 --name csc3109-efficientnet-b0 `
  csc3109-efficientnet-b0:1.0
```

Example API request:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@dataset/val 23/coastal_mansion/coastalmansion701.jpg"
```

The application verifies the model SHA-256 hash against the committed manifest
before accepting requests.

## Verified container

Verification completed on 23 June 2026 with Docker Desktop 4.79.0 and Docker
Engine 29.5.3:

- Image: `csc3109-efficientnet-b0:1.0`
- Platform: Linux AMD64
- Image size: 744,094,351 bytes
- Runtime identity: `appuser`, UID/GID 10001
- Container health: `healthy`
- HTTP 200: `/`, `/health`, `/docs`, and `/predict`

The machine-readable evidence is stored in
`results/efficientnet_b0_deployment_verification.json`.
