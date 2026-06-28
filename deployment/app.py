"""FastAPI WebUI and JSON inference endpoint for the selected model."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError
from tensorflow import keras


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH", ROOT / "results" / "efficientnet_b0_final_best.keras"
    )
)
MANIFEST_PATH = Path(
    os.environ.get(
        "MODEL_MANIFEST_PATH",
        ROOT / "results" / "efficientnet_b0_final_manifest.json",
    )
)
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))

_model: keras.Model | None = None
_manifest: dict[str, Any] | None = None
_prediction_lock = threading.Lock()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_artifacts() -> tuple[keras.Model, dict[str, Any]]:
    """Load and validate the model/manifest pair once per process."""
    global _model, _manifest
    if _model is not None and _manifest is not None:
        return _model, _manifest

    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model checkpoint not found: {MODEL_PATH}")
    if not MANIFEST_PATH.is_file():
        raise RuntimeError(f"Model manifest not found: {MANIFEST_PATH}")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    actual_hash = _sha256(MODEL_PATH)
    if actual_hash != manifest["checkpoint_sha256"]:
        raise RuntimeError(
            "Model checksum mismatch: "
            f"expected {manifest['checkpoint_sha256']}, received {actual_hash}"
        )

    model = keras.models.load_model(MODEL_PATH, compile=False)
    if int(model.output_shape[-1]) != len(manifest["class_names"]):
        raise RuntimeError("Model output width does not match manifest classes")

    _model = model
    _manifest = manifest
    return model, manifest


def _prepare_image(payload: bytes, image_size: int) -> np.ndarray:
    try:
        with Image.open(BytesIO(payload)) as image:
            image = image.convert("RGB")
            image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
            array = np.asarray(image, dtype=np.float32)
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("Uploaded file is not a readable image") from error
    return np.expand_dims(array, axis=0)


def _predict(payload: bytes) -> dict[str, Any]:
    model, manifest = load_artifacts()
    batch = _prepare_image(payload, int(manifest["image_size"]))
    with _prediction_lock:
        probabilities = model.predict(batch, verbose=0)[0]

    if not np.isfinite(probabilities).all():
        raise RuntimeError("Model produced non-finite probabilities")
    class_names = manifest["class_names"]
    predicted_index = int(np.argmax(probabilities))
    return {
        "predicted_label": class_names[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "scores": {
            class_name: float(probability)
            for class_name, probability in zip(class_names, probabilities, strict=True)
        },
    }


@asynccontextmanager
async def lifespan(_: FastAPI):
    await asyncio.to_thread(load_artifacts)
    yield


app = FastAPI(
    title="Aerial Scene Classifier",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return WEB_UI


@app.get("/health")
def health() -> dict[str, Any]:
    _, manifest = load_artifacts()
    return {
        "status": "ok",
        "model": manifest["model_name"],
        "classes": manifest["class_names"],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload must be an image")

    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds upload limit")
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    try:
        return await asyncio.to_thread(_predict, payload)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


WEB_UI = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Aerial Scene Classifier</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, sans-serif; }
    body { margin: 0; min-height: 100vh; background: #07111f; color: #e8f1ff; }
    main { width: min(920px, calc(100% - 32px)); margin: 0 auto; padding: 64px 0; }
    .eyebrow { color: #65d7c0; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    h1 { margin: 12px 0; font-size: clamp(2.2rem, 7vw, 4.8rem); line-height: .98; }
    .lede { max-width: 650px; color: #9fb0c8; font-size: 1.08rem; line-height: 1.6; }
    .panel { margin-top: 36px; padding: 24px; border: 1px solid #213550; border-radius: 20px; background: #0d1b2d; box-shadow: 0 24px 80px #0008; }
    .drop { display: grid; place-items: center; min-height: 220px; border: 1px dashed #3c5878; border-radius: 14px; background: #0a1626; cursor: pointer; text-align: center; overflow: hidden; }
    .drop:hover { border-color: #65d7c0; }
    #preview { display: none; width: 100%; max-height: 420px; object-fit: contain; }
    input { display: none; }
    button { width: 100%; margin-top: 16px; padding: 14px; border: 0; border-radius: 12px; background: #65d7c0; color: #062018; font-weight: 800; font-size: 1rem; cursor: pointer; }
    button:disabled { opacity: .45; cursor: wait; }
    #status { min-height: 24px; color: #9fb0c8; }
    #result { display: none; margin-top: 24px; }
    .prediction { font-size: 1.5rem; font-weight: 800; color: #65d7c0; }
    .score { display: grid; grid-template-columns: 170px 1fr 64px; gap: 10px; align-items: center; margin: 12px 0; }
    .bar { height: 10px; background: #192b42; border-radius: 99px; overflow: hidden; }
    .bar span { display: block; height: 100%; background: #65d7c0; }
    @media (max-width: 620px) { .score { grid-template-columns: 1fr 54px; } .bar { grid-column: 1 / -1; } }
  </style>
</head>
<body>
<main>
  <div class="eyebrow">EfficientNet-B0 · aerial imagery</div>
  <h1>Classify the landscape below.</h1>
  <p class="lede">Upload an aerial image to receive a predicted scene category and confidence scores across all four classes.</p>
  <section class="panel">
    <label class="drop" for="file">
      <div id="prompt"><strong>Choose an aerial image</strong><br><span>JPEG or PNG, up to 10 MB</span></div>
      <img id="preview" alt="Selected aerial image preview">
    </label>
    <input id="file" type="file" accept="image/*">
    <button id="predict" disabled>Run classification</button>
    <p id="status" role="status"></p>
    <div id="result">
      <div class="prediction" id="prediction"></div>
      <div id="scores"></div>
    </div>
  </section>
</main>
<script>
  const fileInput = document.querySelector('#file');
  const preview = document.querySelector('#preview');
  const prompt = document.querySelector('#prompt');
  const button = document.querySelector('#predict');
  const status = document.querySelector('#status');
  const result = document.querySelector('#result');

  fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    button.disabled = !file;
    result.style.display = 'none';
    if (file) {
      preview.src = URL.createObjectURL(file);
      preview.style.display = 'block';
      prompt.style.display = 'none';
    }
  });

  button.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file) return;
    button.disabled = true;
    status.textContent = 'Running EfficientNet-B0…';
    const body = new FormData();
    body.append('file', file);
    try {
      const response = await fetch('/predict', { method: 'POST', body });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || 'Prediction failed');
      document.querySelector('#prediction').textContent = `${payload.predicted_label} · ${(payload.confidence * 100).toFixed(1)}%`;
      const scores = document.querySelector('#scores');
      scores.replaceChildren();
      Object.entries(payload.scores).sort((a, b) => b[1] - a[1]).forEach(([label, value]) => {
        const row = document.createElement('div');
        row.className = 'score';
        const name = document.createElement('span');
        name.textContent = label;
        const bar = document.createElement('div');
        bar.className = 'bar';
        const fill = document.createElement('span');
        fill.style.width = `${value * 100}%`;
        bar.append(fill);
        const percent = document.createElement('span');
        percent.textContent = `${(value * 100).toFixed(1)}%`;
        row.append(name, bar, percent);
        scores.append(row);
      });
      result.style.display = 'block';
      status.textContent = '';
    } catch (error) {
      status.textContent = error.message;
    } finally {
      button.disabled = false;
    }
  });
</script>
</body>
</html>"""
