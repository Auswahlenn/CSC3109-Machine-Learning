# CSC3109 final implementation summary

Verified from the integrated `fix/docs` branch on 25 July 2026.

## Delivery status

| Area | Current implementation | Evidence |
|---|---|---|
| Data protocol | Deterministic train/tuning split with isolated held-out evaluation | `shared/data.py`, `tests/test_data_contract.py` |
| Model comparison | Five final model runs under the shared training contract | `models/`, `results/*_run.json` |
| Evaluation | Accuracy, macro and per-class precision/recall/F1, confusion matrices | `shared/evaluate.py`, `results/` |
| Deployment | EfficientNet-B0 served by a Streamlit UI in Docker | `Dockerfile`, `frontend/`, `docs/figures/deployment_ui.png` |
| Automated checks | Data, training, deployment-path, model-loading, and inference contracts | `tests/` |
| Report | Full LaTeX report with model analyses and live deployment evidence | `docs/final-report.tex` |

## Data and evaluation protocol

- `data/set 23` contains the professor-provided 2,800 training images.
- Two byte-identical files assigned conflicting labels are excluded logically.
- The remaining 2,798 images are split deterministically into 2,378 training
  and 420 internal-tuning images.
- `data/val 23` remains outside fitting and checkpoint selection and contains
  400 held-out images.
- Every saved model accepts raw `256 x 256` RGB data; model-specific
  preprocessing is embedded in the model graph.
- A full automated audit checks that all 3,200 source images are readable RGB
  files at `256 x 256`.
- Final checkpoints are selected by internal tuning accuracy. Held-out results
  are generated after training and do not feed back into another training run.

## Final model comparison

| Model | Accuracy | Macro F1 | Total parameters | Best epoch |
|---|---:|---:|---:|---:|
| EfficientNet-B0 | **0.9775** | **0.9775** | 4,054,695 | 32 |
| ResNet-50 | 0.9750 | 0.9750 | 23,595,908 | 25 |
| ViT-B/16 | 0.9650 | 0.9652 | 85,801,732 | 10 |
| MobileNetV2 | 0.9575 | 0.9576 | 2,263,108 | 30 |
| Custom CNN | 0.8450 | 0.8452 | 26,512,212 | 24 |

The table reports the fixed 400-image held-out evaluation. Detailed per-class
metrics and confusion matrices are stored under `results/`.

## Selected EfficientNet-B0 artifact

- Checkpoint: `results/efficientnet_b0_best.keras`
- Handoff manifest: `results/efficientnet_b0_manifest.json`
- Size: 17,124,387 bytes
- SHA-256:
  `b46df5800879aac344866957d489afb9878dc7c8dc0887a8a94e7b082b20bed7`
- Input: raw `256 x 256 x 3` RGB
- Backbone: ImageNet-pretrained EfficientNet-B0, frozen
- Head: dropout 0.3 and four-unit softmax
- Optimizer: Adam, learning rate 0.001
- Batch size: 16
- Epoch budget / selected epoch: 32 / 32
- Held-out accuracy / macro F1: 0.9775 / 0.9775

The manifest records the reviewed local handoff artifact. The current Streamlit
service does not perform runtime manifest verification.

## Current deployment

The final deployment is a Streamlit demonstrator, not the earlier development
FastAPI service.

- `frontend/inference.py` owns the selected checkpoint path, image preparation,
  checkpoint loading, and probability validation.
- `frontend/web.py` provides image upload, predicted label, an uncalibrated
  maximum softmax score, and a complete four-class score chart.
- `Dockerfile` copies only `efficientnet_b0_best.keras` and exposes port 8501.
- The Docker health check polls Streamlit's `/_stcore/health` endpoint.
- The reviewed image
  `sha256:dc5ca9e7cecf91d50f47bd768cbd2b784a3dac1821f90c7564829d43ee9119fb`
  is 528,328,047 bytes.
- The verification container reached `healthy`.
- A held-out coastal-mansion image returned the correct label with a maximum
  softmax score of 96.3%; the captured UI is included in the report.

There is no machine-to-machine `/predict` API in the final Streamlit
demonstrator. Adding FastAPI, authentication, request logging, batching, and
runtime checksum enforcement remains future production work.

## Verification completed

The root `.venv` collected and passed 20 tests:

- full-file image size, colour-mode, and readability audit;
- deterministic, disjoint split and exact counts;
- raw image/label batch contract;
- safe run-name and recursive model loading;
- configurable EfficientNet factor validation;
- Docker and Streamlit checkpoint-path agreement;
- image preprocessing shape/range;
- probability output shape and normalization;
- Keras checkpoint serialization/deserialization;
- real EfficientNet checkpoint loading and prediction.

Docker verification also completed:

- image build: passed;
- container health: `healthy`;
- Streamlit health endpoint: HTTP 200;
- real held-out upload and inference: passed.

## Current boundaries

- The selected checkpoint is intentionally ignored by Git and must accompany
  the Docker build context or be distributed through the built image.
- Results are single-seed observations; the report's approximate binomial
  intervals do not capture variation across repeated training runs.
- EfficientNet-B0 selected its final checkpoint at the epoch-budget boundary,
  so a longer pre-registered schedule remains worth evaluating.
- The browser UI itself is verified through the running container; automated
  tests cover its inference helpers and artifact contracts rather than
  pixel-level Streamlit rendering.
- The deployment is a coursework demonstrator, not a production inference
  service.
