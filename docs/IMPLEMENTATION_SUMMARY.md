# EfficientNet-B0 Phase 3-6 Implementation Summary

Verified on 23 June 2026 from branch `model/effNet-b0`.

## Delivery status

| Phase | Delivered outcome | Evidence |
|---|---|---|
| 3 | Leakage-free training/tuning pipeline and end-to-end GPU smoke test | `tests/`, smoke-run records in `agent.md` |
| 4 | Three controlled EfficientNet-B0 experiments selected on internal tuning macro F1 | `results/efficientnet_b0_experiment_summary.json` |
| 5 | Reproduced final model, held-out evaluation, checksum manifest, and handoff instructions | `results/efficientnet_b0_final_manifest.json`, `models/EfficientNet-B0/README.md` |
| 6 | Checksum-verified FastAPI WebUI/API and a healthy non-root Docker container | `results/efficientnet_b0_deployment_verification.json` |

All Phase 3-6 acceptance criteria in `agent.md` are complete.

## What is implemented

### Data and evaluation protocol

- `dataset/` is the default source root, with `CSC3109_DATA_DIR` as an override.
- A deterministic stratified split derives 2,378 training and 420 tuning images
  from the supplied training data.
- The 400-image held-out set remains outside tuning and checkpoint selection.
- The byte-identical, conflicting cross-class pair is excluded logically without
  editing the supplied dataset.
- Training, tuning, and held-out file counts and paths are preserved in run
  metadata.

### Model and experiments

- `models/EfficientNet-B0/efficientnet_b0.py` implements the shared
  `build_model(num_classes, augmentation)` contract.
- The saved model includes augmentation, EfficientNet input handling, an
  ImageNet-pretrained frozen EfficientNet-B0 backbone, dropout, and a four-class
  softmax head.
- Three one-factor experiments compare the baseline against dropout 0.4 and a
  learning rate of 0.0003.
- Selection uses internal tuning macro F1 only. The selected baseline uses
  dropout 0.3 and learning rate 0.001.
- Run metadata, histories, tuning metrics, per-class metrics, and confusion
  matrices are committed under `results/`.

### Final held-out result

| Metric | Value |
|---|---:|
| Accuracy | 0.9625 |
| Macro precision | 0.9632 |
| Macro recall | 0.9625 |
| Macro F1 | 0.9625 |

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| `coastal_mansion` | 0.9898 | 0.9700 | 0.9798 |
| `dense_residential` | 0.9495 | 0.9400 | 0.9447 |
| `nursing_home` | 0.9794 | 0.9500 | 0.9645 |
| `sparse_residential` | 0.9340 | 0.9900 | 0.9612 |

The selected checkpoint is 17,122,054 bytes. Its committed SHA-256 is
`14fd5941a8734ef81a41d13211aef7128e5f64179d1b805b15e16b6e3b94b773`.

### Deployment

- FastAPI serves a browser upload interface, readiness response, multipart
  prediction endpoint, and generated OpenAPI documentation.
- The service verifies the checkpoint SHA-256 against the manifest during
  startup.
- Upload validation covers media type, maximum size, image readability, output
  shape, finite values, and normalized scores.
- The Docker image copies only the application, selected checkpoint, manifest,
  and pinned runtime dependencies.
- The container runs as `appuser` with UID/GID 10001 and declares a health
  check.
- Docker Desktop 4.79.0 and Engine 29.5.3 built a 744,094,351-byte Linux AMD64
  image. The verification container reached `healthy`; `/`, `/health`, `/docs`,
  and `/predict` returned HTTP 200.
- A training-set image was used only to exercise the deployed request path. Its
  prediction is not used as model-evaluation evidence.

## What improved

- Held-out leakage was removed from tuning, early stopping, and model selection.
- Experiment changes are controlled one factor at a time and ranked by one
  recorded selection metric.
- Seeds, split details, hyperparameters, durations, best epochs, Git commits,
  and artifacts are retained for reproduction.
- The final checkpoint is checksum-bound to its class order and preprocessing
  contract.
- Training, serialization, evaluation, API validation, and real-checkpoint
  inference have automated contract coverage.
- Deployment now has both local-service and actual-container evidence, including
  health, non-root identity, and an end-to-end image request.

## Verification completed

- Full WSL TensorFlow suite: 21 tests passed in 65.33 seconds.
- Final checkpoint reload and inference: passed.
- Final checkpoint SHA-256: matched the committed manifest.
- Docker image build: passed.
- Container health check: `healthy`.
- Runtime user: UID/GID 10001.
- WebUI, health, OpenAPI, and prediction endpoints: HTTP 200.
- Prediction response: four class scores summing to 0.9999999.

## Current boundaries

- The repository contains no reportable result from another architecture, so a
  cross-architecture performance comparison cannot be verified locally.
- The Docker verification uses CPU inference. TensorFlow reports no CUDA driver
  inside the container.
- Functional WebUI delivery is verified over HTTP; pixel-level inspection with
  the in-app browser is unavailable because its controller encounters a Node
  module-mode conflict outside this repository.
- The deployment verification uses one image request and does not constitute a
  latency, throughput, concurrency, or load benchmark.

## Improvements to consider after Phase 6

1. Add reportable results from the other team architectures using the same split
   and metric schema, then produce a direct comparison table.
2. Evaluate partial backbone fine-tuning as a separate controlled experiment,
   using only the internal tuning split for selection.
3. Repeat selected experiments across multiple seeds and report the distribution
   of macro F1 instead of a single run only.
4. Measure probability calibration on held-out predictions and document any
   confidence threshold used by the interface.
5. Add container start, health, and prediction checks to CI and retain the image
   digest as a build artifact.
6. Add an SBOM and image vulnerability scan to the release workflow.
7. Benchmark CPU latency, warm-up time, memory use, throughput, and concurrent
   requests on the intended deployment hardware.
8. Evaluate a smaller CPU-specific runtime or an exported inference format while
   requiring metric parity with the committed Keras checkpoint.
9. Add browser-level UI tests after resolving the external controller conflict,
   including invalid uploads and narrow-screen layout.
10. Add authentication, rate limiting, request logging policy, and transport
    security before exposing the service outside a trusted local environment.
11. Define post-deployment monitoring for input drift, class distribution, model
    version, latency, and prediction failures.

## Key artifacts

- Model handoff: `models/EfficientNet-B0/README.md`
- Final checkpoint manifest: `results/efficientnet_b0_final_manifest.json`
- Final held-out metrics: `results/efficientnet_b0_final.json`
- Experiment ranking: `results/efficientnet_b0_experiment_summary.json`
- Deployment guide: `deployment/README.md`
- Deployment verification: `results/efficientnet_b0_deployment_verification.json`
