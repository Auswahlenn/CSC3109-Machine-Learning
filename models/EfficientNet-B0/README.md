# EfficientNet-B0 final reproducibility guide

This guide describes the retrained `efficientnet_b0` run used in the final
report and Docker deployment. The run metadata is recorded in
`results/efficientnet_b0_run.json`.

## Final configuration

- Input: raw `256 x 256 x 3` RGB pixels in `[0, 255]`.
- Shared training-only augmentation: horizontal/vertical flips, full rotation,
  zoom, mild contrast, and mild brightness.
- Backbone: ImageNet-pretrained EfficientNet-B0 with global average pooling.
- Backbone state: frozen.
- Classification head: dropout 0.3 and four-unit softmax.
- Optimizer: Adam with learning rate 0.001.
- Loss: categorical cross-entropy.
- Batch size: 16.
- Seed: 42.
- Epoch budget: 32.
- Early-stopping patience: 20 on internal tuning accuracy.
- Selected checkpoint epoch: 32.
- TensorFlow version recorded by the run: 2.21.0.

The saved model owns augmentation and EfficientNet preprocessing, so inference
must supply raw RGB values rather than pre-normalized tensors.

## Data protocol

- `data/set 23` supplies training and internal tuning data.
- Two byte-identical images with conflicting class labels are excluded
  logically by `shared/config.py`.
- A deterministic stratified 15% tuning split is derived only from `set 23`.
- `data/val 23` is excluded from fitting and checkpoint selection.

Verified split counts:

| Split | Total | coastal_mansion | dense_residential | nursing_home | sparse_residential |
|---|---:|---:|---:|---:|---:|
| Training | 2,378 | 594 | 595 | 595 | 594 |
| Tuning | 420 | 105 | 105 | 105 | 105 |
| Held-out | 400 | 100 | 100 | 100 | 100 |

## Final results

| Metric | Tuning | Held-out |
|---|---:|---:|
| Accuracy | 0.9690 | **0.9775** |
| Macro precision | 0.9695 | **0.9779** |
| Macro recall | 0.9690 | **0.9775** |
| Macro F1 | 0.9691 | **0.9775** |

Held-out per-class metrics:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| `coastal_mansion` | 0.9899 | 0.9800 | 0.9849 |
| `dense_residential` | 0.9796 | 0.9600 | 0.9697 |
| `nursing_home` | 0.9898 | 0.9700 | 0.9798 |
| `sparse_residential` | 0.9524 | 1.0000 | 0.9756 |

The held-out set contains nine errors. The largest single confusion is three
`dense_residential` images predicted as `sparse_residential`.

## Reproduce training

From the repository root with Python 3.12:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
python train.py --model efficientnet_b0 --run-name efficientnet_b0 `
  --epochs 32 --patience 20 --batch-size 16 `
  --run-type final --evaluate-held-out
```

For WSL GPU training, use the repository wrapper after configuring
`CSC3109_VENV` if the environment is not at its default location:

```bash
bash scripts/wsl_tensorflow.sh train.py \
  --model efficientnet_b0 --run-name efficientnet_b0 \
  --epochs 32 --patience 20 --batch-size 16 \
  --run-type final --evaluate-held-out
```

The checked-in `scripts/run_efficientnet_final.sh` wraps the same final
configuration for WSL.

## Final artifacts

- `results/efficientnet_b0_best.keras`
- `results/efficientnet_b0.json`
- `results/efficientnet_b0_confusion_matrix.png`
- `results/efficientnet_b0_tuning.json`
- `results/efficientnet_b0_tuning_confusion_matrix.png`
- `results/efficientnet_b0_history.json`
- `results/efficientnet_b0_curves.png`
- `results/efficientnet_b0_run.json`
- `results/efficientnet_b0_manifest.json`

The Keras checkpoint is ignored by Git and must be distributed separately or
inside the Docker image.

Reviewed local checkpoint:

- Size: 17,124,387 bytes
- SHA-256:
  `b46df5800879aac344866957d489afb9878dc7c8dc0887a8a94e7b082b20bed7`

## Load and predict

```python
from PIL import Image

from frontend.inference import load_model, predict_probabilities

model, error = load_model("results/efficientnet_b0_best.keras")
if error:
    raise RuntimeError(error)

scores = predict_probabilities(model, Image.open("image.jpg"))
```

Fixed class order:

1. `coastal_mansion`
2. `dense_residential`
3. `nursing_home`
4. `sparse_residential`

## Docker deployment

The final Docker configuration already targets this checkpoint:

```powershell
docker build -t csc3109-aerial:latest .
docker run --name csc3109-aerial -p 8501:8501 csc3109-aerial:latest
```

Open `http://localhost:8501`, upload an aerial image, and inspect the predicted
label and complete four-class confidence chart.
