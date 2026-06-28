# EfficientNet-B0 reproducibility and handoff

## Selected configuration

Phase 4 selected the frozen-backbone baseline using internal tuning macro F1:

- Input: 224 x 224 RGB, raw `[0, 255]` pixels.
- Shared training-only augmentation: horizontal/vertical flips, rotation,
  contrast, and brightness.
- Backbone: ImageNet-pretrained EfficientNet-B0 with global average pooling.
- Backbone state: frozen.
- Classification head: dropout 0.3 and four-unit softmax.
- Optimizer: Adam with learning rate 0.001.
- Loss: categorical cross-entropy.
- Batch size: 32.
- Seed: 42.
- Epoch budget: 15 with early-stopping patience 3 on internal tuning accuracy.

## Data protocol

- `dataset/set 23` supplies training and internal tuning data.
- The deterministic stratified tuning fraction is 15%.
- The conflicting byte-identical cross-class pair is excluded logically.
- `dataset/val 23` is held out from fitting, checkpoint selection, and
  hyperparameter selection.
- The held-out split is evaluated only for the smoke verification and final
  selected run; smoke metrics are non-reportable.

Verified split counts:

| Split | Total | coastal_mansion | dense_residential | nursing_home | sparse_residential |
|---|---:|---:|---:|---:|---:|
| Training | 2,378 | 594 | 595 | 595 | 594 |
| Tuning | 420 | 105 | 105 | 105 | 105 |
| Held-out | 400 | 100 | 100 | 100 | 100 |

## Controlled experiment result

| Run | Dropout | Learning rate | Best epoch | Tuning macro F1 |
|---|---:|---:|---:|---:|
| `efficientnet_b0_baseline` | 0.3 | 0.001 | 10 | 0.9646 |
| `efficientnet_b0_drop04` | 0.4 | 0.001 | 11 | 0.9621 |
| `efficientnet_b0_lr3e4` | 0.3 | 0.0003 | 9 | 0.9406 |

The complete experiment manifest is
`results/efficientnet_b0_experiment_summary.json`.

## Observed tuning errors

- `nursing_home` had the strongest per-class F1 at 0.9855.
- `coastal_mansion` had the lowest per-class F1 at 0.9459 because 12 images
  from other classes were predicted as coastal mansion.
- Eight `sparse_residential` images were predicted as `coastal_mansion`.
- Four `dense_residential` images were predicted as `coastal_mansion`.
- The repository contains a ResNet50 example implementation but no reportable
  result artifact from another architecture. Cross-architecture performance
  comparison cannot be verified from the current repository.

## Final held-out result

- Accuracy: 0.9625.
- Macro precision: 0.9632.
- Macro recall: 0.9625.
- Macro F1: 0.9625.
- Strongest held-out per-class F1: `coastal_mansion` at 0.9798.
- Weakest held-out per-class F1: `dense_residential` at 0.9447.
- The largest held-out confusion was four `dense_residential` images predicted
  as `sparse_residential`.

The final checkpoint is checksum-tracked in
`results/efficientnet_b0_final_manifest.json`.

## Reproduce training

From PowerShell:

```powershell
wsl.exe --distribution Ubuntu --exec /bin/bash -lc `
  'cd /mnt/c/Users/junki/Desktop/projects/CSC3109-Machine-Learning && bash scripts/run_efficientnet_final.sh'
```

The WSL Python environment defaults to `/home/jk/.venvs/csc3109`. Override it
with `CSC3109_VENV` when invoking `scripts/wsl_tensorflow.sh` if required.

## Load the selected model

```python
from tensorflow import keras

model = keras.models.load_model("results/efficientnet_b0_final_best.keras")
probabilities = model.predict(raw_rgb_batch)
```

The saved model contains augmentation, preprocessing, backbone, and classifier.
Keras disables the random augmentation layers during inference.

Fixed class order:

1. `coastal_mansion`
2. `dense_residential`
3. `nursing_home`
4. `sparse_residential`

## Final artifacts

- `results/efficientnet_b0_final_best.keras`
- `results/efficientnet_b0_final.json`
- `results/efficientnet_b0_final_confusion_matrix.png`
- `results/efficientnet_b0_final_tuning.json`
- `results/efficientnet_b0_final_tuning_confusion_matrix.png`
- `results/efficientnet_b0_final_history.json`
- `results/efficientnet_b0_final_run.json`
- `results/efficientnet_b0_final_manifest.json`
