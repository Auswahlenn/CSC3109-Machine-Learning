# ResNet-50 reproducibility and handoff

## Selected configuration

- Input: 224 x 224 RGB, raw `[0, 255]` pixels (see "Known issue" below re: the
  shared `IMAGE_SIZE` change).
- Shared training-only augmentation: horizontal/vertical flips, full rotation,
  mild contrast and brightness jitter.
- Backbone: ImageNet-pretrained ResNet-50 with global average pooling.
- Backbone state: frozen (feature extraction; 8,196 of 23,595,908 parameters
  trainable).
- Classification head: dropout 0.3 and four-unit softmax.
- Optimizer: Adam with learning rate 0.001.
- Loss: categorical cross-entropy.
- Batch size: 32. Seed: 42.
- Epoch budget: 30 with early-stopping patience 5 on internal tuning accuracy;
  training stopped at epoch 12 with the best checkpoint at epoch 7.

## Data protocol

Identical to the shared team protocol (`shared/data.py`):

| Split | Total | coastal_mansion | dense_residential | nursing_home | sparse_residential |
|---|---:|---:|---:|---:|---:|
| Training | 2,378 | 594 | 595 | 595 | 594 |
| Tuning (15% of `set 23`) | 420 | 105 | 105 | 105 | 105 |
| Held-out (`val 23`) | 400 | 100 | 100 | 100 | 100 |

The held-out split was never used for fitting, checkpoint selection, or
hyperparameter selection. The two byte-identical cross-class duplicates are
excluded logically by the shared loader.

## Results

Internal tuning split (checkpoint selection): accuracy 0.9762, macro F1 0.9762.

Held-out `val 23` (reportable):

- Accuracy: 0.9525
- Macro precision: 0.9542, macro recall: 0.9525, macro F1: 0.9527
- Strongest per-class F1: `nursing_home` at 0.9655 (98/100 correct).
- Weakest per-class F1: `sparse_residential` at 0.9372 — its precision is
  0.9065 because 10 images from other classes were predicted as
  sparse residential, 6 of them true `coastal_mansion`.
- Other notable confusion: 5 `dense_residential` images predicted as
  `nursing_home`.

For comparison, the team's EfficientNet-B0 final run reports held-out accuracy
0.9625 / macro F1 0.9625 on the same protocol.

## Known issue: shared IMAGE_SIZE changed after training

This model was trained at commit `5afcbcc` with `config.IMAGE_SIZE = 224`
(recorded in `results/resnet50_run.json`). Commit `2dc40e8` later changed the
frozen `shared/config.py` to `IMAGE_SIZE = 256`. The saved checkpoint
`results/resnet50_best.keras` has a fixed 224x224 input, so re-evaluating or
serving it against the current 256-pixel loader fails with a shape mismatch.
All existing team results (custom CNN, EfficientNet-B0, ResNet-50) were
produced at 224, so they remain mutually comparable — but the team must either
revert `IMAGE_SIZE` to 224 or retrain every model at 256 before any new runs
are compared against these numbers.

## Reproduce training

```bash
python train.py --model resnet50 --evaluate-held-out --run-type final
```

Controlled hyperparameter experiments (same factors as the EfficientNet-B0
study) via the shared entrypoint:

```bash
python train.py --model resnet50 --run-name resnet50_drop04 --dropout 0.4
python train.py --model resnet50 --run-name resnet50_lr3e4 --learning-rate 3e-4
```

## Load the trained model

```python
from tensorflow import keras

model = keras.models.load_model("results/resnet50_best.keras")
probabilities = model.predict(raw_rgb_batch)  # raw [0, 255], 224x224
```

The saved model contains augmentation, preprocessing, backbone, and classifier;
the random augmentation layers are inactive at inference. Fixed class order:
`coastal_mansion`, `dense_residential`, `nursing_home`, `sparse_residential`.

## Artifacts

- `results/resnet50_best.keras`
- `results/resnet50.json` / `results/resnet50_confusion_matrix.png` (held-out)
- `results/resnet50_tuning.json` / `results/resnet50_tuning_confusion_matrix.png`
- `results/resnet50_history.json`
- `results/resnet50_run.json`

## Supporting article

**He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for
Image Recognition." IEEE Conference on Computer Vision and Pattern Recognition
(CVPR), pp. 770–778. doi:10.1109/CVPR.2016.90** —
<https://arxiv.org/abs/1512.03385>

This is the original ResNet paper and the direct basis for this model. It
introduces residual (skip) connections, which solve the degradation problem
that prevents plain deep networks from training well, allowing the 50-layer
backbone used here to learn richer features than a shallower or plain CNN.
The paper's ResNet-50 — pretrained on ImageNet and reused here as a frozen
feature extractor — is the exact architecture behind this model, and its
demonstrated transferability to other recognition tasks is what justifies
applying it to this small fine-grained aerial dataset.
