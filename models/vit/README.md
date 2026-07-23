# Vision Transformer (ViT-B/16) reproducibility and handoff

## Selected configuration

- Input: 256 x 256 RGB, raw `[0, 255]` pixels (shared `config.IMAGE_SIZE`).
- Shared training-only augmentation: horizontal/vertical flips, full rotation,
  zoom, mild contrast and brightness jitter.
- Model-internal resize: 256 x 256 -> 224 x 224 (bilinear), applied after
  augmentation and before the backbone (see "Why 224" below).
- Preprocessing: raw `[0, 255]` -> `[-1, 1]` (ViT convention; NOT the
  channel-wise ImageNet mean/std used by the CNN backbones).
- Backbone: ImageNet-pretrained ViT-B/16 (`vit_base_patch16_224_imagenet`
  via keras-hub) — 16x16 patches, 12 transformer layers, 768 hidden dim,
  12 attention heads.
- Backbone state: frozen (feature extraction; 3,076 of 85,801,732 parameters
  trainable — the smallest trainable footprint of the team's five approaches).
- Global representation: CLS token embedding (ViT's counterpart to the CNNs'
  global average pooling).
- Classification head: dropout 0.3 and four-unit softmax.
- Optimizer: Adam with learning rate 0.001. Loss: categorical cross-entropy.
- Batch size: 16 (shared team flag). Seed: 42.
- Epoch budget: 32 with early-stopping patience 20 on internal tuning
  accuracy; training stopped at epoch 30 with the best checkpoint at epoch 10.

## Why 224 (not the shared 256)

ViT's learned position embeddings form a fixed table with one entry per patch
position: a 224 x 224 input yields 14 x 14 = 196 patches + 1 CLS token = 197
entries. A 256 x 256 input would produce 257 tokens, which the pretrained
table cannot represent — the backbone raises a hard shape error rather than
degrading silently. CNNs tolerate off-resolution input because convolution
weights are shared across positions; ViT makes the mismatch explicit.

The model therefore resizes to 224 internally, immediately before the
backbone. Two fairness notes for the team comparison:

- The resize sits AFTER the shared augmentation, so regularisation is applied
  identically to all five models at 256 x 256.
- The team's CNN backbones (ResNet-50, EfficientNet-B0, MobileNetV2) were
  also originally pretrained at 224 x 224, so the ViT is arguably the only
  model evaluated at its native pretraining resolution.

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

Internal tuning split (checkpoint selection): accuracy 0.9548, macro F1 0.9549.

Held-out `val 23` (reportable):

- Accuracy: 0.9650
- Macro precision: 0.9659, macro recall: 0.9650, macro F1: 0.9652
- Strongest per-class F1: `nursing_home` at 0.9798 (97/100 correct) and
  `coastal_mansion` at 0.9796 with perfect precision (nothing from any other
  class was ever predicted as coastal mansion).
- Weakest per-class F1: `sparse_residential` at 0.9505 and
  `dense_residential` at 0.9510 — the errors concentrate on this pair, most
  notably 4 true `sparse_residential` images predicted as `dense_residential`.
  This is the most genuinely ambiguous boundary in the partition (a density
  judgement rather than a structural one).
- Tuning vs held-out macro F1 (0.9549 vs 0.9652) are close, consistent with
  the tiny 3,076-parameter head leaving little room to overfit.

## Ablation: ImageNet-1k vs ImageNet-21k pretraining (`ViT21k.py`)

`ViT.py` uses `vit_base_patch16_224_imagenet` (pretrained on ImageNet-21k,
then finetuned on ImageNet-1k). `ViT21k.py` is byte-identical except for the
preset: `vit_base_patch16_224_imagenet21k` (raw 21k weights, never finetuned
to the 1k label space). Rationale: aerial imagery is a domain shift from
natural photos, so features finetuned toward the 1k object categories may be
more specialised than helpful for frozen-backbone transfer.

Both variants share the same seed, split, augmentation, and training flags;
the comparison is made on the internal tuning split only (`val 23` stays
reserved for final runs, matching the EfficientNet-B0 experiment protocol).

Results (internal tuning split, both variants):

| Variant | Preset | Tuning accuracy | Tuning macro F1 | Best epoch |
|---|---|---:|---:|---:|
| `ViT` (selected) | `..._imagenet` (1k-finetuned) | 0.9548 | 0.9549 | 10 |
| `ViT21k` | `..._imagenet21k` (raw 21k) | 0.9405 | 0.9408 | 31 |

The 1k-finetuned checkpoint won on both axes: about 1.4 points higher macro
F1, and it converged three times faster (best epoch 10 vs 31 — the 21k
variant was still inching upward when the 32-epoch budget ended). A frozen
backbone can only offer the features it already has; the 1k finetuning
appears to have sharpened object-discriminative structure that the linear
head can exploit directly, while the raw 21k features left more work to the
3,076-parameter head than it could finish. The 1k variant (`ViT.py`) was
therefore kept as the final model; only it was evaluated on `val 23`.

## Reproduce training

Final run (the reported numbers):

```bash
python train.py --model ViT --run-name ViT --evaluate-held-out --run-type final --batch-size 16 --patience 20 --epochs 32
```

Pretraining-source ablation:

```bash
python train.py --model ViT21k --run-name ViT21k --run-type experiment --batch-size 16 --patience 20 --epochs 32
```

Notes:

- `--model ViT` is CASE-SENSITIVE on WSL/Linux.
- If `shared/config.py`'s default data directory does not match your local
  layout, set `CSC3109_DATA_DIR` to the folder containing `set 23` and
  `val 23` instead of renaming folders.

## Load the trained model

```python
from tensorflow import keras

import models.vit.ViT  # REQUIRED: registers the custom ViTPreprocess layer

model = keras.models.load_model("results/ViT_best.keras")
probabilities = model.predict(raw_rgb_batch)  # raw [0, 255], 256x256
```

> Without the `models.vit.ViT` import, `load_model` fails with
> "Could not locate class 'ViTPreprocess'" — the checkpoint references a
> custom layer that must be registered first. Any deployment script (e.g.
> `frontend/web.py`) selecting this checkpoint needs the same import.

The saved model contains augmentation, the 256->224 resize, preprocessing,
backbone, and classifier; the random augmentation layers are inactive at
inference. Fixed class order: `coastal_mansion`, `dense_residential`,
`nursing_home`, `sparse_residential`.

## Artifacts

- `results/ViT_best.keras` (gitignored, ~344 MB — local only)
- `results/ViT.json` / `results/ViT_confusion_matrix.png` (held-out)
- `results/ViT_tuning.json` / `results/ViT_tuning_confusion_matrix.png`
- `results/ViT_history.json` / `results/ViT_curves.png`
- `results/ViT_run.json`
- `results/ViT21k_*` (ablation, tuning-split only)

## Supporting article

**Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X.,
Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S.,
Uszkoreit, J., & Houlsby, N. (2021). "An Image is Worth 16x16 Words:
Transformers for Image Recognition at Scale." International Conference on
Learning Representations (ICLR).** — <https://arxiv.org/abs/2010.11929>

This is the original Vision Transformer paper and the direct basis for this
model. It shows that a pure transformer applied to sequences of image patches
can match or exceed CNNs — but only when pretrained at scale, because ViT
lacks the convolutional inductive biases (locality, translation equivariance)
that let CNNs learn from small datasets. That finding drives every design
decision here: reusing the paper's exact ViT-B/16 backbone pretrained on
ImageNet, freezing it, and fitting only a small head to our 2,378 training
images. The paper is also the source of the fixed position-embedding grid
that necessitates the 224 x 224 input.
