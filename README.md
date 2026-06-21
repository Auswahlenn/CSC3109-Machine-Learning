# CSC3109-Machine-Learning

Fine-grained aerial image classification over 4 confusable categories
(`coastal_mansion`, `dense_residential`, `nursing_home`, `sparse_residential`)
using Keras/TensorFlow transfer learning.

## Layout

```
data/
  set 23/   # FIXED train set, 700 images/class (one subfolder per class)
  val 23/   # FIXED held-out validation set, 100 images/class
shared/     # FROZEN shared foundation (do not edit without notifying the team)
  config.py    # constants + set_seed()
  data.py      # get_datasets() -> raw (train_ds, val_ds), no normalization
  augment.py   # get_augmentation() -> training-only augmentation
  evaluate.py  # evaluate(model, val_ds, model_name) -> fixed-schema metrics
models/     # one module per team member, each exposes build_model(...)
results/    # per-model JSON + confusion-matrix PNG + best checkpoint
train.py    # shared training entrypoint (--model selects the member's model)
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python train.py --model example_resnet50
python train.py --model <yourname> --epochs 30 --patience 5
```

## Adding your model

Copy `models/example_resnet50.py` to `models/<yourname>.py` and expose:

```python
def build_model(num_classes: int, augmentation: tf.keras.Sequential) -> tf.keras.Model
```

returning a **compiled** model that stacks, in order: the shared `augmentation`
(passed in), your backbone's own `preprocess_input`, the pretrained backbone,
and a new softmax head. The shared loader returns **raw [0, 255] pixels** on
purpose — apply your backbone's `preprocess_input` inside your model so every
member can use a different backbone while sharing identical data and evaluation.

> The `shared/` files are frozen: editing them changes everyone's results and
> invalidates prior numbers. Coordinate with the team before touching them.
