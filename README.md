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

## For team members — how to test your own model

You never run the files in `shared/` directly; they are libraries that
`train.py` imports for you. You only write **one** file (your model) and run
**one** command.

### 1. One-time setup

```powershell
# from the repo root: C:\Github\CSC3109-Machine-Learning
py -3.12 -m venv venv           # MUST be Python 3.12 — TensorFlow has no 3.13/3.14 wheel
.\venv\Scripts\Activate.ps1     # prompt should now show (venv)
pip install -r requirements.txt
```

> If a `venv\` from an older Python already exists, delete it first:
> `Remove-Item -Recurse -Force venv`.

### 2. Write your model

Copy the example (don't edit the original) and change only the backbone:

```powershell
copy models\example_resnet50.py models\<yourname>.py
```

In `models\<yourname>.py`, swap the backbone and its matching
`preprocess_input` (e.g. `EfficientNetB0`, `MobileNetV2`). Keep the function
signature exactly:

```python
def build_model(num_classes: int, augmentation: keras.Sequential) -> keras.Model
```

It must return a **compiled** model that stacks, in order: the shared
`augmentation` (passed in), your backbone's own `preprocess_input`, the
pretrained backbone, and a new softmax head. The shared loader returns
**raw [0, 255] pixels** on purpose — applying your backbone's `preprocess_input`
inside your model is what lets every member use a different backbone while
sharing identical data and evaluation.

### 3. Run it

From the repo root, with the venv active. Use the filename **without** `.py`
or the `models\` prefix:

```powershell
python train.py --model <yourname>
# optional: python train.py --model <yourname> --epochs 30 --patience 5
```

### 4. Collect your results

Three files appear in `results\`, named after your model:

| File | Contents |
|------|----------|
| `<yourname>.json` | accuracy, precision, recall, F1 (macro + per-class) |
| `<yourname>_confusion_matrix.png` | confusion matrix |
| `<yourname>_best.keras` | best checkpoint (highest val accuracy) |

### Rules (so everyone's numbers are comparable)

- **Don't edit anything in `shared/`** — it's frozen; changing it invalidates
  everyone's prior results.
- **Don't re-split the data** — the `set 23` / `val 23` split is fixed.
- Only your `models\<yourname>.py` should differ between members.
- Always activate the venv (`.\venv\Scripts\Activate.ps1`) before running.
