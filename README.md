# CSC3109-Machine-Learning

Fine-grained aerial image classification over 4 confusable categories
(`coastal_mansion`, `dense_residential`, `nursing_home`, `sparse_residential`)
using Keras/TensorFlow transfer learning.

## Layout

```
dataset/
  set 23/   # FIXED train set, 700 images/class (one subfolder per class)
  val 23/   # FIXED held-out validation set, 100 images/class
shared/     # FROZEN shared foundation (do not edit without notifying the team)
  config.py    # constants + set_seed()
  data.py      # internal tuning split + isolated held-out loader
  augment.py   # get_augmentation() -> training-only augmentation
  evaluate.py  # evaluate(model, val_ds, model_name) -> fixed-schema metrics
models/     # one module per team member, each exposes build_model(...)
results/    # per-model JSON + confusion-matrix PNG + best checkpoint
train.py    # shared training entrypoint (--model selects the member's model)
```

## For team members — how to test your own model

You never run the files in `shared/` directly; they are libraries that
`train.py` imports for you. Model comparison uses a deterministic internal
tuning split derived from `set 23`; `val 23` is loaded only for explicitly
requested smoke or final evaluation.

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
python train.py --model <yourname> --run-name <unique-run-name>
# Final selected run only: add --evaluate-held-out --run-type final
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
- **Don't tune against `val 23`** — it is reserved for smoke verification and
  the final selected model. Training/tuning is derived deterministically from
  `set 23` using the shared seed.
- Only your `models\<yourname>.py` should differ between members.
- Always activate the venv (`.\venv\Scripts\Activate.ps1`) before running.
