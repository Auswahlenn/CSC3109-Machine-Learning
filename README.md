# CSC3109-Machine-Learning

Fine-grained aerial image classification over 4 confusable categories
(`coastal_mansion`, `dense_residential`, `nursing_home`, `sparse_residential`)
using Keras/TensorFlow.

## Layout

```
dataset/
  set 23/   # FIXED train set, 700 images/class (one subfolder per class)
  val 23/   # FIXED held-out test set, 100 images/class
shared/     # FROZEN shared foundation (do not edit without notifying the team)
  config.py    # constants + set_seed()
  data.py      # internal tuning split + isolated held-out loader
  augment.py   # get_augmentation() -> training-only augmentation
  evaluate.py  # evaluate(model, val_ds, model_name) -> fixed-schema metrics
models/     # one module per team member, each exposes build_model(...)
results/    # per-model JSON + confusion-matrix PNG + <run-name>_best.keras
train.py    # shared training entrypoint (--model selects the member's model)
```

> The data folder is `dataset/`. If yours lives elsewhere, set
> the `CSC3109_DATA_DIR` environment variable instead of renaming folders.

## For team members — how to test your own model

You never run the files in `shared/` directly; they are libraries that
`train.py` imports for you. Model comparison uses a deterministic internal
tuning split derived from `set 23`; `val 23` is loaded only for explicitly
requested smoke or final evaluation.

> **Run everything from the integrated `main` branch.** It contains the agreed
> baseline (`IMAGE_SIZE = 256`, shared augmentation, isolated tuning/held-out
> loaders, and `--batch-size` support).

### 1. One-time setup

```powershell
# from the repo root: C:\Github\CSC3109-Machine-Learning
py -3.12 -m venv .venv          # MUST be Python 3.12
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
```

> If `.venv\` already exists, reuse it after installing the current
> requirements instead of creating a second environment.

### 2. Train your model

Everyone uses the SAME flags — batch size and patience change results, so they
must not vary between members.

```bash
python train.py --model <name> --run-name <name> --evaluate-held-out --run-type final --batch-size 16 --patience 20 --epochs 32
```

`<name>` must match your model's filename stem exactly:

| Model            | `--model` / `--run-name` |
| ---------------- | ------------------------ |
| Custom CNN       | `custom_cnn`             |
| ResNet-50        | `resnet50`               |
| EfficientNet-B0  | `efficientnet_b0`        |
| MobileNetV2      | `mobilenetv2`            |
| Vision Transformer | `ViT`                  |

> `ViT` is CASE-SENSITIVE on WSL/Linux — `--model vit` will fail there even
> though it works on Windows.

**If `--batch-size 16` runs out of memory, tell the team — do NOT lower it on
your own.** Batch size materially changes results, so it has to stay identical
for everyone.

### 3. Check the __RESULTS__

In `results/` you should have:

| File                                  | What it is                                     |
| ------------------------------------- | ---------------------------------------------- |
| `<name>.json`                         | **held-out `val 23` metrics — the leaderboard number** |
| `<name>_confusion_matrix.png`         | **held-out confusion matrix — use this one**   |
| `<name>_tuning.json` / `_tuning_*.png`| tuning-split metrics (diagnostics only)        |
| `<name>_run.json`                     | run metadata (verify `"image_size": 256`, `"batch_size": 16`) |
| `<name>_best.keras`                   | best-epoch checkpoint (gitignored — not pushed)|

Report the **held-out** numbers (`accuracy` + `macro.f1`), never the `_tuning`
ones — the tuning split was used to pick the checkpoint, so it is optimistically
biased.

Plot your training curves:

```bash
python plot_history.py --run-name <name>
```

### 4. Push your results

Push the contents of `results/` to `dev`. Pull before you push — the whole team
shares the branch. `*.keras` is gitignored (files are ~300 MB), so only the
JSON/PNG get committed.

### 5. Build and run the final Docker UI

The checked-in deployment configuration consistently targets the selected
`results/efficientnet_b0_best.keras` checkpoint in `.dockerignore`,
`dockerfile`, and `frontend/inference.py`.

**Build Image**

```bash
docker build -t your_username/your-repo-name:<version> .
```

**Push to Docker repository** (the tag must match what you built — pushing with
no tag looks for `:latest`, which will not exist)

```bash
docker login
docker push your_username/your-repo-name:<version>
```

Optionally also publish a `latest` tag (tag to a DIFFERENT name, then push it):

```bash
docker tag your_username/your-repo-name:<version> your_username/your-repo-name:latest
docker push your_username/your-repo-name:latest
```

## Deploy UI

```bash
docker run --name <container_name> -p 8501:8501 <image_name:version>
```

Then open <http://localhost:8501> and upload an aerial image to confirm the
model loads correctly.
