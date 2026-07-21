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

### 2. Train your model
```bash
python train.py --model <name> --run-name <name> --evaluate-held-out --run-type final --batch-size 16 --patience 20 
```

### 3. Check the __RESULTS__
`results/`

### 4. Config Docker and deploy UI
To test the deployment of your model ensure the results have a __.keras__

**UPDATE THE DOCKERIGNORE**
```docker
!results/<your_model>.keras
```

Update frontend/**web.py**
```bash
MODEL_PATH = REPO_ROOT / "results" / "<your_model>.keras"
```

**MODIFY THIS IN THE DOCKERFILE**
```bash
COPY results/<your_model>.keras results/<your_model>.keras
```

**Build Image**
```bash
docker build -t your_username/your-repo-name:<version> .
```

**Tag Image**
```bash
docker image tage your_username/your-repo-name:<version> your_username/your-repo-name:<version>
```

**Push to Docker repository**
```bash
docker push your_username/your-repo-name:tag-name
```

## Example
```bash
docker build -t xxjiadexx/custom_cnn:v1.0 .
docker image tag xxjiadexx/custom_cnn:v1.0 xxjiadexx/custom_cnn:v1.0
docker push xxjiadexx/custom_cnn:v1.0
```

## Deploy UI
```bash
docker run --name <container_name> -p 8501:8501 <image_name:version>
```

## Example
```bash
docker run --name custom_cnn -p 8501:8501 xxjiadexx/custom_cnn:v1.0
```