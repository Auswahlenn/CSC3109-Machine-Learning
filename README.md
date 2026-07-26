# CSC3109-Machine-Learning

Fine-grained aerial image classification over 4 confusable categories
(`coastal_mansion`, `dense_residential`, `nursing_home`, `sparse_residential`)
using Keras/TensorFlow.

## Layout

```
dataset/      # NOT committed, ~260 MB, NOT submitted (provided by the module)
  set 23/       # FIXED train set, 700 images/class (one subfolder per class)
  val 23/       # FIXED held-out test set, 100 images/class

docs/         # the report deliverable
  final-report.tex      # integrated final report (LaTeX source)
  final-report.pdf      # compiled report -> renamed to T23.pdf on submission
  titlepage.tex, includes.tex, notation.tex   # shared report formatting
  mybib.bib             # bibliography
  figures/              # EDA samples, curves, confusion matrices, slide charts,
                        #   deployment screenshot
  IMPLEMENTATION_SUMMARY.md   # verified implementation + deployment status

shared/       # FROZEN shared foundation (do not edit without notifying the team)
  config.py     # constants + set_seed()
  data.py       # internal tuning split + isolated held-out loader
  augment.py    # get_augmentation() -> training-only augmentation
  evaluate.py   # evaluate(model, val_ds, model_name) -> fixed-schema metrics

models/       # one subfolder per team member, each exposing build_model(...)
  custom_cnn/, resnet50/, EfficientNet-B0/, mobilenetv2/, vit/

results/      # per-model JSON + confusion-matrix PNG (+ gitignored checkpoints)

frontend/     # the containerised inference UI
  inference.py  # model loading, preprocessing, validated prediction (UI-free)
  web.py        # Streamlit presentation layer only

tests/        # contract tests: data split, training args, deployment wiring
scripts/      # experiment runners, manifest, and figure-generation utilities
train.py      # shared training entrypoint (--model selects the member's model)
Dockerfile    # serves results/efficientnet_b0_best.keras on port 8501
```

> The data folder is `dataset/`. If yours lives elsewhere, set
> the `CSC3109_DATA_DIR` environment variable instead of renaming folders.

What each part of the repository is there to show:

| Part                        | Why                                                   |
| --------------------------- | ----------------------------------------------------- |
| `docs/`                     | LaTeX source, `figures/`, bibliography — shows provenance |
| `models/`                   | the five architectures, one per member (§ *Investigation of deep learning approaches*) |
| `shared/`, `train.py`, `plot_history.py` | the common pipeline every model was trained through |
| `frontend/`, `Dockerfile`, `.dockerignore`, `requirements.txt` | containerisation + deep learning inference evidence |
| `results/` (JSON + PNG)     | the held-out numbers quoted in the report              |
| `tests/`, `scripts/`        | reproducibility: split integrity and figure generation |
| `README.md`                 | how a marker reruns any of it                          |

## Containerization and Deployment

EfficientNet-B0 is the selected model. Pull the image:

```bash
docker pull 4a6b3b/csc3109-efficientnet-b0_best.keras
```

Run it:

```bash
docker run -p 8501:8501 4a6b3b/csc3109-efficientnet-b0_best.keras
```

Then open <http://localhost:8501> and upload an aerial image to confirm the
model loads correctly.

### Deploying the other models

Every architecture was published as its own image. Each is built from the same
`Dockerfile` and Streamlit UI, so the pull/run commands above work unchanged —
substitute the reference below.

| Approach                   | Docker Hub reference                        | Held-out macro-F1 |
| -------------------------- | ------------------------------------------- | ----------------- |
| EfficientNet-B0 (deployed) | `4a6b3b/csc3109-efficientnet-b0_best.keras` | **0.9775**        |
| ResNet-50                  | `darylchk/resnet50-aerial`                  | 0.9750            |
| ViT-B/16                   | `enteringthefray/vit`                       | 0.9652            |
| MobileNetV2                | `daddycation23/csc3109-mobilenetv2`         | 0.9576            |
| Custom CNN                 | `xxjiadexx/custom_cnn`                      | 0.8452            |

Only one container can bind port 8501 at a time — stop the running one, or map
a different host port (`-p 8502:8501`), before starting a second model.
