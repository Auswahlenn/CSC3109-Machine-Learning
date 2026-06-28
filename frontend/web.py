import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from shared.config import CLASS_NAMES, IMAGE_SIZE  


MODEL_PATH = REPO_ROOT / "results" / "resnet50_best.keras"


@st.cache_resource
def load_model():
    """Load the model once; return (model, error). error is None on success."""
    try:
        from tensorflow import keras

        return keras.models.load_model(MODEL_PATH, compile=False), None
    except Exception as exc:  
        return None, str(exc)


model, model_error = load_model()

st.title("Aerial Scene Classifier 🛰️")
st.write(
    "Upload an aerial image and the model predicts one of four categories: "
    + ", ".join(CLASS_NAMES)
    + "."
)

if model is None:
    st.warning(
        "⚠️ **UI preview mode** — the model could not be loaded, so predictions "
        "below are placeholders. Once the model loads, real predictions appear."
    )
    with st.expander("Why?"):
        st.code(model_error or "Unknown error")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if model is not None:
        x = np.expand_dims(np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)), "float32"), 0)
        probs = model.predict(x)[0]
    else:
        probs = np.full(len(CLASS_NAMES), 1.0 / len(CLASS_NAMES))

    top = int(np.argmax(probs))
    st.success(f"Prediction: **{CLASS_NAMES[top]}**  ({probs[top]:.1%} confidence)")
    st.subheader("Confidence per class")
    st.bar_chart({CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))})
