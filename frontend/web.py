import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from frontend.inference import MODEL_PATH, load_model, predict_probabilities
from shared.config import CLASS_NAMES


@st.cache_resource
def load_selected_model():
    return load_model(MODEL_PATH)


model, model_error = load_selected_model()

st.title("Aerial Scene Classifier")
st.write(
    "Upload an aerial image and the model predicts one of four categories: "
    + ", ".join(CLASS_NAMES)
    + "."
)

if model is None:
    st.error(
        "The EfficientNet-B0 checkpoint could not be loaded. "
        "Predictions are disabled until the deployment artifact is available."
    )
    with st.expander("Why?"):
        st.code(model_error or "Unknown error")
    st.stop()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", width=420)

    probs = predict_probabilities(model, image)

    top = int(np.argmax(probs))
    st.success(
        f"Prediction: **{CLASS_NAMES[top]}**  "
        f"(maximum softmax score: {probs[top]:.1%})"
    )
    st.subheader("Softmax score per class")
    st.bar_chart(
        {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
        height=240,
    )
