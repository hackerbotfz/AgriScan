import json
import os

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

# CONFIG
MODEL_PATH       = "agriscan_model.h5"
CLASS_INDEX_PATH = "class_indices.json"
IMG_SIZE         = (224, 224)
TOP_K            = 3

# Short treatment look-up table (extend this for all 38 PlantVillage classes)
TREATMENTS = {
    "Tomato_Late_blight": (
        "🍅 Tomato Late Blight",
        "Apply copper-based fungicide (e.g. Ridomil Gold) immediately. "
        "Remove and destroy infected leaves. Avoid overhead watering. "
        "Ensure good air circulation between plants."
    ),
    "Tomato_Early_blight": (
        "🍅 Tomato Early Blight",
        "Spray with chlorothalonil or mancozeb every 7–10 days. "
        "Mulch around the base to prevent soil splash. "
        "Remove lower infected leaves."
    ),
    "Cassava_Bacterial Blight": (
        "🌿 Cassava Bacterial Blight",
        "Use certified disease-free planting material. "
        "Remove and burn infected stems. "
        "Apply copper oxychloride spray. "
        "Observe a 3-week crop-free period after an outbreak."
    ),
    "Cassava_Brown Streak Disease": (
        "🌿 Cassava Brown Streak Disease",
        "Plant resistant varieties (e.g. NASE 14, Narocass 1). "
        "Use virus-free cuttings from certified sources. "
        "Control whitefly vectors with neem-based insecticide."
    ),
    "Cassava_Green Mottle": (
        "🌿 Cassava Green Mottle",
        "Use resistant cultivars. Remove infected plants. "
        "Control aphid and mite vectors with appropriate insecticides."
    ),
    "Cassava_Mosaic Disease": (
        "🌿 Cassava Mosaic Disease",
        "Use mosaic-resistant varieties (e.g. TMS 30572). "
        "Rogue out and burn infected plants within 4 weeks of planting. "
        "Control whitefly populations."
    ),
    "Pepper__bell___Bacterial_spot": (
        "🌶️ Pepper Bacterial Spot",
        "Apply copper-based bactericide. "
        "Avoid working in fields when wet. "
        "Use disease-free seed and resistant varieties."
    ),
    "Corn_(maize)___Northern_Leaf_Blight": (
        "🌽 Maize Northern Leaf Blight",
        "Apply propiconazole or azoxystrobin fungicide. "
        "Plant resistant hybrids. "
        "Rotate crops – avoid continuous maize."
    ),
    "Corn_(maize)___Common_rust_": (
        "🌽 Maize Common Rust",
        "Plant resistant varieties. "
        "Apply foliar fungicide (mancozeb) at early signs. "
        "Ensure adequate plant spacing."
    ),
    "Grape___Black_rot": (
        "🍇 Grape Black Rot",
        "Apply myclobutanil or captan fungicide before and after rain. "
        "Remove mummified berries and infected canes. "
        "Prune for air circulation."
    ),
    "Apple___Apple_scab": (
        "🍎 Apple Scab",
        "Apply fungicide (captan, myclobutanil) at green tip stage. "
        "Rake and destroy fallen leaves. "
        "Plant scab-resistant apple varieties."
    ),
    "Potato___Late_blight": (
        "🥔 Potato Late Blight",
        "Apply metalaxyl + mancozeb fungicide immediately. "
        "Hill up soil around stems to protect tubers. "
        "Harvest as soon as tops die down."
    ),
    "Potato___Early_blight": (
        "🥔 Potato Early Blight",
        "Apply chlorothalonil or azoxystrobin. "
        "Ensure adequate fertilisation – stressed plants are more susceptible. "
        "Rotate potato with non-Solanaceae crops."
    ),
}
DEFAULT_TREATMENT = (
    "⚠️ Unknown or Healthy",
    "The model has identified this leaf. "
    "If your plant appears unhealthy, consult your nearest agricultural extension officer. "
    "If the leaf looks healthy, no action is required."
)


# HELPER FUNCTIONS

@st.cache_resource(show_spinner="Loading AgriScan model…")
def load_model():
    """Load the trained model once and cache it across sessions."""
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model file `{MODEL_PATH}` not found in the app directory. "
            "Please make sure you uploaded it alongside this script."
        )
        st.stop()
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


@st.cache_data
def load_class_index():
    """Load and invert the class-index JSON saved during training."""
    if not os.path.exists(CLASS_INDEX_PATH):
        st.error(
            f"`{CLASS_INDEX_PATH}` not found. "
            "Make sure you downloaded it from Kaggle after training."
        )
        st.stop()
    with open(CLASS_INDEX_PATH) as f:
        class_indices = json.load(f)
    return {v: k for k, v in class_indices.items()}  # {int: class_name}


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Resize and normalise a PIL image for MobileNetV2 input."""
    img  = pil_image.convert("RGB").resize(IMG_SIZE)
    arr  = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # → (1, 224, 224, 3)


def predict(model, img_array: np.ndarray, idx_to_class: dict, top_k=3):
    """Return top-k (class_name, confidence) pairs."""
    preds       = model.predict(img_array, verbose=0)[0]
    top_indices = np.argsort(preds)[::-1][:top_k]
    return [(idx_to_class[i], float(preds[i])) for i in top_indices]


def get_treatment(class_name: str):
    """Return (display_name, advice) for a predicted class."""
    # Try direct match first, then partial match
    if class_name in TREATMENTS:
        return TREATMENTS[class_name]
    for key, value in TREATMENTS.items():
        if key.lower() in class_name.lower() or class_name.lower() in key.lower():
            return value
    return DEFAULT_TREATMENT


def confidence_color(conf: float) -> str:
    """Green for high confidence, amber for medium, red for low."""
    if conf >= 0.80:
        return "🟢"
    elif conf >= 0.50:
        return "🟡"
    return "🔴"


# STREAMLIT UI

def main():
    # Page config 
    st.set_page_config(
        page_title="AgriScan – Crop Disease Detector",
        page_icon="🌿",
        layout="centered",
    )

    # Header 
    st.markdown(
        """
        <div style='text-align:center; padding: 1rem 0'>
            <h1>🌿 AgriScan</h1>
            <p style='font-size:1.1rem; color:#555'>
                Smart Crop Disease Detector for Nigerian Farmers<br>
                Upload a leaf photo → Get an instant diagnosis and treatment advice
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # Load assets
    model       = load_model()
    idx_to_class = load_class_index()

    # File uploader
    uploaded_file = st.file_uploader(
        "📷 Upload a leaf image (JPG, JPEG, or PNG)",
        type=["jpg", "jpeg", "png"],
        help="Take a clear, close-up photo of the affected leaf in good lighting.",
    )

    if uploaded_file is None:
        st.info(
            "👆 Upload a leaf photo above to get started. "
            "Works with cassava, tomato, maize, potato, pepper, apple, and grape leaves."
        )

        # Show example tip cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**📸 Good photo tips**\n- Clear, close-up shot\n- Good natural lighting\n- Single leaf visible")
        with col2:
            st.markdown("**🌿 Supported crops**\n- Cassava\n- Tomato\n- Maize\n- Potato\n- Pepper\n- Apple\n- Grape")
        with col3:
            st.markdown("**⚡ Instant results**\n- Disease name\n- Confidence score\n- Treatment advice")
        return

    #Display uploaded image
    pil_image = Image.open(uploaded_file)
    col_img, col_info = st.columns([1, 1])

    with col_img:
        st.image(pil_image, caption="Uploaded leaf image", use_container_width=True)

    #Run inference
    with st.spinner("🔍 Analysing your leaf…"):
        img_array   = preprocess_image(pil_image)
        predictions = predict(model, img_array, idx_to_class, top_k=TOP_K)

    top_class, top_conf = predictions[0]
    display_name, advice = get_treatment(top_class)

    #Main result
    with col_info:
        st.markdown("### 🔬 Diagnosis")

        # Confidence colour indicator
        icon = confidence_color(top_conf)
        st.markdown(
            f"<h4 style='margin-bottom:0'>{icon} {display_name}</h4>",
            unsafe_allow_html=True,
        )
        st.progress(top_conf, text=f"Confidence: {top_conf:.1%}")

        if top_conf < 0.50:
            st.warning(
                "⚠️ Confidence is below 50%. Please take a clearer, "
                "closer photo of the affected leaf and try again."
            )

        st.divider()
        st.markdown("### 💊 Recommended Action")
        st.info(advice)

    #Top-3 predictions breakdown
    with st.expander("📊 See full prediction breakdown (Top 3)"):
        for rank, (cls, conf) in enumerate(predictions, 1):
            st.markdown(f"**{rank}. {cls.replace('_', ' ')}**")
            st.progress(conf, text=f"{conf:.1%}")

    #Disclaimer
    st.divider()
    st.caption(
        "⚠️ AgriScan is an AI tool intended to assist farmers. "
        "For severe outbreaks or uncertain results, always consult a qualified "
        "agricultural extension officer in your state."
    )

    st.caption(
        "Built with ❤️ by TechieVersity | Week 4 Challenge | "
        "Model: MobileNetV2 trained on PlantVillage dataset"
    )


if __name__ == "__main__":
    main()