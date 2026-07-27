import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# Page setup
st.set_page_config(
    page_title="Tomato Disease AI",
    page_icon="🌱"
)


# Load AI model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
    "tomato_model.h5",
    compile=False
)
    return model


model = load_model()


# Disease classes
class_names = [
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold"
]


# Solutions
solutions = {
    "Tomato_Early_blight":
    "🍂 Remove infected leaves.\n\nImprove air circulation.\n\nApply suitable treatment.",

    "Tomato_healthy":
    "🌱 Plant looks healthy!\n\nContinue good care.",

    "Tomato_Late_blight":
    "⚠️ Remove infected plants.\n\nAvoid overwatering.\n\nApply treatment.",

    "Tomato_Leaf_Mold":
    "🌫️ Reduce humidity.\n\nImprove ventilation.\n\nRemove infected leaves."
}


# Website title
st.title("🌱 AI Tomato Disease Detector")

st.write(
    "Upload a tomato leaf image and AI will detect possible diseases."
)


# Upload image
uploaded_file = st.file_uploader(
    "📷 Upload tomato leaf image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=350
    )


    # Image preprocessing
    img = image.convert("RGB")

    img = img.resize(
        (224, 224)
    )

    img_array = np.array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = img_array.astype(
        np.float32
    ) / 255.0


    # AI prediction
    prediction = model.predict(
        img_array
    )


    result = np.argmax(
        prediction
    )

    confidence = float(
        np.max(prediction)
    ) * 100


    disease = class_names[result]


    # Display result
    st.subheader("🦠 Result")

    st.success(
        disease
    )


    st.subheader("📊 Confidence")

    st.write(
        f"{confidence:.2f}%"
    )


    st.subheader("💡 Solution")

    st.info(
        solutions[disease]
    )
