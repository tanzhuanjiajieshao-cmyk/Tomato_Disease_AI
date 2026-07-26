import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.utils import img_to_array


model = tf.keras.models.load_model(
    "model/tomato_model.keras"
)


class_names = [
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold"
]


solutions = {
    "Tomato_Early_blight":
    "🍂 Remove infected leaves.\n\nImprove air circulation.\n\nApply suitable fungicide.",

    "Tomato_Late_blight":
    "⚠️ Remove infected plants.\n\nAvoid overwatering.\n\nApply treatment.",

    "Tomato_Leaf_Mold":
    "🌫️ Reduce humidity.\n\nImprove ventilation.\n\nRemove infected leaves.",

    "Tomato_healthy":
    "🌱 Plant looks healthy!\n\nContinue good watering and sunlight."
}


st.set_page_config(
    page_title="Tomato Disease AI",
    page_icon="🌱"
)


st.title("🌱 AI Tomato Disease Detector")

st.write(
    "Upload a tomato leaf image and let AI detect the disease."
)


uploaded_file = st.file_uploader(
    "📷 Upload Tomato Leaf Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Tomato Leaf",
        width=350
    )


    img = image.resize((224,224))

    img_array = img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = img_array / 255.0


    prediction = model.predict(
        img_array,
        verbose=0
    )


    index = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    disease = class_names[index]


    st.subheader("🦠 Result")

    st.success(
        disease
    )


    st.metric(
        "📊 Confidence",
        f"{confidence:.2f}%"
    )


    st.subheader("💡 Solution")

    st.info(
        solutions[disease]
    )