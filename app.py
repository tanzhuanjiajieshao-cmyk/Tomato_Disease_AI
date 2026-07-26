import streamlit as st
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter
import os


# Page settings
st.set_page_config(
    page_title="Tomato Disease AI",
    page_icon="🌱",
    layout="centered"
)


# Title
st.title("🌱 AI Tomato Disease Detector")

st.write(
    "Upload a tomato leaf image and AI will detect possible diseases."
)


# Check model
model_path = "model/tomato_model.tflite"

if not os.path.exists(model_path):
    st.error("❌ Model file not found!")
    st.stop()


# Load TFLite model
interpreter = Interpreter(
    model_path=model_path
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Classes
class_names = [
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold"
]


# Solutions
solutions = {

    "Tomato_Early_blight":
    """
🍂 Early Blight detected

Solutions:
- Remove infected leaves
- Improve air circulation
- Avoid water staying on leaves
- Apply suitable treatment
""",

    "Tomato_healthy":
    """
🌱 Plant looks healthy!

Continue:
- Good watering
- Enough sunlight
- Regular monitoring
""",

    "Tomato_Late_blight":
    """
⚠️ Late Blight detected

Solutions:
- Remove infected plants
- Avoid overwatering
- Improve ventilation
- Apply suitable treatment
""",

    "Tomato_Leaf_Mold":
    """
🌫️ Leaf Mold detected

Solutions:
- Reduce humidity
- Improve airflow
- Remove infected leaves
"""
}


# Upload image
uploaded_file = st.file_uploader(
    "📷 Upload tomato leaf image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Leaf",
        width=350
    )


    # Preprocess image
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


    # Prediction
    interpreter.set_tensor(
        input_details[0]["index"],
        img_array
    )

    interpreter.invoke()


    prediction = interpreter.get_tensor(
        output_details[0]["index"]
    )


    result = np.argmax(
        prediction[0]
    )


    confidence = float(
        np.max(prediction[0])
    ) * 100


    disease = class_names[result]


    # Display result
    st.divider()

    st.subheader("🦠 Result")

    st.success(
        disease
    )


    st.subheader("📊 Confidence")

    st.progress(
        confidence / 100
    )

    st.write(
        f"{confidence:.2f}%"
    )


    st.subheader("💡 Solution")

    st.info(
        solutions[disease]
    )


st.divider()

st.caption(
    "AI Tomato Disease Detection System | Science Expo Project"
)
