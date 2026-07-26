import streamlit as st
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter


# Load TFLite model
interpreter = Interpreter(
    model_path="model/tomato_model.tflite"
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


class_names = [
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold"
]


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


# Page design
st.set_page_config(
    page_title="Tomato Disease AI",
    page_icon="🌱"
)


st.title("🌱 AI Tomato Disease Detector")

st.write(
    "Upload a tomato leaf image and AI will detect possible diseases."
)


uploaded_file = st.file_uploader(
    "📷 Upload leaf image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=350
    )


    img = image.resize((224,224))

    img_array = np.array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = img_array.astype(np.float32) / 255.0


    # Prediction
    interpreter.set_tensor(
        input_details[0]["index"],
        img_array
    )

    interpreter.invoke()


    prediction = interpreter.get_tensor(
        output_details[0]["index"]
    )


    result = np.argmax(prediction)

    confidence = float(
        np.max(prediction)
    ) * 100


    disease = class_names[result]


    st.subheader("🦠 Result")

    st.success(disease)


    st.subheader("📊 Confidence")

    st.write(
        f"{confidence:.2f}%"
    )


    st.subheader("💡 Solution")

    st.info(
        solutions[disease]
    )
