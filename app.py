import streamlit as st
import numpy as np
from PIL import Image


# Load TFLite model
import tensorflow as tf

interpreter = tf.lite.Interpreter(
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

    "Tomato_Late_blight":
    "⚠️ Remove infected plants.\n\nAvoid overwatering.\n\nApply treatment.",

    "Tomato_Leaf_Mold":
    "🌫️ Reduce humidity.\n\nImprove ventilation.\n\nRemove infected leaves.",

    "Tomato_healthy":
    "🌱 Plant looks healthy!\n\nContinue good care."
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

    img_array = np.array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = img_array.astype(
        np.float32
    ) / 255.0


    interpreter.set_tensor(
        input_details[0]["index"],
        img_array
    )

    interpreter.invoke()


    prediction = interpreter.get_tensor(
        output_details[0]["index"]
    )


    index = np.argmax(prediction)

    confidence = float(
        np.max(prediction)
    ) * 100


    disease = class_names[index]


    st.subheader("🦠 Result")

    st.success(disease)


    st.metric(
        "📊 Confidence",
        f"{confidence:.2f}%"
    )


    st.subheader("💡 Solution")

    st.info(
        solutions[disease]
    )
