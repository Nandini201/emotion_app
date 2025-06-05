import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import io
import requests

# Constants
IMG_SIZE = 48
MODEL_FILE = "emotion_model.keras"
SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png")

st.set_page_config(page_title="Emotion Detection (Pretrained Model Only)")


# Load model
@st.cache_resource(show_spinner=False)
def load_trained_model():
    if os.path.exists(MODEL_FILE):
        model = load_model(MODEL_FILE)
        st.success(f"Loaded model from {MODEL_FILE}")
        return model
    else:
        st.error(f"Model file '{MODEL_FILE}' not found. Please add it to the app folder.")
        return None


# Real face check (anti-spoofing)
def is_real_face(image):
    try:
        # Convert image to bytes
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        buf.seek(0)

        # API call to AI Image Detector
        response = requests.post(
            "https://aiimagedetector.org/api/analyze",
            files={"image": ("image.png", buf, "image/png")}
        )

        if response.status_code == 200:
            result = response.json()
            ai_score = result.get("ai_score", 0)
            # Threshold: If AI score < 0.5 → considered real
            return ai_score < 0.5
        else:
            st.warning("Anti-spoofing API error. Proceeding as real.")
            return True
    except Exception as e:
        st.warning(f"Anti-spoofing check failed: {e}")
        return True


# Image preprocessing
def preprocess_image(img):
    img = img.convert("L")  # Grayscale
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img).reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    return img


# Load model
model = load_trained_model()
label_map = {str(i): i for i in range(model.output_shape[-1])} if model else None

# UI
st.title("Emotion Detection (Using Pretrained Model)")

if model and label_map:
    # Predict from uploaded image
    st.header("Predict Emotion from Uploaded Image")
    uploaded_image = st.file_uploader("Upload a face image", type=SUPPORTED_FORMATS)

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict Emotion"):
            st.info("Checking for spoofing...")
            if not is_real_face(image):
                st.error("AI-generated or spoofed image detected. Cannot proceed with prediction.")
            else:
                input_img = preprocess_image(image)
                preds = model.predict(input_img)
                class_idx = np.argmax(preds)
                confidence = preds[0][class_idx]
                label = str(class_idx)
                st.success(f"Predicted Emotion: **{label}** with confidence {confidence:.2f}")

    # Live webcam
    st.header("Live Webcam Emotion Detection")
    run_webcam = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])

    if run_webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam.")
        else:
            while run_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to capture image")
                    break

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)

                if is_real_face(pil_img):
                    input_img = preprocess_image(pil_img)
                    preds = model.predict(input_img)
                    class_idx = np.argmax(preds)
                    confidence = preds[0][class_idx]
                    label = str(class_idx)

                    cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            cap.release()
else:
    st.info("Please ensure the pretrained model file 'emotion_model.keras' is in the app directory.")
