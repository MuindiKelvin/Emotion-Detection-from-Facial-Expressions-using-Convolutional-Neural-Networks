import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import os
import urllib.request

st.set_page_config(page_title="Emotion Detection App", layout="wide")

# Load the trained model
@st.cache_resource
def load_model_cached():
    return load_model('emotion_detection_model.h5')

model = load_model_cached()

# --- Robust Haar cascade loading ---
# cv2.data.haarcascades is unreliable on some Streamlit Cloud / headless
# OpenCV builds (the 'data' submodule with bundled XML files can be
# missing even though cv2 itself imports fine, causing an AttributeError).
# Bundling the XML file locally, with an automatic one-time download as a
# fallback, avoids depending on that path at all.
CASCADE_FILENAME = "haarcascade_frontalface_default.xml"
CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

@st.cache_resource
def load_face_cascade():
    cascade_path = CASCADE_FILENAME

    try:
        # Prefer a copy already sitting next to app.py (recommended: commit
        # it to the repo so no network call is needed at runtime).
        if not os.path.exists(cascade_path):
            try:
                cascade_path = cv2.data.haarcascades + CASCADE_FILENAME
            except AttributeError:
                cascade_path = None

        if not cascade_path or not os.path.exists(cascade_path):
            # Last resort: download it once and cache locally.
            urllib.request.urlretrieve(CASCADE_URL, CASCADE_FILENAME)
            cascade_path = CASCADE_FILENAME

        classifier = cv2.CascadeClassifier(cascade_path)
    except Exception as e:
        st.error(
            "⚠️ Failed to initialize OpenCV's face detector. This usually means "
            "the `cv2` install on the server is broken — most commonly caused by "
            "having both `opencv-python` and `opencv-python-headless` in "
            "requirements.txt. Keep only `opencv-python-headless`, then use "
            "'Reboot app' (not just redeploy) in Streamlit Cloud's Manage app menu."
        )
        st.exception(e)
        st.stop()

    if classifier.empty():
        st.error("⚠️ Could not load the face detection model (Haar cascade). Please check the app's logs.")
        st.stop()
    return classifier

face_cascade = load_face_cascade()

# Define emotion labels with emojis
emotion_labels = ['Angry 😡', 'Disgust 🤢', 'Fear 😨', 'Happy 😊', 'Neutral 😐', 'Sad 😢', 'Surprise 😲']

# Function to preprocess image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)  # Convert to RGB
        face = np.expand_dims(face, axis=0)  # Add batch dimension
        face = face / 255.0
        return face, (x, y, w, h)
    return None, None

BODY_TEXT_COLOR = "#1b1b1f"

# Animated background with many balloons + explicit text-color fix
st.markdown(
    """
    <style>
    /* Pin a light color scheme so the app never inherits an invisible
       dark-mode text color from the visitor's browser/OS. */
    :root {
        color-scheme: light;
    }

    .stApp {
        background-color: whitesmoke;
        overflow: hidden;
    }

    /* Force every piece of text in the app to a readable dark color */
    .stApp, .stApp p, .stApp span, .stApp div, .stApp h1, .stApp h2, .stApp h3,
    .stApp label, .stApp footer, .stApp footer p {
        color: #1b1b1f !important;
    }

    /* File uploader: force light background + dark text/icons, since the
       dark widget skin otherwise makes this box (and its label above it)
       unreadable, same issue as the number inputs in the other app. */
    [data-testid="stFileUploader"] {
        background-color: #ffffff !important;
        border: 2px dashed #FFB6C1;
        border-radius: 10px;
        padding: 10px;
    }
    [data-testid="stFileUploader"] * {
        color: #1b1b1f !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: #ffffff !important;
    }
    /* The little "filename / size" chip that appears after a file is
       uploaded renders as its own dark widget and wasn't caught by the
       broader [data-testid="stFileUploader"] * rule above. */
    [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploaderFile"] * {
        background-color: #ffffff !important;
        color: #1b1b1f !important;
    }
    [data-testid="stFileUploaderFileName"] {
        color: #1b1b1f !important;
    }
    [data-testid="stBaseButton-secondary"] {
        background-color: #FFB6C1 !important;
        color: #1b1b1f !important;
        border: none !important;
    }

    .balloon {
        width: 60px;
        height: 80px;
        position: fixed;
        background-color: #FFB6C1;
        border-radius: 50%;
        animation: floatBalloons 10s infinite ease-in-out;
        z-index: -1;
    }

    @keyframes floatBalloons {
        0% {
            transform: translate(0, 0);
        }
        100% {
            transform: translate(-200vw, -200vh);
        }
    }

    /* Random balloon positioning with delays */
    .balloon1 { left: -60px; top: -80px; animation-delay: 0s; }
    .balloon2 { right: -60px; top: -80px; animation-delay: 1s; }
    .balloon3 { left: -60px; bottom: -80px; animation-delay: 2s; }
    .balloon4 { right: -60px; bottom: -80px; animation-delay: 3s; }
    .balloon5 { left: 50vw; top: -80px; animation-delay: 4s; }
    .balloon6 { right: 50vw; bottom: -80px; animation-delay: 5s; }
    .balloon7 { left: 25vw; bottom: -80px; animation-delay: 6s; }
    .balloon8 { right: 25vw; top: -80px; animation-delay: 7s; }
    .balloon9 { left: -60px; bottom: 50vh; animation-delay: 8s; }
    .balloon10 { right: -60px; top: 50vh; animation-delay: 9s; }
    </style>

    <!-- Add 10 balloons -->
    <div class="balloon balloon1"></div>
    <div class="balloon balloon2"></div>
    <div class="balloon balloon3"></div>
    <div class="balloon balloon4"></div>
    <div class="balloon balloon5"></div>
    <div class="balloon balloon6"></div>
    <div class="balloon balloon7"></div>
    <div class="balloon balloon8"></div>
    <div class="balloon balloon9"></div>
    <div class="balloon balloon10"></div>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.title('Emotion Detection App 😎')

uploaded_file = st.file_uploader("Choose an image... 📷", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Simulate processing time
    for i in range(100):
        status_text.text(f"Processing: {i+1}% 🔄")
        progress_bar.progress(i + 1)
        time.sleep(0.01)  # Adjust this value to control the speed of the progress bar

    face, rect = preprocess_image(image)

    # Clear the progress bar and status text
    progress_bar.empty()
    status_text.empty()

    if face is not None:
        prediction = model.predict(face)[0]
        emotion = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Draw rectangle and emotion on image
        (x, y, w, h) = rect
        processed_image = image.copy()
        cv2.rectangle(processed_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(processed_image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display images and text on the same grid
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(image, channels="BGR", caption="Uploaded Image 📸")
        with col2:
            st.image(processed_image, channels="BGR", caption="Processed Image 🎨")
        with col3:
            st.write(f"**Detected emotion:** {emotion}")
            st.write(f"**Confidence Score:** {confidence:.2f} 🎯")
            st.write("Note: This is a demo app. For best results, use clear, well-lit images with a single face. 😊")
    else:
        st.image(image, channels="BGR", caption="Uploaded Image")
        st.write("No face detected in the image. 😕")

st.write("**Note: For best results, use clear, well-lit images with a single face. 😊**")

# Copyright information
st.markdown(
    """
    <footer style='text-align: center; margin-top: 20px; color: #1b1b1f;'>
        <p style='color: #1b1b1f;'>© 2025 Kelvin Muindi. All rights reserved.</p>
    </footer>
    """,
    unsafe_allow_html=True
)
