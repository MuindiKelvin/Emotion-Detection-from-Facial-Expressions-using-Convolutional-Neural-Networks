import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

# Load the trained model
@st.cache_resource
def load_model_cached():
    return load_model('emotion_detection_model.h5')

model = load_model_cached()

# Define emotion labels with emojis
emotion_labels = ['Angry 😡', 'Disgust 🤢', 'Fear 😨', 'Happy 😊', 'Neutral 😐', 'Sad 😢', 'Surprise 😲']

# Function to preprocess image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
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

# Animated background with many balloons
st.markdown(
    """
    <style>
    .stApp {
        background-color: whitesmoke;
        overflow: hidden;
    }

    .balloon {
        width: 60px;
        height: 80px;
        position: fixed;
        background-color: #FFB6C1;
        border-radius: 50%;
        animation: floatBalloons 10s infinite ease-in-out;
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
    <footer style='text-align: center; margin-top: 20px;'>
        <p>© 2025 Kelvin Muindi. All rights reserved.</p>
    </footer>
    """,
    unsafe_allow_html=True
)
