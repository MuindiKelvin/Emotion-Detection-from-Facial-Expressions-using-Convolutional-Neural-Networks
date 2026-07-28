# Emotion Detection App 😎

A Streamlit web app that detects a face in an uploaded photo and predicts the
person's emotion using a trained Keras/TensorFlow CNN model.

## Features

- 📷 Upload a JPG/JPEG/PNG image
- 🙂 Automatic face detection with OpenCV's Haar cascade classifier
- 🧠 Emotion classification into 7 categories: **Angry, Disgust, Fear, Happy,
  Neutral, Sad, Surprise**
- 🎯 Confidence score for the predicted emotion
- 🎨 Side-by-side view of the original and annotated (bounding box + label)
  image
- 🎈 Lightweight animated UI with a forced light color scheme so text stays
  readable regardless of the visitor's system theme
- 🛡️ Robust cascade-file loading with a local-file → `cv2.data` →
  auto-download fallback chain, plus diagnostics if OpenCV itself is broken

## Demo

1. Open the app.
2. Upload a clear, well-lit photo containing a single face.
3. Wait for the progress bar to finish processing.
4. View the detected emotion, confidence score, and annotated image.

## Requirements

- Python 3.9+
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/) (`opencv-python` **or**
  `opencv-python-headless` — not both, see Troubleshooting below)
- [TensorFlow](https://www.tensorflow.org/) / Keras
- NumPy

### `requirements.txt`

```
streamlit
opencv-python-headless
tensorflow
numpy
```

> ⚠️ Do **not** list both `opencv-python` and `opencv-python-headless` in the
> same requirements file — having both installed is a common cause of a
> broken `cv2` import on Streamlit Cloud and other headless servers.

## Installation

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
```

## Model file

This app expects a trained Keras model file named:

```
emotion_detection_model.h5
```

placed in the same directory as `app.py`. The model should:

- Accept input images of shape `(48, 48, 3)` (RGB, resized from a grayscale
  face crop)
- Output a 7-class probability distribution corresponding to the labels
  `['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']`

If you don't have a trained model, you can train one on a dataset such as
[FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) and export it
with `model.save('emotion_detection_model.h5')`.

## Face detection cascade file

The app uses OpenCV's `haarcascade_frontalface_default.xml` for face
detection. To avoid depending on `cv2.data.haarcascades` (which is
unreliable on some Streamlit Cloud / headless builds), it looks for the
cascade in this order:

1. A local copy named `haarcascade_frontalface_default.xml` next to `app.py`
   (recommended — commit this file to your repo)
2. The cascade bundled with your local OpenCV install
   (`cv2.data.haarcascades`)
3. A one-time automatic download from the OpenCV GitHub repository, cached
   locally afterward

For faster, network-independent startup, download the file once and commit
it to your repo:

```bash
curl -o haarcascade_frontalface_default.xml \
  https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
```

## Usage

Run the app locally with:

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`) in
your browser.

## Project structure

```
.
├── app.py                                 # Main Streamlit application
├── emotion_detection_model.h5             # Trained Keras model (not included — add your own)
├── haarcascade_frontalface_default.xml    # Face detection cascade (optional, auto-downloaded if missing)
├── requirements.txt
└── README.md
```

## Troubleshooting

**"Failed to initialize OpenCV's face detector"**
This usually means the `cv2` install is broken. Common causes:
- Both `opencv-python` and `opencv-python-headless` are in
  `requirements.txt` — keep only one (prefer `opencv-python-headless` for
  server/cloud deployments).
- A local file or folder named `cv2` in your repo is shadowing the real
  installed package.
The app prints diagnostic info (`cv2.__file__`, `cv2.__version__`, whether
`CascadeClassifier` is available) to help pinpoint the cause.

**"No face detected in the image"**
Try a clearer, well-lit, front-facing photo with a single visible face.

**Slow first run**
If the cascade file has to be downloaded automatically, the very first run
will take slightly longer. Subsequent runs use the cached local copy.

## Notes

- This is a demo application; accuracy depends entirely on the quality of
  the trained model and the input image.
- Only the first detected face in an image is analyzed.

## License

© 2026 Kelvin Muindi. All rights reserved.
