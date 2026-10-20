"""
face_utils.py
OpenCV based face detection, sample capturing, LBPH training and recognition.
"""
import cv2
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAINER_PATH = os.path.join(BASE_DIR, "trainer", "trainer.yml")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(os.path.dirname(TRAINER_PATH), exist_ok=True)

# Use the cascade file bundled with the project first (works even if the opencv
# install has no data folder), then fall back to the one shipped with opencv.
_LOCAL_CASCADE = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
FACE_CASCADE = cv2.CascadeClassifier(_LOCAL_CASCADE)
if FACE_CASCADE.empty() and hasattr(cv2, "data"):
    FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if FACE_CASCADE.empty():
    raise RuntimeError(
        "Could not load haarcascade_frontalface_default.xml. Make sure it exists next to "
        "face_utils.py, or reinstall: pip install \"opencv-contrib-python>=4.10,<5\""
    )

SAMPLES_NEEDED = 60
FACE_SIZE = (200, 200)
CONFIDENCE_THRESHOLD = 90  # lower = stricter (LBPH: lower distance = better match). Raise this if real matches show as Unknown; lower it if wrong matches are accepted.
ATTENDANCE_COOLDOWN_FRAMES = 1


def preprocess_face(gray_face):
    """Resize + histogram-equalize so lighting differences between registration
    and live attendance affect matching less."""
    face_resized = cv2.resize(gray_face, FACE_SIZE)
    return cv2.equalizeHist(face_resized)


def detect_faces(gray_frame):
    return FACE_CASCADE.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))


def employee_dataset_dir(employee_id):
    path = os.path.join(DATASET_DIR, str(employee_id))
    os.makedirs(path, exist_ok=True)
    return path


def save_face_sample(employee_id, gray_face, index):
    face_resized = cv2.resize(gray_face, FACE_SIZE)
    path = os.path.join(employee_dataset_dir(employee_id), f"{index}.jpg")
    cv2.imwrite(path, face_resized)


def train_model():
    """Train LBPH recognizer using all images inside dataset/<employee_id>/*.jpg"""
    if not hasattr(cv2, "face"):
        return False, (
            "cv2.face module not found. This means 'opencv-python' and "
            "'opencv-contrib-python' are both installed and conflicting. "
            "Fix: pip uninstall opencv-python opencv-python-headless "
            "opencv-contrib-python opencv-contrib-python-headless -y "
            "then: pip install opencv-contrib-python"
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []

    if not os.path.isdir(DATASET_DIR):
        return False, "No dataset found."

    for employee_id_str in os.listdir(DATASET_DIR):
        emp_dir = os.path.join(DATASET_DIR, employee_id_str)
        if not os.path.isdir(emp_dir):
            continue
        try:
            label = int(employee_id_str)
        except ValueError:
            continue

        for img_name in os.listdir(emp_dir):
            img_path = os.path.join(emp_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.equalizeHist(cv2.resize(img, FACE_SIZE))
            faces.append(img)
            labels.append(label)

    if len(faces) == 0:
        return False, "No face samples available to train."

    recognizer.train(faces, np.array(labels))
    recognizer.save(TRAINER_PATH)
    return True, f"Model trained on {len(faces)} samples."


def load_recognizer():
    if not os.path.exists(TRAINER_PATH):
        return None
    if not hasattr(cv2, "face"):
        return None
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_PATH)
    return recognizer


def recognize_face(recognizer, gray_face):
    """Returns (employee_id, confidence) or (None, confidence) if no recognizer / bad match.
    Lower confidence = better match for LBPH."""
    if recognizer is None:
        return None, None
    face_processed = preprocess_face(gray_face)
    label, confidence = recognizer.predict(face_processed)
    if confidence <= CONFIDENCE_THRESHOLD:
        return label, confidence
    return None, confidence
