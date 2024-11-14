import cv2
import streamlit as st
import numpy as np
from PIL import Image
import io
import os
from tensorflow.keras.models import load_model

# Suppress oneDNN warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Model paths
face_proto = "deploy.prototxt"
face_model = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
age_proto = "age_deploy.prototxt"
age_model = "age_net.caffemodel"
gender_proto = "gender_deploy.prototxt"
gender_model = "gender_net.caffemodel"
emotion_model_path = r"C:\Users\Aryan Negi\Desktop\Project\emotion_model.hdf5"  # Updated path for the uploaded model

# Load models with error handling
try:
    face_net = cv2.dnn.readNet(face_proto, face_model)
    age_net = cv2.dnn.readNet(age_proto, age_model)
    gender_net = cv2.dnn.readNet(gender_proto, gender_model)
except Exception as e:
    st.error(f"Error loading face, age, or gender models: {e}")

# Attempt to load the emotion model
try:
    emotion_model = load_model(emotion_model_path, compile=False)
except Exception as e:
    st.error(f"Error loading emotion model: {e}")
    emotion_model = None  # Set emotion_model to None if loading fails

# Constants for age, gender, and emotion detection
AGE_LIST = ['(0-3)', '(4-9)', '(10-14)', '(15-20)','(21-26)', '(27-35)', '(36-47)', '(48-59)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
EMOTION_LIST = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_face_age_gender_emotion(img):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")

            # Extract the face region
            face = img[y:y1, x:x1]

            # Predict gender
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = GENDER_LIST[gender_preds[0].argmax()]

            # Predict age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = AGE_LIST[age_preds[0].argmax()]

            # Predict emotion
            if emotion_model is not None:
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (64, 64)) / 255.0
                face_reshaped = np.reshape(face_resized, (1, 64, 64, 1))
                emotion_preds = emotion_model.predict(face_reshaped)
                emotion = EMOTION_LIST[np.argmax(emotion_preds)]
            else:
                emotion = "Emotion model not loaded"

            # Draw bounding box and labels
            cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 2)
            label = f"{gender}, {age}, {emotion}"
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return img

# Streamlit App
st.title("Face, Age, Gender, and Emotion Detection")

# Input type selection
option = st.selectbox("Choose input type", ("Photo", "Live Video"))

if option == "Photo":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img = np.array(Image.open(io.BytesIO(uploaded_file.read())))
        result_img = detect_face_age_gender_emotion(img)
        st.image(result_img, caption="Processed Image", use_container_width=True)

elif option == "Live Video":
    stframe = st.empty()
    stop_button = st.checkbox("Stop Live Video")

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        st.error("Unable to access webcam. Please check if it is connected and permissions are granted.")

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            st.error("Could not access webcam.")
            break

        result_frame = detect_face_age_gender_emotion(frame)
        stframe.image(result_frame, channels="BGR", use_container_width=True)

        # Check stop condition
        if stop_button:
            break
    
    video_capture.release()
