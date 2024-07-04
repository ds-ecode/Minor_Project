import streamlit as st
import cv2
import numpy as np
import torch
import timm
from torch import nn
from PIL import Image

# FaceModel
class FaceModel(nn.Module):
    def __init__(self):
        super(FaceModel, self).__init__()
        self.eff_net = timm.create_model('efficientnet_b0', pretrained=True, num_classes=7)

    def forward(self, images, labels=None):
        logits = self.eff_net(images)
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return logits, loss
        return logits

# Load Haarcascade for face detection
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Helper function to detect faces
def detect_faces(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9)
    return faces_rect

# Capture face from webcam
def capture_face():
    WEBCAMNUM = 0
    video_capture = cv2.VideoCapture(WEBCAMNUM)
    for i in range(3):
        video_capture.read()
    ret, frame = video_capture.read()
    video_capture.release()
    return frame

# Convert uploaded file to image array
def byte_to_array(image_byte):
    image = Image.open(image_byte)
    image = np.array(image)
    return image

# Prediction function
def predict(image, model, device):
    # Preprocess the image
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))
    image = torch.tensor(image, dtype=torch.float32)
    image = image.unsqueeze(0)

    # Move the image to the device
    image = image.to(device)

    # Make the prediction
    with torch.no_grad():
        output = model(image)

    # Get the predicted class
    predicted_class = output.argmax(dim=1)
    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    # Return the predicted class
    return predicted_class.item(), classes[predicted_class.item()], output

# Variables
DEVICE = "cpu"
model = FaceModel()
model.to(DEVICE)
model.load_state_dict(torch.load("path_to_your_model_weights.pth", map_location=DEVICE))

# Main Streamlit code
st.markdown("""
    # Facial Emotion Recognition APP
    > By Divyansh Sharma And Devansh Gupta
""")

option = st.selectbox("Select file source?", ("Webcam", "Upload Picture"))

if option == "Webcam":
    if st.button('Capture Image'):
        frame = capture_face()
        if frame is not None:
            face_locations = detect_faces(frame)
            for (x, y, w, h) in face_locations:
                face = frame[y:y+h, x:x+w]
                predicted_class, emotion, _ = predict(face, model, DEVICE)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            st.image(frame[:, :, ::-1], caption='Captured Image', use_column_width=True)

elif option == "Upload Picture":
    image_byte = st.file_uploader(label="Select a picture containing faces:", type=['jpg', 'png'])
    if image_byte is not None:
        image_array = byte_to_array(image_byte)
        face_locations = detect_faces(image_array)
        for (x, y, w, h) in face_locations:
            face = image_array[y:y+h, x:x+w]
            predicted_class, emotion, _ = predict(face, model, DEVICE)
            cv2.rectangle(image_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image_array, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        st.image(image_array, caption='Uploaded Image', use_column_width=True)
