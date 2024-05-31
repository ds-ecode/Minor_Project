import streamlit as st
import cv2
import numpy as np
import face_recognition


def capture_face(video_capture):
    # got 3 frames to auto adjust webcam light
    for i in range(3):
        video_capture.read()
    while(True):
        ret, frame = video_capture.read()
        FRAME_WINDOW = st.image([])
        FRAME_WINDOW.image(frame[:, :, ::-1])
        # face detection
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_small_frame)
        if len(face_locations) > 0:
            video_capture.release()
            return frame


def faces (image):
  img = cv2.imread("download.jpeg") 
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  haar_cascade = cv2.CascadeClassifier('Haarcascade_frontalface_default.xml') 
  faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9) 
  return(faces_rect)
  '''for (x, y, w, h) in faces_rect: 
  	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2) 
   O/P - [[  2  35  70  70]
   [ 60  22  80  80]
   [123  28  83  83]]'''


# main Code

st.markdown("""
    # Facial Emotion Recognition APP
    > By Divyansh Sharma And Devansh Gupta""")

option = st.selectbox(
    "Select file source?",
    ("Webcam", "Upload Picture"))
if option =="Webcam":
  #camera
  WEBCAMNUM = 0
  video_capture = cv2.VideoCapture(WEBCAMNUM)
  frame = capture_face(video_capture)
  FRAME_WINDOW = st.image([])
  FRAME_WINDOW.image(frame)
  
  
if option =="Upload Picture":
  # displays a file uploader widget and return to BytesIO
    image_byte = st.file_uploader(
        label="Select a picture contains faces:", type=['jpg', 'png']
    )
   # detect faces in the loaded image
    max_faces = 0
    rois = []  # region of interests (arrays of face areas)
    if image_byte is not None:
        image_array = byte_to_array(image_byte)
        face_locations = face_recognition.face_locations(image_array)
        for idx, (top, right, bottom, left) in enumerate(face_locations):
            # save face region of interest to list
            rois.append(image_array[top:bottom, left:right].copy())
    bgr_img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    st.image(bgr_img, width=720)
    max_faces = len(face_locations)
  
  






