import cv2
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt
import streamlit as st

st.title('Facial landmark extraction')
st.text('Work hard... Play hard')

data = st.file_uploader("Upload a video", type=None, accept_multiple_files=True)

# Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()


if data is not None:
    cap = cv2.VideoCapture('/home/becode2/UCL_Face/2022-04-07-105629.webm')
    ret, image = cap.read()
    height, width, channels = image.shape
    name= "out.mp4"
    out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*"mp4v"), 30, (image.shape[0], image.shape[1]), isColor=False)

    while cap.isOpened():
        ret, image = cap.read()
        if ret is not True:
            break
        height, width, channels = image.shape
        # Facial landmarks
        result = face_mesh.process(image)

        canvas = np.zeros((image.shape[1], image.shape[0], 1), dtype = "uint8")
        for facial_landmarks in result.multi_face_landmarks:
            for i in range(0, 468):
                pt1 = facial_landmarks.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)
                cv2.circle(canvas, (x, y), 3, (100, 100, 0), -1)
        out.write(canvas)
   

    cap.release()
    out.release()