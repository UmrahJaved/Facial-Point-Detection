from tkinter import Y
import cv2
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt
import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)

title = '<p style="font-family:Courier; color:Black; font-size: 60px; font-weight:bold;">Facial landmarks recognition</p>'
st.markdown(title, unsafe_allow_html=True)
st.text('Work hard... Play hard')

data = st.file_uploader("Upload a video", type=None)

# Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

if data is not None:
    image1 = cv2.imread('data/face.jpg')
    rgb_image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_image)
    
    height, width, _ = image1.shape
    
    # Facial landmarks
    result = face_mesh.process(image1)
    landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
                296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
                380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]
    canvas = np.zeros((image1.shape[1], image1.shape[0], 1), dtype = "uint8")
    d_pxl = 3
    e_indexes = []
    
    with st.container():

        col1,col2 = st.columns(2)

        with col1:
            st.image(rgb_image)
            
        with col2:

            for facial_landmarks in result.multi_face_landmarks:
                for i in landmark_points_68 :
                    pt1 = facial_landmarks.landmark[i]
                    x = int(pt1.x * width)
                    y = int(pt1.y * height)
                    cv2.circle(canvas, (x, y), d_pxl, (100, 100, 0), -1)                
            plt.imshow(canvas)
            fig = plt.show()
            st.pyplot(fig)
    
    with st.container():
        for facial_landmarks in result.multi_face_landmarks:
            for i in landmark_points_68 :
                pt1 = facial_landmarks.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height) 
                cv2.putText(canvas, str(i), (x,y), 0, 0.3, (255,255,255))             
        plt.imshow(canvas)
        fig = plt.show()
        st.pyplot(fig)

    with st.container():
        video_path = f'./data/{data.name}'
        print('video path: ', video_path)
        cap = cv2.VideoCapture(video_path)
        ret, image2 = cap.read()
        height, width, channels = image2.shape
        name= "out.mp4"
        out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*"mp4v"), 30, (image2.shape[0], image2.shape[1]), isColor=False)
        
        while cap.isOpened():
            ret, image3 = cap.read()
            if ret is not True:
                break
            height, width, channels = image3.shape
            # Facial landmarks
            print('img ', image3.shape)
            result = face_mesh.process(image3)
            landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
                    296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
                    380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]

            canvas = np.zeros((image3.shape[1], image3.shape[0], 1), dtype = "uint8")
            if not result.multi_face_landmarks:
                print("landmark NOT found")
                continue            
            print("landmark found")
            for facial_landmarks in result.multi_face_landmarks: 
                for i in landmark_points_68 :
                    pt1 = facial_landmarks.landmark[i]
                    x = int(pt1.x * width)
                    y = int(pt1.y * height)
                    cv2.circle(canvas, (x, y), 3, (100, 100, 0), -1)
            out.write(canvas)
    
        cap.release()
        out.release()