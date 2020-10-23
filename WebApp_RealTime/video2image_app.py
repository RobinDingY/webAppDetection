import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os, urllib, cv2
import matplotlib.pyplot as plt
from HandFaceTracker import Hand_Face_Tracker
from HandFaceTracker import MultiHandTracker3D
from utils import drawLinesOnHand
from utils import singleHandDetection
import math
import time
import threading
from PIL import Image
from clf import predict
import io
import tempfile

uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
palm_model_path = "palm_detection_without_custom_op.tflite"
landmark_model_path = "hand_landmark_3d.tflite"
hand_anchors_path = "hand_anchors.csv"

h_detector = MultiHandTracker3D(palm_model_path, landmark_model_path, hand_anchors_path,box_enlarge=1.3,box_shift=0.2)

if uploaded_file is not None:
    # video_bytes = uploaded_file.read() # BytesIO buffer
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    vf = cv2.VideoCapture(tfile.name)
    count = 0
    frame_list = []
    while vf.isOpened():
        ret, frame = vf.read()
        if ret:
            count += 1
            frame_list.append(frame)
        else:
            print("Can't receive frame (stream end?). Exiting ...")
            break
    arr = np.array(frame_list)
    label_classes=[]
    index = 0
    for img_bgr in arr:
        out_img,hasHand = singleHandDetection(img_bgr, h_detector,blank=True,cropped=True)
        if not hasHand:
            st.write("No hands found")
            label_class = 'None'
        else:
            label_class = predict(out_img)
        label_classes.append([index,label_class])
        index += 1

    # out_img= cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    # pil_img = Image.fromarray(out_img)
    
    st.write('total frame: ', count)
    st.write(arr.shape)
    st.write('frame labels display: ',label_classes)
    # cap = cv2.VideoCapture(uploaded_file)
    # frame_list = []
    # while True:
    #     ret, frame = cap.read()
    #     if ret:
    #         frame_list.append(frame)
    #     else:
    #         break   
    st.write('Convert vide to images')
    # st.write('Total pictures: ', len(frame_list))
    