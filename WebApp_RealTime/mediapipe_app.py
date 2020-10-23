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

palm_model_path = "palm_detection_without_custom_op.tflite"
landmark_model_path = "hand_landmark_3d.tflite"
hand_anchors_path = "hand_anchors.csv"

h_detector = MultiHandTracker3D(palm_model_path, landmark_model_path, hand_anchors_path,box_enlarge=1.3,box_shift=0.2)


file_up = st.file_uploader("Upload an image", type="jpg")

if file_up is not None:
    st.write(file_up)
    pil_image = Image.open(file_up)
    img_rgb = np.array(pil_image) 
    img_bgr = img_rgb[:,:,::-1]

    out_img,hasHand = singleHandDetection(img_bgr, h_detector,blank=True,cropped=True)
    if not hasHand:
        st.write("No hands found")
        label_class = 'None'
    else:
        label_class = predict(out_img)

    out_img= cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(out_img)

    st.image(pil_img, caption='Uploaded Image.', use_column_width=True)
    st.write("Hand label class is ",label_class)