# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages
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
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import queue
# from imutils.video import FPS
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


palm_model_path = "palm_detection_without_custom_op.tflite"
landmark_model_path = "hand_landmark_3d.tflite"
hand_anchors_path = "hand_anchors.csv"
h_detector = MultiHandTracker3D(palm_model_path, landmark_model_path, hand_anchors_path, box_enlarge=1.3, box_shift=0.2)

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to
# warmup
# vs = VideoStream(usePiCamera=1).start()
# src = "http://admin:admin@10.19.80.54:8081/"
src = 0
# caram 摄像头启动
vs = VideoStream(src).start()
# fps = vs.get(cv.CAP_PROP_FPS)
# print(fps)
time.sleep(2.0)


@app.route("/")
def index():
    return render_template("index.html")
    # return render_template("index.html",**templateData)


def detect_motion(frameCount):
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, outputFrame, outputFrame_label_pic, lock
    global label_class

    # initialize the motion detector and the total number of frames
    # read thus far
    # total = 0
    # label_classes = []

    # loop over frames from the video stream
    while True:
        frame = vs.read()
        # 【label—_pic】图片裁剪
        height, width, depth = frame[0:200, 0:400].shape
        img_blank = np.zeros((height, width, depth), np.uint8)
        img_blank.fill(255)  # 255-white 0-black
        frame_label_pic = img_blank
        frame = imutils.resize(frame, width=600)
        frame_label_pic = imutils.resize(frame_label_pic, width=400)

        # 输出裁减的图片，图片手部检测状态判断(T,F），以及手部的点
        out_img, hasHand, hand_keypoints = singleHandDetection(frame, h_detector, blank=True, cropped=True)
        if not hasHand:
            label_class = 'None'
        else:
            # 这里的label_calss 是预测出来的值
            label_class = predict(out_img)
            drawLinesOnHand(frame, hand_keypoints[:, :2])

            # cv2.putText(frame,label_class,(50,150), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
            label_coord = (frame_label_pic.shape[1] // 2 - 80, frame_label_pic.shape[0] // 2 + 30)
            cv2.putText(frame_label_pic, label_class, label_coord, cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)

        
        # 将预测的值存入到一个列表中
        # label_classes.append([total, label_class])
        # total += 1

        # grab the current timestamp and draw it on the frame
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # acquire the lock, set the output frame, and release the
        # lock
        with lock:
            outputFrame = frame.copy()
            outputFrame_label_pic = frame_label_pic.copy()




# fps.update()

def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


def generate_label_pic():
    # grab global references to the output frame and lock variables
    global outputFrame_label_pic

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        # with lock:
        # check if the output frame is available, otherwise skip
        # the iteration of the loop
        if outputFrame_label_pic is None:
            continue

        # encode the frame in JPEG format
        (flag, encodedImage_label_pic) = cv2.imencode(".jpg", outputFrame_label_pic)

        # ensure the frame was successfully encoded
        if not flag:
            continue

        # yield the output frame in the byte format
        yield (b'--frame_label_pic\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage_label_pic) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/label_pic_feed")
def label_pic_feed():
    return Response(generate_label_pic(),
                    mimetype="multipart/x-mixed-replace; boundary=frame_label_pic")


# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
                    help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

    # start a thread that will perform motion detection
    #     for t in range(3):
    t = threading.Thread(target=detect_motion, args=(
        args["frame_count"],))
    t.daemon = True
    t.start()

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)

# release the video stream pointer
# fps.stop()
# 显示手势识别程序的运行时间
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
# 显示大概的FPS值
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
vs.stop()