#!/usr/bin/env python
# coding: utf-8

# 已转为py文件

# In[112]:


from google.colab import files

from IPython.display import HTML
from base64 import b64encode

import cv2
import argparse
import os

import numpy as np

import io
import base64
from IPython.display import HTML

class convertVideoToImages:

    def __init__(self,InputVideo_path,OutputImages_path,MergedImages_path,MergedVideo_path,interval,fps):
        
        #视频 -> 图片
        self.InputVideo_path = InputVideo_path      #待切割的视频路径
        self.OutputImages_path = OutputImages_path  #切割为图片后的保存路径 OutputImages_path = 'content/SavedImages/'
        
        #图片 -> 视频
        self.MergedImages_path = MergedImages_path  #待合并的图片的路径
        self.MergedVideo_path = MergedVideo_path    #合并为视频的保存路径 MergedVideo_path = 'content/SavedMergedVideo.avi'
        
        #视频 -> 图片
        self.interval = interval # 帧间隔30 = 1s  当default = 30时，每一秒截取一张图片
        #图片 -> 视频
        self.fps = fps  #24视频每秒24帧 #每秒传输帧数(Frames Per Second)
   
    
    def displayVideo(self):
        mp4 = open(self.MergedVideo_path,'rb').read()
        data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
        HTML("""
        <video width=400 controls>
              <source src="%s" type="video/mp4">
        </video>
        """ % data_url)


    def playVideo(self):
        video = io.open(self.MergedVideo_path, 'r+b').read()
        encoded = base64.b64encode(video)
        return HTML(data='''<video alt="test" controls>
                        <source src="data:video/mp4;base64,{0}" type="video/mp4"/>
                    </video>'''.format(encoded.decode('ascii')))
    
    def convert_Video_to_Images(self):

        if not os.path.exists(self.OutputImages_path):
            os.makedirs(self.OutputImages_path)

        cap = cv2.VideoCapture(self.InputVideo_path)
        #num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        expand_name = '.jpg'
        if not cap.isOpened():
            print("Please check the path.")
    
        cnt = 0
        count = 0
        while 1:
            ret, frame = cap.read()
            cnt += 1
            #  how many frame to cut
            if cnt % self.interval == 0:   # 默认1秒30帧，间隔多少帧取一张图片 当default = 30时，每一秒截取一张图片
                count += 1
                cv2.imwrite(os.path.join(self.OutputImages_path, str('%05d'%count) + expand_name), frame)
                #print("number " ,count)
    
            if not ret:
                break
        print("The total number of images are saved is " , count)

        
    
    def convert_Images_to_Video(self):
 
        #path = 'content/SavedImages/'
        if not os.path.exists(self.MergedImages_path):
            os.makedirs(self.MergedImages_path)


        filelist = os.listdir(self.MergedImages_path)


        eg = cv2.imread(self.MergedImages_path+'00001.jpg',cv2.IMREAD_UNCHANGED)
        height , width , layers =  eg.shape
        size = (width, height)  #需要转为视频的图片的尺寸 可以使用cv2.resize()进行修改

        #self.fps = 24 #24视频每秒24帧 #每秒传输帧数(Frames Per Second)
    
        video = cv2.VideoWriter(self.MergedVideo_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), self.fps, size)
        #video = cv2.VideoWriter( self.MergedVideo_path, cv2.VideoWriter_fourcc('M','J','P','G'), self.fps, size)
        #用图片生成avi格式的视频保存在选定的目录下
        # 例 MergedVideo_path = "content/VideoTest1.avi"

        
        for i,item in enumerate(filelist):
            # print(i,item)
            # if item.endswith('.jpg'): #找到路径中所有后缀名为.jpg的文件，可以更换为.png或其它
            item = self.MergedImages_path + str('%05d'%i) +'.jpg'
            img = cv2.imread(item)
            #print(i)
            video.write(img)

        print("The video is successfully merged.")
        
        video.release()
        cv2.destroyAllWindows()

