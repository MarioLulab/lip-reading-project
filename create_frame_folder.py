import cv2
import os
import dlib
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt
import copy
from scipy import optimize
import json
import time
from torch.autograd import Variable


class Video_point():
    def __init__(self, File_Path, File_Path2, video_path, landmarks_data_path, allpicture, sampletimes=0, frameFrequency=1):
        # input : File_Path             用于对视频流分割、降噪、滤波的工作文件夹
        #         File_Path2            储存唇部图片的工作文件夹
        #         video_path            要进行唇语识别的目标视频流路径
        #         landmarks_data_path   dlib人脸检测识别库
        #         frameFrequency        提取帧的频率。每frameFrequency帧提取一帧来进行特征点识别
        #         allpicture            关键帧的总长度

        self.times = 0                                             # 初始化计时次数
        self.allpicture = allpicture                               # 关键帧的固定总长度
        self.sampletimes = sampletimes                             # dlib的detector的采样倍数。sampletimes越高，识别精度越高，但运行得也越慢，经过实验，发现选择2已经几乎可以保证100%的提取特征点的正确率了
        self.detector = dlib.get_frontal_face_detector()           # 创建人脸检测detector
        self.predictor = dlib.shape_predictor(landmarks_data_path) # 创建predictor
        self.index = []                                            # 初始化关键帧下标序列
        self.frameFrequency = frameFrequency                       # 每frameFrequency帧从video从提取一帧
    
    def process(self):
        
        # 获取当前文件目录
        if not os.path.exists(File_Path):  # 检查是否存在文件夹
            os.makedirs(File_Path)  # 不存在新建
        if not os.path.exists(File_Path2):  # 检查是否存在文件夹
            os.makedirs(File_Path2)  # 不存在新建

        cap = cv2.VideoCapture(video_path)  # 创建cap对象
        
        while True:
            self.times += 1
            res, image = cap.read()         #若成功读取帧，则res=True。每次循环，cap.read()读到并返回的帧image都不同
            if not res :
                print('not res , not image')
                break

            if self.times % self.frameFrequency == 0:
                cv2.imwrite(File_Path + str(self.times) + '.jpg', image)  # 保存原始视频帧

        # 当while循环结束时，所有的原始视频帧都已得到保存。原始视频帧的最大帧数记为max_index
        max_index = self.times - 1

        # 生成关键帧下标序列
        random_factor = np.random.uniform(-1,1,self.allpicture)  # 生成随机数因子
        for i in range(0, self.allpicture):
            num = max_index/self.allpicture * (i + random_factor[i-1]/2)
            int_num = int(np.round(num))    # 将抽取的关键帧下标结果四舍五入并强制转换为整型
            self.index.append(int_num)

        self.index = np.clip(self.index,1,max_index-1)       # 将关键帧下标序列index限制在1~max_index-1之间
        
        p_index = 0     # 创建序号，用于记录被提取的帧的索引（从0开始，方便创建数据集）
        # 抽取视频关键帧
        for index_i in self.index:
            img = cv2.imread(File_Path + str(index_i) + '.jpg', cv2.IMREAD_GRAYSCALE)  # 读取视频帧
            result = cv2.medianBlur(img,5)  # 中值滤波

            cv2.imwrite(File_Path + str(index_i) + 'dist.jpg', result)  # 保存滤波后图片

            # 人脸检测
            faces = self.detector(result, self.sampletimes)

            pos_68 = []
            
            if (len(faces)>0):
                landmarks = np.matrix([[p.x, p.y] for p in self.predictor(result, faces[0]).parts()]) # result 为滤波后的结果
                # print("提取成功!")
                for idx, point in enumerate(landmarks):
                    pos = (point[0, 0], point[0, 1])

                    pos_68.append(pos)

                # 提取人脸检测标注到的第49，51，53，55和58个点，用来划分嘴唇区域
                pos_49to58 = []

                for s in [48,50,52,54,57]:
                    pos_49to58.append(pos_68[s][0])
                    pos_49to58.append(pos_68[s][1])

                a = np.array(pos_49to58)    # a : [ 1, 5*2 ]。将matrix变为array

                # 求出嘴唇中心点centre
                total_x = 0.
                total_y = 0.
                for s in range(0, 5):
                    total_x += a[2*s]
                    total_y += a[2*s+1]
                centre_x = total_x / 5
                centre_y = total_y / 5
                width = a[6] - a[0]      # 取嘴唇最宽两点作差，求得嘴唇宽度
                height = a[9] - np.min([a[3],a[5]])  # 求得嘴唇的高度

                # 确定截取嘴唇的方框
                x0 = (int)(centre_x - width * 0.75)
                x1 = (int)(centre_x + width * 0.75)
                y0 = (int)(centre_y - height * 0.75)
                y1 = (int)(centre_y + height * 0.75)

                img_cropped = result[y0:y1, x0:x1]                              # 切片给出的坐标为需要裁剪的图片在原图片上的坐标，顺序为[y0:y1, x0:x1]，其中原图的左上角是坐标原点
                img_resized = cv2.resize(img_cropped,(256,256),interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(File_Path2 + str(p_index) + 'lip.jpg', img_resized)  # 保存嘴唇方框图片
                p_index = p_index + 1


        cap.release()


if __name__ == '__main__':

    landmarks_data_path = "C:\\Users\\luqi\\Desktop\\shape_predictor_68_face_landmarks.dat"

    namelist = ["aiban","minzhu","zgm","hexie"]

    for name in namelist:
        for index in range(1,161):
            File_Path = "E:\\lip_tracking_project\\cnn_and_lstm_0822\\allpics=10\\train\\" + name + str(index) + "\\" + "jiangzao\\"
            File_Path2 = "E:\\lip_tracking_project\\cnn_and_lstm_0822\\allpics=10\\train\\" + name + str(index) + "\\" + "tezhengdiantiqu\\"
            video_path = "E:\\lip_tracking_project\\8.20视频集\\训练集\\" + name + "\\" + name + str(index) +".avi"
            # 获取当前文件目录
            if not os.path.exists(File_Path):  # 检查是否存在文件夹
                os.makedirs(File_Path)  # 不存在则新建
            if not os.path.exists(File_Path2):  # 检查是否存在文件夹
                os.makedirs(File_Path2)  # 不存在则新建
            test = Video_point(File_Path=File_Path, File_Path2=File_Path2, video_path=video_path, landmarks_data_path=landmarks_data_path, allpicture=10, sampletimes=0,frameFrequency=1)
            test.process()
    
    for name in namelist:
        for index in range(1,41):
            File_Path = "E:\\lip_tracking_project\\cnn_and_lstm_0822\\allpics=10\\test\\" + name + str(index) + "\\" + "jiangzao\\"
            File_Path2 = "E:\\lip_tracking_project\\cnn_and_lstm_0822\\allpics=10\\test\\" + name + str(index) + "\\" + "tezhengdiantiqu\\"
            video_path = "E:\\lip_tracking_project\\8.20视频集\\测试集\\" + name + "\\" + name + str(index) +".avi"
            # 获取当前文件目录
            if not os.path.exists(File_Path):  # 检查是否存在文件夹
                os.makedirs(File_Path)  # 不存在则新建
            if not os.path.exists(File_Path2):  # 检查是否存在文件夹
                os.makedirs(File_Path2)  # 不存在则新建
            test = Video_point(File_Path=File_Path, File_Path2=File_Path2, video_path=video_path, landmarks_data_path=landmarks_data_path, allpicture=10, sampletimes=0,frameFrequency=1)
            test.process()
