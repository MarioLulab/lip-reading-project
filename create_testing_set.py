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

filename_ab = "E:\\lip_tracking_project\\cnn_and_lstm_0822\\数据集\\aiban测试集0822_四个词.txt"
filename_hx = "E:\\lip_tracking_project\\cnn_and_lstm_0822\\数据集\\hexie测试集0822_四个词.txt"
filename_mz = "E:\\lip_tracking_project\\cnn_and_lstm_0822\\数据集\\minzhu测试集0822_四个词.txt"
filename_zgm = "E:\\lip_tracking_project\\cnn_and_lstm_0822\\数据集\\zgm测试集0822_四个词.txt"

target_file = "D:\\lip_reading_project\\测试集0822_四个词.txt"

target_dic = {}

i = 0

with open(filename_ab) as f:
    data_tra = json.load(f)
    for key, value in data_tra.items():
        value[1] = 0
        target_dic[str(i)] = value
        i = i + 1

with open(filename_hx) as f:
    data_tra = json.load(f)
    for key, value in data_tra.items():
        value[1] = 2
        target_dic[str(i)] = value
        i = i + 1

with open(filename_mz) as f:
    data_tra = json.load(f)
    for key, value in data_tra.items():
        value[1] = 1
        target_dic[str(i)] = value
        i = i + 1

with open(filename_zgm) as f:
    data_tra = json.load(f)
    for key, value in data_tra.items():
        value[1] = 3
        target_dic[str(i)] = value
        i = i + 1

with open(target_file,'w') as f:
    f.write(json.dumps(target_dic))
print("测试集视频总个数 = {}".format(i))


