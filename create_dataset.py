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

# function : 利用pytorch预训练的alexnet模型创建自定义的alexnet，取原模型的倒数第二层输出作为自定义模型的输出
class alex(nn.Module):
    def __init__(self):
        super(alex, self).__init__()
        alexnet = models.alexnet(pretrained=True)
        self.feature = alexnet.features
        self.classifier = nn.Sequential(*list(alexnet.classifier.children())[:-2])
        pretrained_dict = alexnet.state_dict()
        model_dict = self.classifier.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.classifier.load_state_dict(model_dict)
 
    def forward(self, x):
        output = self.feature(x)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        return output

# 自定义数据集
class MyDataset(torch.utils.data.Dataset): #创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, Listdataset, transform=None, target_transform=None): #初始化一些需要传入的参数
        super(MyDataset,self).__init__()
        timelist = []
        for key,value in Listdataset.items():
            tra = torch.tensor(value[0])
            label = value[1]
            timelist.append((tra,int(label)))

        self.timelist = timelist
        self.transform = transform
        self.target_transform = target_transform
 
    def __getitem__(self, index):    #这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        tra = self.timelist[index][0]
        label = self.timelist[index][1]
        return tra,label     #return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
 
    def __len__(self): #这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.timelist)

def create_dataset(net, name, savepath, train_or_test, allpicture):
    start_time = time.time()
    Listdataset = {}

    if name == "aiban":
        label = 0
    elif name == "minzhu":
        label = 1
    elif name == "hexie" :
        label = 2
    elif name == "zgm":
        label = 3

    if train_or_test == "train":
        for index in range(1,161):
            feature_frame_folder = "E:\\lip_tracking_project\\cnn_and_lstm_0822\\" + "allpics=10\\" + train_or_test + "\\" + name + str(index) + "\\tezhengdiantiqu\\"
            frame = np.zeros([1, 4096], dtype=float)  # 该视频提取到的frame,刚开始只有维度1只有1行

            for i in range(0,allpicture):
                # 生成单个视频的特征矩阵
                image_path = feature_frame_folder + str(i) + "lip.jpg"
                result = extractor(image_path, net)     # 返回某一张图片的特征矩阵
                if i==0 :
                    # 如果是第一张帧
                    frame[0] = result
                else :
                    # 如果不是第一帧，则按行往下增加一帧
                    frame = np.row_stack((frame, result))
            
            Listdataset[str(index)] = [frame.tolist(), label]

    elif train_or_test == "test":
        for index in range(1,41):
            feature_frame_folder = "E:\\lip_tracking_project\\cnn_and_lstm_0822\\" + "allpics=10\\" + train_or_test + "\\" + name + str(index) + "\\tezhengdiantiqu\\"
            frame = np.zeros([1, 4096], dtype=float)  # 该视频提取到的frame,刚开始只有维度1只有1行

            for i in range(0,allpicture):
                # 生成单个视频的特征矩阵
                image_path = feature_frame_folder + str(i) + "lip.jpg"
                result = extractor(image_path, net)     # 返回某一张图片的特征矩阵
                if i==0 :
                    # 如果是第一张帧
                    frame[0] = result
                else :
                    # 如果不是第一帧，则按行往下增加一帧
                    frame = np.row_stack((frame, result))
            
            Listdataset[str(index)] = [frame.tolist(), label]
    
    else : print("plz input correct type!")

    with open(savepath, "w") as f :
        f.write(json.dumps(Listdataset))

    end_time = time.time()

    print(name + "的" + train_or_test + "集构建耗时时间 {} seconds".format(end_time-start_time))


# function : 将图片转换为可用输入已pretrained的alexnet模型
# input : 图片地址，
#         pytorch已经pretrained好的神经网络模型
# return : alexnet的倒数第二层输出的array
def extractor(img_path, net):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]
    )
 
    img = Image.open(img_path)
    img = img.convert("RGB")
    img = transform(img)
 
    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    y = net(x)
    y = torch.squeeze(y)
    y = y.data.numpy()

    return y

if __name__ == "__main__":

    namelist = {"0": "爱班",
            "1": "民主",
            "2": "和谐",
            "3": "中国梦",
            }

    net = alex()    # 实例化一个alexnet模型
    if not os.path.exists("E:\\lip_tracking_project\\cnn_and_lstm_0822\\数据集\\"):  # 检查是否存在文件夹
        os.makedirs("E:\\lip_tracking_project\\cnn_and_lstm_0822\\数据集\\")  # 不存在则新建

    for name_of_label in namelist:
        savepath = "E:\\lip_tracking_project\\cnn_and_lstm_0822\\数据集\\" + name_of_label + "训练集0822_四个词.txt"
        create_dataset(net=net, name=name_of_label, savepath=savepath, train_or_test="train",allpicture=10)
        print(name_of_label + "训练集构造完成！")

    for name_of_label in namelist:
        savepath = "E:\\lip_tracking_project\\cnn_and_lstm_0822\\数据集\\" + name_of_label + "测试集0822_四个词.txt"
        create_dataset(net=net, name=name_of_label, savepath=savepath, train_or_test="test",allpicture=10)
        print(name_of_label + "测试集构造完成！")