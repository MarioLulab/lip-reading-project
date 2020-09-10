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


namelist = {"0": "爱班",
            "1": "民主",
            "2": "和谐",
            "3": "中国梦",
            }

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


traindataset_filename = "D:\\lip_reading_project\\训练集0822_四个词.txt"
testdataset_filename = "D:\\lip_reading_project\\测试集0822_四个词.txt"

with open(traindataset_filename) as f:
    train_dataset = json.load(f)

with open(testdataset_filename) as f:
    test_dataset = json.load(f)

training_dataset = MyDataset(train_dataset)
train_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=64, shuffle=True)
testing_dataset = MyDataset(test_dataset)
test_loader = torch.utils.data.DataLoader(dataset=testing_dataset, batch_size=64, shuffle=True)

batch_size = 64
input_size = 4096 # 从alexnet模型输出维度为4096的特征向量，所以输入给LSTM中1个cell的输入向量维度也为4096
hidden_size = 512 # memory size
num_layers = 2  # lstm的层数
batch_first = True
output_size0 = 1 #第一个全连接层的输出维度
num_classification = 4
output_size1 = num_classification # 第二个全连接层（最后一个全连接层）的输出维度，为类别的个数
num_frame = 11

h0 = torch.randn(batch_size, 2, hidden_size)
c0 = torch.randn(batch_size, 2, hidden_size)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = batch_first,
            dropout = 0.5,
        )
        self.fc0 = nn.Linear(hidden_size,output_size0)
        self.fc1 = nn.Linear(num_frame-1,output_size1)


    def forward(self, x):
        # 默认C0 和 h0 为 0
        # input: x [batch_size,num_frame-1,num_feature]
        # output: output [ batch_size , num_classificaiton ]
        output,(hidden,cell) = self.lstm(x) # output : [batch_size,num_frame-1,hidden_size]
        output = self.fc0(output) # output : [batch_size, num_frame-1, 1]
        output = output.view(-1,num_frame-1) # output : [batch_size,num_frame-1]
        output = self.fc1(output)   # output : [batch_size , num_classification]
        return output,(hidden,cell)





lr = 0.01
model = Net() # 实例化一个模型（即我们之前封装好的LSTM模型）
CostFunction = nn.CrossEntropyLoss() # 代价函数定义为交叉熵函数
optimizer = optim.SGD(model.parameters(), lr= lr, momentum= 0.5) # 创建优化器
scheduler = lr_scheduler.StepLR(optimizer, step_size= 10, gamma= 1) # 训练过程中学习率也会更新

def train(model, CostFunction, optimizer):
    model.train()
    torch.set_grad_enabled(True)
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs,(hidden_output, cell_output) = model(inputs)
        loss = CostFunction(outputs, labels)
        loss.backward()
        optimizer.step()
    return model

def test(model, CostFunction):
    model.eval()
    torch.set_grad_enabled(False)
    cnt = 0
    loss_sum = 0.0
    for inputs, labels in test_loader:

        outputs,(hidden_output, cell_output) = model(inputs)
        # preds_index 储存outputs中每一行最大值的index
        max_, preds_index = torch.max(outputs, 1)
        loss = CostFunction(outputs, labels)
        loss_sum += loss.cpu().item()
        cnt += torch.sum(labels == preds_index)
    now_acc = 1.0 * cnt.item() / 160     # 160是测试集中视频的个数
    return now_acc, loss_sum

def training_model(model, CostFunction, optimizer, scheduler, num_epochs=100):
    best_acc = 0.0
    min_loss = 100
    best_model_state_dict = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        scheduler.step()
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print('-' * 60)
        model = train(model, CostFunction, optimizer)
        now_acc, loss = test(model, CostFunction)

        if min_loss > loss:
            best_acc = now_acc
            min_loss = loss
            best_model_state_dict = copy.deepcopy(model.state_dict())

        print("Now Test accuracy = {:.4f}%".format(now_acc * 100))
        print("Now Loss = {}".format(loss))

    print('*' * 60)
    print("Training Complete")
    print("Best accuracy = {}%".format(best_acc * 100))
    model.load_state_dict(best_model_state_dict)
    torch.save(model.state_dict(),
               'E:\\lip_tracking_project\\' + '4个词10帧迭代100隐藏层512两层layer的CNN+lstm的loss=' + str(min_loss)[:5] + ' best_accuracy=' + str(best_acc)[:5] + '.tar') # 存储模型参数
    return model

training_model(model, CostFunction, optimizer, scheduler, num_epochs=100)