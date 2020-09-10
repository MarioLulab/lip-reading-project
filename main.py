import threading
import time
from tkinter import*
import tkinter.messagebox
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
from torch.autograd import Variable
import torch.nn.functional as F



# 状态变量
flag = 0
flag_list = {"0":"待机状态","1":"拍照状态","2":"识别状态"}
g_max_index = 0

# 给几个视频标识编号

class_name = {0: "爱班", 1: "民主", 2: "和谐", 3: "中国梦"}

batch_size = 64
input_size = 4096 # LSTM的输入维度。自定义Alexnet模型输出是4096维的图片空间特征
hidden_size = 512 # LSTM隐含层数目
num_layers = 2  # lstm的层数
batch_first = True
output_size0 = 1 #第一个全连接层的输出维度
num_classification = 4  # 分类的词的种类
output_size1 = num_classification # 第二个全连接层（最后一个全连接层）的输出维度，为类别的个数
num_frame = 11

h0 = torch.randn(batch_size, 2, hidden_size)
c0 = torch.randn(batch_size, 2, hidden_size)

def thread_it(func, *args):
    # 创建线程
    t = threading.Thread(target=func, args=args)
    # 启动
    t.start()

def start():  # 定义一个 改变文本的函数
      global flag
      flag = 1

def stop():
    global flag
    flag = 2    # 进入识别状态
    # tkinter.messagebox.showinfo("识别结果","你说的是“爱国”吧")

def create_gui():
    root = Tk()     # 初始旷的声明 . 
    root.title("唇语识别")


    # 创建一个文本Label对象

    textLabel = Label(root,           # 绑定到root初始框上面
                    text="温馨提示：\n请按开始按钮开始视频录制，\n按停止按钮结束视频录制。",
                    justify=LEFT,
                    padx=10).grid(row=0,column=0)  # 字体 位置 

    Button(root,text="开始", width=10,command=start).grid(row=3,column=0,sticky=W,padx=10,pady=5)  # 按下按钮 执行 start函数
    
    Button(root,text="停止",width=10,command=stop).grid(row=3,column=1,sticky=E,padx=10,pady=5)

    mainloop()

class Video_point():
    def __init__(self, File_Path, File_Path2, landmarks_data_path, allpicture, sampletimes=0):
        # input : File_Path             用于对视频流分割、降噪、滤波的工作文件夹
        #         File_Path2            用于对每一帧进行特征点提取的工作文件夹
        #         video_path            要进行唇语识别的目标视频流路径
        #         landmarks_data_path    dlib人脸检测识别库
        #         frameFrequency        提取帧的频率。每frameFrequency帧提取一帧来进行特征点识别
        #         allpicture            关键帧的总长度

        self.times = 0                                             # 初始化计时次数
        self.allpicture = allpicture                               # 关键帧的固定总长度
        self.sampletimes = sampletimes                             # dlib的detector的采样倍数。sampletimes越高，识别精度越高，但运行得也越慢，经过实验，发现选择2已经几乎可以保证100%的提取特征点的正确率了
        self.detector = dlib.get_frontal_face_detector()           # 创建人脸检测detector
        self.predictor = dlib.shape_predictor(landmarks_data_path) # 创建predictor
        self.index = []                                            # 初始化关键帧下标序列
        self.File_Path = File_Path
        self.File_Path2 = File_Path2

    def salt(self, img, n):
        # funciton : 降噪滤波函数
        # input : img : cv2.imread()读取的视频帧
        #         n   : 某个参数，我也不知道。得问梁梁
        # return : 降噪滤波处理完成后的视频帧
        for k in range(n):
            i = int(np.random.random() * img.shape[1])
            j = int(np.random.random() * img.shape[0])
            if img.ndim == 2:
                img[j, i] = 255
            elif img.ndim == 3:
                img[j, i, 0] = 255
                img[j, i, 1] = 255
                img[j, i, 2] = 255
            return img
    
    def process(self):
        # function : 对视频进行降噪、滤波、提取特征点坐标的处理
        # return   : frame 提取得到的该视频的frame，frame的格式见“相关数据格式.jpg”的说明
        
        # 获取当前文件目录
        if not os.path.exists(self.File_Path):  # 检查是否存在文件夹
            os.makedirs(self.File_Path)  # 不存在新建
        if not os.path.exists(self.File_Path2):  # 检查是否存在文件夹
            os.makedirs(self.File_Path2)  # 不存在新建

        global g_max_index
        max_index = g_max_index

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
            img = cv2.imread(self.File_Path + str(index_i) + '.jpg', cv2.IMREAD_GRAYSCALE)  # 读取视频帧
            result = self.salt(img, 500)  # 去噪滤波

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
                cv2.imwrite(self.File_Path2 + str(p_index) + 'lip.jpg', img_resized)  # 保存嘴唇方框图片
                p_index = p_index + 1


# LSTM模型
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

# CNN模型
class alex(nn.Module):
    def __init__(self):
        super(alex, self).__init__()
        alexnet = models.alexnet(pretrained=True)                                      # 加载预训练好的alexnet模型
        self.feature = alexnet.features
        self.classifier = nn.Sequential(*list(alexnet.classifier.children())[:-2])     # 选择Fc7层的提取到的图片特征为输出
        pretrained_dict = alexnet.state_dict()                                         
        model_dict = self.classifier.state_dict()                                      
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}# 提取预训练好的alexnet模型参数
        model_dict.update(pretrained_dict)                                             
        self.classifier.load_state_dict(model_dict)                                    # 将预训练好的模型参数加载入自定义的alexnet模型
 
    def forward(self, x):                                                              # 定义前向传播函数
        output = self.feature(x)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        return output

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


def pred(model,allpicture,File_Path,File_Path2,landmarks_data_path):
    start_time = time.time()
    # File_Path = "E:\\lip_tracking_project\\cnn_and_lstm\\jiangzao\\" 
    # File_Path2 = "E:\\lip_tracking_project\\cnn_and_lstm\\tezhengdiantiqu\\"

    # landmarks_data_path = "C:\\Users\\luqi\\Desktop\\shape_predictor_68_face_landmarks.dat"

    # File_Path=File_Path
    # File_Path2=File_Path2
    # landmarks_data_path=landmarks_data_path
    test = Video_point(File_Path=File_Path, File_Path2=File_Path2, landmarks_data_path=landmarks_data_path, allpicture=allpicture, sampletimes=0)
    test.process()
    end0_time = time.time()
    print("从视频提取帧耗时{} 秒".format(end0_time - start_time))
    frame = np.zeros([1, 4096], dtype=float)  # 该视频提取到的frame,刚开始只有维度1只有1行
    for i in range(0,allpicture):
        # 生成单个视频的特征矩阵
        image_path = File_Path2 + str(i) + "lip.jpg"
        result = extractor(image_path, alexnet)     # 返回某一张图片的特征矩阵
        if i==0 :
            # 如果是第一张帧
            frame[0] = result
        else :
            # 如果不是第一帧，则按行往下增加一帧
            frame = np.row_stack((frame, result))
    end1_time = time.time()
    print("运行CNN，耗时{} 秒".format(end1_time - end0_time))
    all_frames = np.zeros((batch_size,allpicture,4096), dtype=float)
    
    # for k in range(batch_size):
        # all_frames[k] = frame
    all_frames[0] = frame
    
    all_frames = torch.from_numpy(all_frames).float()
    output,(hidden_output, cell_output) = model(all_frames)
    output_max, pred_index = torch.max(output, 1)
    pred_softmax = F.softmax(output)
    max_softmax,_ = torch.max(pred_softmax,dim=1)
    end2_time = time.time()
    print("运行LSTM，耗时{} 秒".format(end2_time - end1_time))
    end_time = time.time()
    print("总共耗时{} 秒".format(end_time - start_time))
    return pred_index

def snapShotCt(output_dir,camera_idx=1):  # camera_idx的作用是选择摄像头。如果为0则使用内置摄像头，比如笔记本的摄像头，用1或其他的就是切换摄像头。
    # 舍弃前10张图片和最后一张图片
    cap = cv2.VideoCapture(camera_idx)
    cap.set(3,640)
    cap.set(4,480)
    i=-5
    ret, frame = cap.read()  # cao.read()返回两个值，第一个存储一个bool值，表示拍摄成功与否。第二个是当前截取的图片帧。
    while (ret and flag==1):
        output_path = os.path.join(output_dir, "%d.jpg" % i)
        cv2.imwrite(output_path, frame)
        time.sleep(0.05)  # 休眠一秒 可通过这个设置拍摄间隔，类似帧。
        ret, frame = cap.read()  # 下一个帧图片
        i+=1
    cap.release()
    max_idx = i - 2 
    return max_idx


def predict_loop():
    global flag
    while(1):
        if (flag == 2):
            # 识别状态
            result = pred(model=LSTM,allpicture=10,File_Path = "E:\\lip_tracking_project\\cnn_and_lstm\\jiangzao\\",
                        File_Path2 = "E:\\lip_tracking_project\\cnn_and_lstm\\tezhengdiantiqu\\",
                        landmarks_data_path = "C:\\Users\\luqi\\Desktop\\shape_predictor_68_face_landmarks.dat")
            if result[0] == 0:
                tkinter.messagebox.showinfo("识别结果","你说的是“爱班”吧")
            elif result[0] == 1:
                tkinter.messagebox.showinfo("识别结果","你说的是“民主”吧")
            elif result[0] == 2:
                tkinter.messagebox.showinfo("识别结果","你说的是“和谐”吧")
            elif result[0] == 3:
                tkinter.messagebox.showinfo("识别结果","你说的是“中国梦”吧")
            # elif result[0] == 4:
            #     tkinter.messagebox.showinfo("识别结果","你说的是“中国梦”吧")
            # elif result[0] == 5:
            #     tkinter.messagebox.showinfo("识别结果","你说的是“自由”吧")            


            # 把标志位置为0，进入待机模式
            flag = 0
        elif(flag == 1):
            # 拍照状态
            global g_max_index
            # 拍照
            g_max_index = snapShotCt(output_dir='E:\\lip_tracking_project\\cnn_and_lstm\\jiangzao\\', camera_idx=0)
        elif(flag == 0):
            pass


if __name__  == "__main__":
     # 加载模型
    model_path = 'D:\\lip_reading_project\\4个词10帧迭代100隐藏层512两层layer的CNN+lstm的loss=0.970 best_accuracy=0.6.tar' # 已训练好的模型位置
    LSTM = Net()
    alexnet = alex()
    LSTM.load_state_dict(torch.load(model_path, map_location='cpu'))
    LSTM.eval()
    
    # 创建两个线程
    try:
        thread_it(create_gui)
        time.sleep(3)   # 创建gui窗口后等待3秒，才创建主循环线程
        thread_it(predict_loop)
    except:
        print ("Error: unable to start thread")
    
    while(1):
        pass