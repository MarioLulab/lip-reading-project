①先打开“create_frame_folder.py”,生成每个视频对应的唇部区域
②打开“create_dataset.py”，根据每个视频的唇部图片生成每种词语对应的训练集和测试集的json格式文件
③依次运行“create_training_set.py”和“create_testing_set.py”，生成总的训练集和测试集的json格式文件
④运行“train.py”，设置好训练的参数，得到训练好的模型
⑤将训练好的模型导入“main.py”中，在树莓派上运行“main.py”代码