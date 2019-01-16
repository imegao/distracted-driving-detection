##随机抽取几张图片，做展示，查看是否均匀分布
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

train_dir_name='../data/imgs/train' ##训练集路径
test_dir_name='../data/imgs/test'   ##测试集路径
state_des = {'c0':'safe driving','c1':'texting - right hand','c2':'talking on the phone - right','c3':'texting - left hand',  \
             'c4':'talking on the phone - left hand','c5':'operating the radio','c6':'drinking','c7':'reaching behind','c8':'hair and makeup',  \
             'c9':'talking to passenger'};

## class that you want to display
c = 0

## random choose the filenames of the class
dis_dir = train_dir_name + '/c' + str(c)
dis_filenames = os.listdir(dis_dir)

dis_list = np.random.randint(len(dis_filenames), size=(6))
dis_list = [dis_filenames[index] for index in dis_list]

plt.figure(1, figsize=(10, 10))

for i,filename in enumerate(dis_list):
    image = cv2.imread(dis_dir +  '/' + str(filename))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    ax1=plt.subplot(3,3,i+1)
    plt.imshow(image)
    plt.axis("off")
    plt.title(state_des['c'+str(c)] + "\n" + str(image.shape))
    
plt.show()