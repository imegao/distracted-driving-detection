##该文件主要对数据集基本信息的处理里了解##
## 1.统计样本的数量
## 2.每一类训练数据的分布
"""
c0: 安全驾驶
c1: 右手打字
c2: 右手打电话
c3: 左手打字
c4: 左手打电话
c5: 调收音机
c6: 喝饮料
c7: 拿后面的东西
c8: 整理头发和化妆
c9: 和其他乘客说话

"""
import os
import matplotlib.pyplot as plt


train_dir_name='../data/imgs/train' ##训练集路径
test_dir_name='../data/imgs/test'   ##测试集路径

train_size=0
train_class_size={}

train_class_dir_name=os.listdir(train_dir_name) ##训练集类别代码
test_size=len(os.listdir(test_dir_name)) ##测试集的数量

##遍历每个类别的文件夹里图片数量，并统计
for dname in train_class_dir_name:
    file_names=os.listdir(train_dir_name+'/'+dname)     
    train_class_size[dname]=len(file_names) 
    train_size=train_class_size[dname]+train_size 
    
print("训练集各类别数量",train_class_size)
print("测试集数量",test_size)
print("训练集数量",train_size)

##画出统计图
fig=plt .figure(figsize=(10,6))
plt.bar(train_class_size.keys(),train_class_size.values(),0.5,color="green")
plt.xlabel("Classes")
plt.ylabel("imgs_number")
plt.title("Classes distribution")
plt.show()
 


