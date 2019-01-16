##此文件主要用于统计数据集中每个司机的图片数
import pandas as pd
import h5py
import matplotlib.pyplot as plt

df=pd.read_csv('../data/driver_imgs_list.csv')
df.describe()
ts = df['subject'].value_counts()
print (ts)
fig = plt.figure(figsize=(15,5))  
plt.bar(ts.index.tolist(), ts.iloc[:].tolist(), 0.4, color="green")  
plt.xlabel("Driver ID")  
plt.ylabel("File nums")  
plt.title("Driver ID distribution")
plt.show()
