import cv2
import os
img_root = 'E:/graduation_project/data/all/'#这里写你的文件夹路径，比如：/home/youname/data/img/,注意最后一个文件夹要有斜杠
fps = 1    #保存视频的FPS，可以适当调整
size=(640,480)
#可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter('train_deal_video.avi',fourcc,fps,size)#最后一个是保存图片的尺寸

#for(i=1;i<471;++i)
for filename in os.listdir(r"E:/graduation_project/data/all"): 
    frame = cv2.imread(img_root+filename)
    videoWriter.write(frame)
videoWriter.release()
