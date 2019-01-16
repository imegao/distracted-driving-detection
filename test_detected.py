import sys
import cv2
import numpy as np
import keras_applications
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.applications import resnet50
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image,ImageDraw,ImageFont
from ui_screen import *

state_des = {'c0':'安全驾驶','c1':'右手打字','c2':'右手打电话','c3':'左手打字',  \
             'c4':'左手打电话','c5':'调收音机','c6':'喝饮料','c7':'拿后面东西','c8':'打理头部或脸部',  \
             'c9':'和其他乘客说话'};
			 
res_image_size = (224, 224)
res_input_shape = (224, 224, 3)
resNet_input_shape = (224,224,3)
image_path = '../data/imgs/test/2.jpg'

## video width and high
main_video_width = 1280
main_video_high = 720

## subwindow size
sub_video_width = int(main_video_width/2)
sub_video_high = int(main_video_high*0.6)

## subwindow coordinate
sub1_coord_1 = int((main_video_high-sub_video_high)/2)
sub1_coord_2 = int((main_video_high-sub_video_high)/2) + sub_video_high
sub1_coord_3 = 0
sub1_coord_4 = sub1_coord_3 + sub_video_width

sub2_coord_1 = int((main_video_high-sub_video_high)/2)
sub2_coord_2 = int((main_video_high-sub_video_high)/2) + sub_video_high
sub2_coord_3 = sub1_coord_4
sub2_coord_4 = main_video_width

class Mywindow(QMainWindow,Ui_MainWindow):
    def __init__(self,formObj): 
        super(Mywindow, self).__init__()
        self.setupUi(formObj)
        self.open_danger.clicked.connect(self.Open_Denger)
        self.close_danger.clicked.connect(self.Close_Denger)
        self.open_cam.clicked.connect(self.open_camer)
        self.close_cam.clicked.connect(self.close_camer)
        self.exit_btn.clicked.connect(self.exit_sys)
        self.timer_camera = QTimer(self)
        self.timer_camera.timeout.connect(self.show_cam)
        self.cap=None
        self.flag=0
    
    def cam_model(self,MODEL, input_shape, preprocess_input, output_num, weights_file_name):
        """
        #MODEL 使用的模型 是resnet50
        #input_shape 输入的图片大小（224,224,3）
        #preprocess_input对数据进行预处理，主要是通过简单缩放方法是通过对数据各个维度上的值进行重新调节，使得数据整体上分布在[0,1]或[-1,1]区间
        #output_num 输出的大小 ，输出为10类
        #weights_file_name 权重文件的名称
        """     
        ## 获取输入维度尺寸
        x = Input(shape=input_shape)
        ##对数据进行预处理
        if preprocess_input:
            x = Lambda(preprocess_input)(x)
        ##MODEL是获得resnet50模型结构，但不采用它的权重和顶部输出
        notop_model = MODEL(include_top=False, weights=None, input_tensor=x, input_shape=input_shape)
        ##通过求平均的方式对全连接层进行替代，避免了全连接层需要大量权重参数 参考论文：Network In Network 
        x = GlobalAveragePooling2D(name='global_average_2d_1')(notop_model.output)

        ## 建立顶层的全连接网络，其实由于上一步的全局平均化已经可以代替全连接网络，只需要再加一层输出 参考论文 ：Dropout
        x = Dropout(0.5, name='dropout_1')(x)
        out = Dense(output_num, activation='softmax', name='dense_1')(x)
        #该模型包含从输入到输出的所有层，是一个单输入多输出的形式，输出有最后一层和倒数第二层
        ret_model = Model(inputs=notop_model.input, outputs=[out, notop_model.layers[-1].output])
    
        ## 加载预先训练好的模型权重
        ret_model.load_weights(weights_file_name)
    
        ## 获得输出层的权重
        weights = ret_model.layers[-1].get_weights()
    
        return ret_model, np.array(weights[0])

    def generate_hot_map(self,frame, cam_model, model_input_size, cam_weights, cam_size):
        """
        frame   输入的图像
        cam_model:      使用的模型
        model_input_size 模型应该输入图片的大小
        cam_weights:    模型权重
        cam_size:       摄像头图片的大小
        """
    
        # 调整大小
        img_for_model = cv2.resize(frame, model_input_size)
        img_for_model = np.expand_dims(img_for_model,axis=0)
    
        # 获得预测的结果 在模型中有两个输出 所以会有两个预测结果
        predict_m, feature_maps_m = cam_model.predict(img_for_model)
    
        """ 2. get the calss activation maps """
        predict = predict_m[0]
        feature_maps = feature_maps_m[0]
      
        # 获得预测结果最大值
        class_index = np.argmax(predict)

        # 获得该结果的权重
        cam_weights_c = cam_weights[:, class_index]

        # 矩阵相乘
        cam = np.matmul(feature_maps, cam_weights_c)

        # 对结果标准化
        cam = (cam - cam.min())/(cam.max())

        # 低于0.2的设为0
        cam[np.where(cam<0.2)] = 0

        cam = cv2.resize(cam, cam_size)
        cam = np.uint8(255*cam)
    
        """ 3. show the hot map """
        des = state_des['c'+str(class_index)]

        # draw the hotmap
        hotmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

        # linear combine the picture with cam
        image_input = cv2.resize(frame, cam_size)
        dis = cv2.addWeighted(image_input, 0.8, hotmap, 0.4, 0)

        return dis,predict

    def generate_video_with_classfication(self,model, model_input_size, video_name_or_camera, cam_weights, generate_video_name='output.avi'):
        """
        model:                 预测模型
        model_input_size:      图片输入尺寸 
        video_name_or_camera:  使用照相机或者视频
        cam_weights:           权重
        generate_video_name:   权重文件名称
        """

   
    
        ''' 0. create a new image '''
        showBigImage = np.zeros((int(main_video_high), int(main_video_width), 3), np.uint8)
    
        ''' 1. create video writer '''
        
    
        if(self.cap.isOpened() == False):
            print ("Failed to open " + video_name_or_camera)
            return
    
        
        success, frame=self.cap.read()
        if success:
            
            """ 2. preprocessing and predict """
            
            sub_frame_2, predict = self.generate_hot_map(frame, model, model_input_size, cam_weights, (sub_video_width, sub_video_high))
            
            class_index = np.argmax(predict)
            if(self.flag==0):
                self.label.setPixmap(QPixmap(""))
            else:
                if(str(class_index)=='0'):
                    self.label.setPixmap(QPixmap("UI/image/safe.jpg"))
                else:
                    self.label.setPixmap(QPixmap("UI/image/danger.jpg"))
            
            for idx, val in enumerate(predict):
                predict[idx]=float('%.3f' % val)
            
            self.lcd_class_1.display(float(predict[0]))
            self.lcd_class_2.display(float(predict[1]))
            self.lcd_class_3.display(float(predict[2]))
            self.lcd_class_4.display(float(predict[3]))
            self.lcd_class_5.display(float(predict[4]))
            self.lcd_class_6.display(float(predict[5]))
            self.lcd_class_7.display(float(predict[6]))
            self.lcd_class_8.display(float(predict[7]))
            self.lcd_class_9.display(float(predict[8]))
            self.lcd_class_10.display(float(predict[9]))

            text = 'Predicted:  C{}  {}'.format(class_index, state_des['c'+str(class_index)])

            cv2img = cv2.cvtColor(sub_frame_2, cv2.COLOR_BGR2RGB)
            pilimg = Image.fromarray(cv2img)
            draw = ImageDraw.Draw(pilimg)  
            font = ImageFont.truetype("simhei.ttf", 30, encoding="utf-8")
            draw.text((0, 0), text, (0, 0, 0), font=font)
            sub_frame_2 = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)

            cv2img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pilimg = Image.fromarray(cv2img)
            draw = ImageDraw.Draw(pilimg)  
            font = ImageFont.truetype("simhei.ttf", 30, encoding="utf-8")
            draw.text((0, 0), text, (255, 0, 0), font=font)
            frame = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)

            """ 3. add text to the image and show"""
            font = cv2.FONT_HERSHEY_SIMPLEX
           
        
            """ 4. resize and fill 2 subwindow """
            frame = cv2.resize(frame, (sub_video_width, sub_video_high))
            showBigImage[sub1_coord_1:sub1_coord_2, sub1_coord_3:sub1_coord_4] = frame
            showBigImage[sub2_coord_1:sub2_coord_2, sub2_coord_3:sub2_coord_4] = sub_frame_2
              
            """ 5. show video """
           
            show = cv2.cvtColor(showBigImage, cv2.COLOR_BGR2RGB)
            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
           
            self.video_feed.setScaledContents (True)
            self.video_feed.setPixmap(QPixmap.fromImage(showImage))    
           

    def open_camer(self):
        
        self.cap = cv2.VideoCapture('train_deal_video.avi')
        self.cam_model1, self.cam_weights = self.cam_model(resnet50.ResNet50, res_input_shape, resnet50.preprocess_input, 10, 'model_weights_1_15.h5')
        self.timer_camera.start(10)
    def Open_Denger(self):
        self.flag=1
    def Close_Denger(self):
        self.flag=0
    def close_camer(self):
        self.video_feed.setPixmap(QPixmap(""))
        self.timer_camera.stop()
        self.cap.release()
    def show_cam(self):
        self.generate_video_with_classfication(self.cam_model1, (224, 224), 1, self.cam_weights, generate_video_name='output.avi')
    def exit_sys(self):
        sys.exit()
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    formObj=QtWidgets.QMainWindow()
    ui = Mywindow(formObj)    
    formObj.show()
    sys.exit(app.exec())