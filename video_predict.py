import cv2
import numpy as np
import keras_applications
import os
import time
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.applications import resnet50
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image,ImageDraw,ImageFont
state_des = {'c0':'安全驾驶','c1':'右手打字','c2':'右手打电话','c3':'左手打字',  \
             'c4':'左手打电话','c5':'调收音机','c6':'喝饮料','c7':'拿后面东西','c8':'打理头部或脸部',  \
             'c9':'和其他乘客说话'};
			 
res_image_size = (224, 224)
res_input_shape = (224, 224, 3)
resNet_input_shape = (224,224,3)
image_path = '../data/imgs/test/2.jpg'

			 
def cam_model(MODEL, input_shape, preprocess_input, output_num, weights_file_name):
    """
        MODEL:                  pretrained model
        input_shape:            pre-trained model's input shape
        preprocessing_input:    pre-trained model's preprocessing function
        weights_file_name:      weights trained on driver datasheet
    """
    
    ## get pretrained model
    x = Input(shape=input_shape)
    
    if preprocess_input:
        x = Lambda(preprocess_input)(x)
    
    notop_model = MODEL(include_top=False, weights=None, input_tensor=x, input_shape=input_shape)
    
    x = GlobalAveragePooling2D(name='global_average_2d_1')(notop_model.output)

    ## build top layer
    x = Dropout(0.5, name='dropout_1')(x)
    out = Dense(output_num, activation='softmax', name='dense_1')(x)
    
    ret_model = Model(inputs=notop_model.input, outputs=[out, notop_model.layers[-2].output])
    
    ## load weights
    ret_model.load_weights(weights_file_name)
    
    ## get the output layer weights
    weights = ret_model.layers[-1].get_weights()
    
    return ret_model, np.array(weights[0])

def generate_hot_map(frame, cam_model, model_input_size, cam_weights, cam_size):
    """
        image_input_m:  CAM model's input
        cam_model:      CAM model
        cam_weights:    weights for CAM
        cam_size:       size of the output picture 
    """
    
    # resize frame for predict
    img_for_model = cv2.resize(frame, model_input_size)
    img_for_model = np.expand_dims(img_for_model,axis=0)
    
    """ 1. predict """
    # predict and get feature maps
    predict_m, feature_maps_m = cam_model.predict(img_for_model)
    
    """ 2. get the calss activation maps """
    predict = predict_m[0]
    feature_maps = feature_maps_m[0]

    # get the class result
    class_index = np.argmax(predict)

    # get the class_index unit's weights
    cam_weights_c = cam_weights[:, class_index]

    # get the class activation map
    cam = np.matmul(feature_maps, cam_weights_c)

    # normalize the cam
    cam = (cam - cam.min())/(cam.max())

    # do not care the low values
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

def show_hot_map(image_path, model, cam_weights, input_shape):
    """ 1. predict """
    # read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image_input = cv2.resize(image, (input_shape[0], input_shape[1]))

    image_input_m = np.expand_dims(image_input,axis=0)

    # predict and get feature maps
    predict_m, feature_maps_m = model.predict(image_input_m)
    
    """ 2. get the calss activation maps """
    predict = predict_m[0]
    feature_maps = feature_maps_m[0]

    # get the class result
    class_index = np.argmax(predict)

    # get the class_index unit's weights
    cam_weights_c = cam_weights[:, class_index]

    # get the class activation map
    cam = np.matmul(feature_maps, cam_weights_c)

    # normalize the cam
    cam = (cam - cam.min())/(cam.max())

    # do not care the low values
    cam[np.where(cam<0.2)] = 0

    cam = cv2.resize(cam, (input_shape[0], input_shape[1]))
    cam = np.uint8(255*cam)
    
    """ 3. show the hot map """
  
  

    des = state_des['c'+str(class_index)]

    # draw the hotmap
    hotmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    # linear combine the picture with cam
    dis = cv2.addWeighted(image_input, 0.8, hotmap, 0.4, 0)

    plt.title("Predict C" + str(class_index) + ':' + des)
    plt.imshow(dis)
    print("1")
    plt.axis("off")
    plt.show()


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

def generate_video_with_classfication(model, model_input_size, video_name_or_camera, cam_weights, generate_video_name='output.avi'):
    """
        model:                 model to predict the video
        model_input_size:      image size of the model 
        video_name_or_camera:  read videl from camera or local video
        cam_weights:           weights for CAM
        generate_video_name:   the output video name
    """

    
    """0. create videl reader and writer, and get more video message """
    cap = cv2.VideoCapture(video_name_or_camera)
    
    video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_high = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    #print ("video size:({width}, {high})   fps:{fps}".format(width=video_width, high=video_high, fps=video_fps))
    
    ''' 0. create a new image '''
    showBigImage = np.zeros((int(main_video_high), int(main_video_width), 3), np.uint8)
    
    ''' 1. create video writer '''
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #writer = cv2.VideoWriter(generate_video_name, fourcc, 20.0, (main_video_width, main_video_high))
    
    if(cap.isOpened() == False):
        print ("Failed to open " + video_name_or_camera)
        return
    
    while True:
        start_time = time.time() #帧数读取
        time.sleep(1)
        """ 2. preprocessing and predict """
        # get fram
        ret, frame = cap.read()
        
        # check if the video is over
        if(ret != True):
            print ("Ending!")
            break
        
        # get hot map
        sub_frame_2, predict = generate_hot_map(frame, model, model_input_size, cam_weights, (sub_video_width, sub_video_high))
        
        class_index = np.argmax(predict)
        print(predict)
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
        showBigImage[:] = 0
        #cv2.putText(showBigImage, text, (10, sub1_coord_1-10), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        """ 4. resize and fill 2 subwindow """
        frame = cv2.resize(frame, (sub_video_width, sub_video_high))
        showBigImage[sub1_coord_1:sub1_coord_2, sub1_coord_3:sub1_coord_4] = frame
        showBigImage[sub2_coord_1:sub2_coord_2, sub2_coord_3:sub2_coord_4] = sub_frame_2
        
        """ 5. show video """
        cv2.imshow('image', showBigImage)
        
        """ 6. save video if need """
        #writer.write(showBigImage)
        print("FPS: ", 1.0 / (time.time() - start_time)) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
	
#cam_model, cam_weights = cam_model(resnet50.ResNet50, res_input_shape, resnet50.preprocess_input, 10, 'my_model_weights.h5')
#generate_video_with_classfication(cam_model, res_image_size, 'train_deal_video.avi', cam_weights, generate_video_name='output.avi')



