import cv2
import numpy as np
from keras.models import load_model
from keras.applications import resnet50
import keras_applications
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
state_des = {'c0':'safe driving','c1':'texting - right hand','c2':'talking on the phone - right','c3':'texting - left hand',  \
             'c4':'talking on the phone - left hand','c5':'operating the radio','c6':'drinking','c7':'reaching behind','c8':'hair and makeup',  \
             'c9':'talking to passenger'};

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


res_image_size = (224, 224)
res_input_shape = (224, 224, 3)

cam_model, cam_weights = cam_model(resnet50.ResNet50, res_input_shape, resnet50.preprocess_input, 10, 'my_model_weights.h5')

print (cam_weights.shape)

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
	
image_path = '../data/imgs/test/2.jpg'
resNet_input_shape = (224,224,3)
show_hot_map(image_path, cam_model, cam_weights, res_input_shape)

