
#提取所有样本在top layer之前的输出结果(对于resnet，该输出结果的维度是112048，以后统一叫做特征向量)，
#将其保存起来，从而将其作为top layer的输入来达到加速调参的过程。也算是一种用空间换取时间的策略吧！
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import resnet50
from sklearn.utils import shuffle
from os.path import isfile, isdir
import numpy as np
import os
import h5py

train_dir_name='../data/imgs/train' ##训练集路径
test_dir_name='../data/imgs/test'   ##测试集路径

resNet_input_shape = (224,224,3)
res_x = Input(shape=resNet_input_shape)
res_x = Lambda(resnet50.preprocess_input)(res_x)
res_model = resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=res_x, input_shape=resNet_input_shape)
res_model.summary()
out = GlobalAveragePooling2D()(res_model.output)
res_vec_model = Model(inputs=res_model.input, outputs=out)
def model_vector_catch(MODEL, image_size, vect_file_name, vec_dir, train_dir, test_dir, preprocessing=None):
    """
        MODEL:the model to extract bottleneck features
        image_size：MODEL input size(h, w, channels)
        vect_file_name:file to save vector
        preprocessing:whether or not need preprocessing
    """
    if isfile(vec_dir + '/' + vect_file_name):
        print ("%s already OK!" % (vect_file_name))
        return
    
    input_tensor = Input(shape=(image_size[0], image_size[1], 3))
    
    if preprocessing:
        ## check if need preprocessing
        input_tensor = Lambda(preprocessing)(input_tensor)
    
    model_no_top = MODEL(include_top=False, weights='imagenet', input_tensor=input_tensor, input_shape=(image_size[0], image_size[1], 3))
   
    ## flatten the output shape and generate model
    out = GlobalAveragePooling2D()(model_no_top.output)
    new_model = Model(inputs=model_no_top.input, outputs=out)
    
    ## get iamge generator
    gen = ImageDataGenerator()
    test_gen = ImageDataGenerator()
    
    """
    classes = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', ] -- cat is 0, dog is 1, so we need write this
    class_mode = None -- i will not use like 'fit_fitgenerator', so i do not need labels
    shuffle = False -- it is unneccssary
    batch_size = 64 
    """
    class_list = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', ]
    train_generator = gen.flow_from_directory(train_dir, image_size, color_mode='rgb', \
                                              classes=class_list, class_mode=None, shuffle=False, batch_size=64)
    
    #test_generator = test_gen.flow_from_directory(test_dir, image_size, color_mode='rgb', \
                                          #class_mode=None, shuffle=False, batch_size=64)
   
    """
    steps = None, by default, the steps = len(generator)
    """
    train_vector = new_model.predict_generator(train_generator)
    #test_vector = new_model.predict_generator(test_generator)
    
    with h5py.File(vec_dir + "/" + (vect_file_name), 'w') as f: 
        f.create_dataset('x_train', data=train_vector)
        f.create_dataset("y_train", data=train_generator.classes)
        #f.create_dataset("test", data=test_vector)
    print ("Model %s vector cached complete!" % (vect_file_name))
	
vec_dir = 'vect'

if not isdir(vec_dir):
    os.mkdir(vec_dir)

res_vect_file_name = 'resnet50_vect.h5'

model_vector_catch(resnet50.ResNet50, resNet_input_shape[:2], res_vect_file_name, vec_dir, train_dir_name, test_dir_name, resnet50.preprocess_input)
	
driver_classes = 10

input_tensor = Input(shape=(2048,))
x = Dropout(0.5)(input_tensor)
x = Dense(driver_classes, activation='softmax', name='res_dense_1')(x)

resnet50_model = Model(inputs=input_tensor, outputs=x)

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

x_train = []
y_train = []

with h5py.File(vec_dir + '/' + res_vect_file_name, 'r') as f:
    x_train = np.array(f['x_train'])
    y_train = np.array(f['y_train'])
    
    #one-hot vector
    y_train = convert_to_one_hot(y_train, driver_classes)
    
    x_train, y_train = shuffle(x_train, y_train, random_state=0)
	
resnet50_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

hist = resnet50_model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.2)
