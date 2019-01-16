from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import resnet50
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle
from os.path import isfile, isdir
import shutil
import pandas as pd
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt

train_dir_name=r'../data/imgs/train/' ##训练集路径
test_dir_name=r'../data/imgs/test/'   ##测试集路径

link_path = 'train_link/'
link_train_path = 'train_link\\train'
link_valid_path = 'train_link\\validation'

test_link = 'test_link/'
test_link_path = 'test_link/data/'

resNet_input_shape = (224,224,3)
res_x = Input(shape=resNet_input_shape)
res_x = Lambda(resnet50.preprocess_input)(res_x)
res_model = resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=res_x, input_shape=resNet_input_shape)
res_model.summary()
out = GlobalAveragePooling2D()(res_model.output)
res_vec_model = Model(inputs=res_model.input, outputs=out)

classes = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']

classes = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']


					
def get_data_generator(train_dir, valid_dir, test_dir, image_size): 
    #gen = ImageDataGenerator(shear_range=0.3, zoom_range=0.3, rotation_range=0.3)
    gen = ImageDataGenerator(rotation_range=10.,width_shift_range=0.05,height_shift_range=0.05,shear_range=0.1,zoom_range=0.1,zca_whitening=True)
    #gen = MergeImageDataGenerator(rotation_range=10.,width_shift_range=0.05,height_shift_range=0.05,shear_range=0.1,zoom_range=0.1)
    gen_valid = ImageDataGenerator()
    test_gen = ImageDataGenerator()
    
    """
    classes = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'] -- cat is 0, dog is 1, so we need write this
    class_mode = categorical, the returned label mode
    shuffle = True, we need, 
    batch_size = 64 
    """
    class_list = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
    
    # create train generator
    train_generator = gen.flow_from_directory(train_dir, image_size, color_mode='rgb', \
                                              classes=class_list, class_mode='categorical', shuffle=True, batch_size=32)
    
    # create validation generator
    valid_generator = gen_valid.flow_from_directory(valid_dir, image_size, color_mode='rgb', \
                                              classes=class_list, class_mode='categorical', shuffle=False, batch_size=32)
    
    test_generator = test_gen.flow_from_directory(test_dir, image_size, color_mode='rgb', \
                                          class_mode=None, shuffle=False, batch_size=32)
    
    return train_generator, valid_generator, test_generator
	
def get_test_result(model_obj, test_generator, model_name="default"):
    print("Now to predict")
    pred_test = model_obj.predict_generator(test_generator, len(test_generator), verbose=1)
    pred_test = np.array(pred_test)
    pred_test = pred_test.clip(min=0.005, max=0.995)
    print("create datasheet")
    result = pd.DataFrame(pred_test, columns=['c0', 'c1', 'c2', 'c3',
                                                 'c4', 'c5', 'c6', 'c7',
                                                 'c8', 'c9'])
    test_filenames = []
    for f in test_generator.filenames:
        test_filenames.append(os.path.basename(f))
    result.loc[:, 'img'] = pd.Series(test_filenames, index=result.index)
    result.to_csv('%s.csv' % (model_name), index=None)
    print ('test result file %s.csv generated!' % (model_name))
    
def model_built(MODEL, input_shape, preprocess_input, classes, last_frozen_layer_name):
    """
        MODEL:                  pretrained model
        input_shape:            pre-trained model's input shape
        preprocessing_input:    pre-trained model's preprocessing function
        last_frozen_layer_name: last layer to frozen  
    """
    
    ## get pretrained model
    x = Input(shape=input_shape)
    
    if preprocess_input:
        x = Lambda(preprocess_input)(x)
    
    notop_model = MODEL(include_top=False, weights='imagenet', input_tensor=x, input_shape=input_shape)
    
    x = GlobalAveragePooling2D()(notop_model.output)

    ## build top layer
    x = Dropout(0.5, name='dropout_1')(x)
    out = Dense(classes, activation='softmax', name='dense_1')(x)
    
    ret_model = Model(inputs=notop_model.input, outputs=out)
    
    ## Frozen some layer
    #for layer in ret_model.layers:
        #layer.trainable = False
        #if layer.name == last_frozen_layer_name:
        #break
    
    return ret_model

resnet50_train_generator, resnet50_valid_generator, resnet50_test_generator = get_data_generator(link_train_path, link_valid_path, test_link, resNet_input_shape[:2])
resnet50_model = model_built(resnet50.ResNet50, resNet_input_shape, resnet50.preprocess_input, 10, None)
sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)

resnet50_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

history = resnet50_model.fit_generator(resnet50_train_generator, len(resnet50_train_generator), epochs=20,workers=4,validation_data=resnet50_valid_generator, validation_steps=len(resnet50_valid_generator))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#get_test_result(resnet50_model, resnet50_test_generator, model_name="resnet-50-result")
resnet50_model.save_weights('model_weights_1_15.h5')
resnet50_model.save('mymodel_1_15.h5')
del resnet50_model
