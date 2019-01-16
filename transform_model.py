import tensorflow as tf
import os 
import os.path as osp
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.applications import resnet50
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model


res_image_size = (224, 224)
res_input_shape = (224, 224, 3)

#路径参数
input_path = 'E:/graduation_project/program/'
weight_file = 'my_model_weights.h5'
weight_file_path = osp.join(input_path,weight_file)
output_graph_name = tf_my_model_weights + '.pb'

#输出路径
output_dir = osp.join(os.getcwd(),"trans_model")

def h5_to_pb(h5_model,output_dir,model_name,out_prefix = "output_",log_tensorboard = True):
    if osp.exists(output_dir) == False:
        os.mkdir(output_dir)
    out_nodes = []
    for i in range(len(h5_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(h5_model.output[i],out_prefix + str(i + 1))
    sess = K.get_session()
    from tensorflow.python.framework import graph_util,graph_io
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess,init_graph,out_nodes)
    graph_io.write_graph(main_graph,output_dir,name = model_name,as_text = False)
    if log_tensorboard:
        from tensorflow.python.tools import import_pb_to_tensorboard
        import_pb_to_tensorboard.import_to_tensorboard(osp.join(output_dir,model_name),output_dir)

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
    h5_to_pb(ret_model,output_dir = output_dir,model_name = output_graph_name)
    
    
    
cam_model, cam_weights = cam_model(resnet50.ResNet50, res_input_shape, resnet50.preprocess_input, 10, 'my_model_weights.h5')

