import numpy as np
import tensorflow as tf
from keras import Model
from keras.layers import *
from keras import backend as K
from keras.utils import plot_model
from contextlib import redirect_stdout
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class ArgumentForLoss():
    def __init__(self, block_channels, channels, loss):
        self.model_dest = model_dest
        self.blocks = blocks
        self.block_conv = block_conv
        self.block_channels = block_channels
        self.channels = channels
        self.loss = loss

def output_layer(inputs):
    with tf.variable_scope("output"):
        output = Conv2D(1 ,name='output', kernel_size=(1, 1), padding='same',activation='sigmoid')(inputs)
    return output

def model_summary(model, model_dest):
    plot_model(model,to_file=model_dest+'/model.png',show_shapes=True)
    trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    nontrainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
    with open(model_dest+'/modelsummary.txt', 'w') as f:
        f.write('Number of non-trainable parameters: '+str(nontrainable_count)+'\n')
        f.write('Number of trainable parameters:     '+str(trainable_count)+'\n')
        f.write('Number of total parameters:         '+str(nontrainable_count+trainable_count)+'\n')
        with redirect_stdout(f):
            model.summary()

    print('######################################-Summary-#########################################')
    print('Number of non-trainable parameters:',nontrainable_count)
    print('Number of trainable parameters:    ',trainable_count)
    print('Number of total parameters:        ',nontrainable_count+trainable_count)
    print('########################################################################################')
    return None

def U_net(channels, args):
    blocks, block_conv, block_channels = args.blocks,  args.block_conv,  args.block_channels
    model_dest = args.model_dest
    output_frame = 64
    up_block_channels = [block_channels[-(x+2)] for x in range(len(block_channels)-1)]+ [output_frame]
    maxp_list = list()
    input_img = Input(batch_shape=(None,None,None,channels))
    #----down sampling----
    for i in range(blocks):
        for j in range(block_conv):
            layer_name = 'down_block_'+str(i)+'_conv_'+str(j)
            layer_name_BN = 'down_block_'+str(i)+'_BN_'+str(j)
            if i==0 and j == 0:
                conv = Conv2D(block_channels[i] ,name=layer_name, kernel_size=(3,3), padding='same',activation='relu')(input_img)
                conv = BatchNormalization(name=layer_name_BN)(conv)
            elif i!=0 and j==0:
                conv = Conv2D(block_channels[i] ,name=layer_name, kernel_size=(3,3), padding='same',activation='relu')(maxp_list[i-1])
                conv = BatchNormalization(name=layer_name_BN)(conv)
            else:
                conv = Conv2D(block_channels[i] ,name=layer_name, kernel_size=(3,3), padding='same',activation='relu')(conv)
                conv = BatchNormalization(name=layer_name_BN)(conv)

        maxp = MaxPooling2D(name='down_block_'+str(i)+'_maxp')(conv)
        maxp_list.append(maxp)
    #----down sampling----
    #-----up sampling-----
    for i in range(blocks):
        for j in range(block_conv+1):
            layer_name = 'up_block_'+str(i)+'_conv_'+str(j)
            layer_name_BN = 'up_block_'+str(i)+'_BN_'+str(j)
            layer_name_convt = 'up_block_'+str(i)+'_ConvT'
            if i==0 and j == 0:
                up = Conv2D(block_channels[-1] ,name='mini_maxp_conv1', kernel_size=(3,3), padding='same',activation='relu')(maxp_list[-(i+1)])
                up = BatchNormalization(name='mini_maxp_BN1')(up)
                up = Conv2D(block_channels[-1] ,name='mini_maxp_conv2', kernel_size=(3,3), padding='same',activation='relu')(up)
                up = BatchNormalization(name='mini_maxp_BN2')(up)
                up = Conv2DTranspose(up_block_channels[i],name=layer_name_convt, kernel_size=(2,2),strides=(2, 2), padding='same',
                    activation='relu')(up)
                up = BatchNormalization(name=layer_name_BN)(up)
                
            elif i!=0 and j==0:
                up = concatenate([maxp_list[-(i+1)], up])
                up = Conv2DTranspose(up_block_channels[i],name=layer_name_convt, kernel_size=(2,2),strides=(2, 2), padding='same',
                    activation='relu')(up)
                up = BatchNormalization(name=layer_name_BN)(up)
            else:
                up = Conv2D(up_block_channels[i] ,name=layer_name, kernel_size=(3,3), padding='same',activation='relu')(up)
                up = BatchNormalization(name=layer_name_BN)(up)
    #-----up sampling-----
    output = output_layer(up)
    model = Model(input_img,output)
    model_summary(model, model_dest)

    return model
    pass



if __name__ == '__main__':
    blocks = 5
    block_conv = 2
    block_channels = [12,24,36,48,64]
    model_dest = "./model_dst/"
    args = ArgumentForLoss(blocks, block_conv, block_channels, model_dest)
    channels = 4

    model = U_net(channels, args)