import tensorflow as tf
import keras.backend as K
import numpy as np


def binary_cross_entropy(y_true, y_pred):
    loss = (1-y_true)*K.log(1-y_pred+K.epsilon())+y_true*K.log(y_pred+K.epsilon())
    loss = K.mean(loss, axis=[1,2,3])
    return -K.mean(loss)
    pass

def Dice_loss(y_true,y_pred):
	loss = 1-((K.sum(y_true*y_pred,axis=[1,2])+K.epsilon())/(K.sum(y_true+y_pred,axis=[1,2])+K.epsilon()))-((K.sum((1-y_true)*(1-y_pred),axis=[1,2])+K.epsilon())/(K.sum((2-y_true-y_pred),axis=[1,2])+K.epsilon()))
	loss = K.mean(loss,-1)
	return K.mean(loss)
	pass


def loss_arg(loss, args=None):
    if loss == 'CE': return 'binary_cross_entropy', binary_cross_entropy
    if loss == 'DL': return 'Dice_loss', Dice_loss

    return None
