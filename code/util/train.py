import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import numpy as np
from util.loss import *
from keras.callbacks import Callback
import warnings, keras


class ModelCheckpointForBatch(Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                    save_best_only=False, save_weights_only=False,
                    mode='auto', period=1):
        super(ModelCheckpointForBatch, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                            'fallback to auto mode.' % (mode),
                            RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                    'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\t %s improved from %0.5f to %0.5f,'
                                    ' saving model to %s'
                                    % (self.monitor, self.best,
                                        current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose >= 1:
                            print('\t %s did not improve from %0.5f' %
                                    (self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\t saving model to %s' % (filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


def create_callbacks(folder):
    checkpoint_loss = ModelCheckpointForBatch(folder+'/model_loss.h5',
                                    monitor = 'val_loss', verbose = 0.5,
                                    save_best_only = True, mode = 'min')
    checkpoint_loss_train = ModelCheckpointForBatch(folder+'/model_train_loss.h5',
                                    monitor = 'loss', verbose = 0.5,
                                    save_best_only = True, mode = 'min')
    return [checkpoint_loss_train, checkpoint_loss]


def binary_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=[1,2])

def om(y_true, y_pred):
    intersection = K.sum(y_true*K.round(y_pred),axis=[1,2])
    trueSet = K.sum(y_true,axis=[1,2])
    predSet = K.sum(K.round(y_pred),axis=[1,2])
    union = trueSet + predSet - intersection
    overlap_metric = (intersection + K.epsilon()) / (union + K.epsilon())
    return K.mean(overlap_metric)

def save_history(history,path):
    his = np.array(history.history)
    np.save(path,his)

def loss_arg(loss):
    if loss == 'CE': return 'binary_cross_entropy', binary_cross_entropy
    if loss == 'DL': return 'Dice_loss', Dice_loss

    return None


def fit_training(X,y, X_val, y_val, model, loss_fcn, batch_size, epochs, args):
    print('loss function:', loss_fcn)
    print("Input data shape:{}".format(X.shape))
    call_backs = create_callbacks(args.model_dest)
    model.compile(loss=loss_fcn,optimizer='adam',metrics=[om])
    history = model.fit(X, y, validation_data=(X_val, y_val), callbacks=call_backs, 
        batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True)
    save_history(history,args.model_dest+'/history.npy')
