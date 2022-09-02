import numpy as np
import tensorflow as tf
from logan import plots
from logan import dataprocessing
from logan import self_built
from keras_unet_collection import models, utils, losses
from tensorflow import keras
from keras import backend as K


#Custom Loss Functions
def hybrid_loss(y_true, y_pred):
    loss_focal = losses.focal_tversky(y_true, y_pred, alpha=0.5, gamma=4/3)
    loss_iou = losses.iou_seg(y_true, y_pred)
    return loss_focal+loss_iou
  
def reconstruction_loss(real, reconstruction):
  return tf.reduce_mean(
      tf.reduce_sum(
          keras.losses.categorical_crossentropy(real, reconstruction),
          axis=(1,2)
      )
  )

def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def jacard_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)

def train(train_input,
          train_label,
          test_input,
          test_label,
          dim,
          model_type,
          num_epoch,
          num_batch,
          num_sample,
          num_class,
          patience,
          min_del,
          backbone,
          LR,
          optimizer,
          loss,
          save_weight,
          channel):

    if model_type == 'unet':
        model = models.att_unet_2d((dim, dim, channel), filter_num=[64, 128, 256, 512, 1024], n_labels=num_class,
                                   stack_num_down=2, stack_num_up=2, activation='ReLU',
                                   output_activation='Sigmoid',
                                   batch_norm=True, pool=False, unpool=False,
                                   backbone=backbone, weights='imagenet',
                                   freeze_backbone=True, freeze_batch_norm=True,
                                   name='unet')
    if model_type == 'att-unet':
        model = models.att_unet_2d((dim, dim, channel), filter_num=[64, 128, 256, 512, 1024], n_labels=num_class,
                                   stack_num_down=2, stack_num_up=2, activation='ReLU',
                                   atten_activation='ReLU', attention='add', output_activation='Sigmoid',
                                   batch_norm=True, pool=False, unpool=False,
                                   backbone=backbone, weights='imagenet',
                                   freeze_backbone=True, freeze_batch_norm=True,
                                   name='att-unet')
    if model_type == 'r2-unet':
        model = models.att_unet_2d((dim, dim, channel), filter_num=[64, 128, 256, 512, 1024], n_labels=num_class,
                                   stack_num_down=2, stack_num_up=2, activation='ReLU',
                                   output_activation='Sigmoid',
                                   recur_num=2,
                                   batch_norm=True, pool=False, unpool=False,
                                   backbone=backbone, weights='imagenet',
                                   freeze_backbone=True, freeze_batch_norm=True,
                                   name='r2-unet')

    if optimizer == 'Adam':
        optimizer = keras.optimizers.Adam(LR)
    if optimizer == 'SGD':
        optimizer = keras.optimizers.SGD(LR)
    if optimizer == 'Adadelta':
        optimizer = keras.optimizers.Adadelta(LR)

    if loss == 'categorical crossentropy':
        loss = keras.losses.categorical_crossentropy
    if loss == 'binary crossentropy':
        loss = keras.losses.binary_crossentropy
    if loss == 'reconstruction':
        loss = reconstruction_loss
    if loss == 'jaccard':
        loss = jacard_loss
    if loss =='hybrid':
        loss = hybrid_loss

    model.compile(loss=loss, optimizer=optimizer)

    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=min_del,
        patience=patience,
        restore_best_weights=True
    )

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_weight,
        save_weights_only=True,
        save_best_only=True,
        verbose=1
    )
    callbacks_list = [early_stop,cp_callback]

    model.fit(
        train_input,
        train_label,
        batch_size=num_sample,
        epochs=num_epoch,
        steps_per_epoch=num_batch,
        validation_split=0.2,
        callbacks=callbacks_list,
        shuffle=True
    )

    y_pred = model.predict([test_input])
    print('Testing set cross-entropy = {}'.format(np.mean(keras.losses.categorical_crossentropy(test_label, y_pred))))
    if num_class == 2:
        plots.test_viewer(test_input,test_label,y_pred)
    if num_class == 3:
        plots.multi_test_viewer(test_input,test_label,y_pred)

def evaluate(weight_path,
             img_path,
             mask_path,
             model_type,
             num_class,
             backbone, LR, optimizer, loss,
             channel, norm, scale, eval_dim):
    dim = eval_dim
    big_crops, big_masks = dataprocessing.eval(dim,num_class,img_path,mask_path,norm,scale)
    if model_type == 'unet':
        model = models.att_unet_2d((dim, dim, channel), filter_num=[64, 128, 256, 512, 1024], n_labels=num_class,
                                   stack_num_down=2, stack_num_up=2, activation='ReLU',
                                   output_activation='Sigmoid',
                                   batch_norm=True, pool=False, unpool=False,
                                   backbone=backbone, weights='imagenet',
                                   freeze_backbone=True, freeze_batch_norm=True,
                                   name='unet')
    if model_type == 'att-unet':
        model = models.att_unet_2d((dim, dim, channel), filter_num=[64, 128, 256, 512, 1024], n_labels=num_class,
                                   stack_num_down=2, stack_num_up=2, activation='ReLU',
                                   atten_activation='ReLU', attention='add', output_activation='Sigmoid',
                                   batch_norm=True, pool=False, unpool=False,
                                   backbone=backbone, weights='imagenet',
                                   freeze_backbone=True, freeze_batch_norm=True,
                                   name='att-unet')
    if model_type == 'r2-unet':
        model = models.att_unet_2d((dim, dim, channel), filter_num=[64, 128, 256, 512, 1024], n_labels=num_class,
                                   stack_num_down=2, stack_num_up=2, activation='ReLU',
                                   output_activation='Sigmoid',
                                   recur_num=2,
                                   batch_norm=True, pool=False, unpool=False,
                                   backbone=backbone, weights='imagenet',
                                   freeze_backbone=True, freeze_batch_norm=True,
                                   name='r2-unet')

    if optimizer == 'Adam':
        optimizer = keras.optimizers.Adam(LR)
    if optimizer == 'SGD':
        optimizer = keras.optimizers.SGD(LR)
    if optimizer == 'Adadelta':
        optimizer = keras.optimizers.Adadelta(LR)

    if loss == 'categorical crossentropy':
        loss = keras.losses.categorical_crossentropy
    if loss == 'binary crossentropy':
        loss = keras.losses.binary_crossentropy
    if loss == 'reconstruction':
        loss = reconstruction_loss
    if loss == 'jaccard':
        loss = jacard_loss
    if loss == 'hybrid':
        loss = hybrid_loss

    model.compile(loss=loss, optimizer=optimizer)
    model.load_weights(weight_path)
    y_pred = model.predict([big_crops])
    print('Testing set cross-entropy = {}'.format(np.mean(keras.losses.categorical_crossentropy(big_masks, y_pred))))
    if num_class == 2:
        plots.eval_viewer(big_crops, big_masks, y_pred)
    if num_class == 3:
        plots.multi_eval_viewer(big_crops, big_masks, y_pred)

