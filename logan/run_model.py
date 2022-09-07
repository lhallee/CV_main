import numpy as np
import tensorflow as tf
import skimage
from glob import glob
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
    if model_type == 'self-unet':
        model = self_built.simple_unet_model(dim,dim,channel)
    if model_type == 'self-multiunet':
        model = self_built.multi_unet_model(dim,dim,channel,num_class)

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
    model.summary()
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
             model_type,
             num_class,
             backbone, LR, optimizer, loss,
             channel, norm, scale, eval_dim):
    dim = eval_dim
    #big_crops, big_masks = dataprocessing.eval_crops(dim,num_class,img_path,mask_path,norm,scale)
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
    if model_type == 'self-unet':
        model = self_built.simple_unet_model(dim,dim,channel)
    if model_type == 'self-multiunet':
        model = self_built.multi_unet_model(dim,dim,channel,num_class)

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

    full_img_list = []
    big_imgs_path = sorted(glob(img_path + '*.png'))
    big_img_path = big_imgs_path[1]
    big_img = tf.io.read_file(big_img_path)
    big_img = tf.image.decode_png(big_img, channels=3)
    big_img = np.array(big_img)

    if norm:
        big_img = keras.utils.normalize(np.array(big_img), axis=1)
    if scale:
        big_img = big_img / 255

    H, W, C = big_img.shape
    step = dim
    patch_imgs = skimage.util.view_as_windows(big_img, (dim, dim, C), step=step)
    for i in range(len(patch_imgs)):
        for j in range(len(patch_imgs[1])):
            full_img_list.append(patch_imgs[i][j])
    full_stack = tf.stack(full_img_list)
    full_stack = np.array(full_stack)
    full_stack = full_stack.reshape(len(full_stack), dim, dim, C)
    recon = np.zeros((int(H / dim) * dim, int(W / dim) * dim))
    row_num = int(W / step)
    col_num = int(H / step)
    y_pred = model.predict([full_stack])

    plots.y_pred_viewer(y_pred,dim)

    k = 0
    for i in range(col_num):
        for j in range(row_num):
            recon[i*dim:(i+1)*dim, j*dim:(j+1)*dim] = y_pred[k,...,1]
            k += 1

    plots.eval_viewer(recon)


    '''
    if num_class == 2:
        plots.eval_viewer(big_crops, big_masks, y_pred)
    if num_class == 3:
        plots.multi_eval_viewer(big_crops, big_masks, y_pred)
    '''
