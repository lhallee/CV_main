import numpy as np
import tensorflow as tf
import skimage
import matplotlib.pyplot as plt
from glob import glob
from logan import plots
from logan import dataprocessing
from logan import self_built
from keras_unet_collection import models, utils, losses
from skimage import metrics
from tensorflow import keras
from keras import backend as K
from timeit import default_timer as timer


#Custom Loss Functions
def hybrid_loss(y_true, y_pred):
    hybrid = keras.losses.categorical_crossentropy(y_true, y_pred) - jacard_coef(y_true, y_pred)
    return hybrid
  
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

#Metrics
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def model_selection(dim,
                    model_type,
                    num_class,
                    backbone,
                    LR,
                    optimizer,
                    loss,
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
    if model_type == 'swin-unet':
        model = models.swin_unet_2d((dim, dim, channel), filter_num_begin=64, n_labels=num_class, depth=5, stack_num_down=2, stack_num_up=2,
                            patch_size=(4, 4), num_heads=[4, 8, 8, 8, 8], window_size=[4, 2, 2, 2, 2], num_mlp=1024,
                            output_activation='Sigmoid', shift_window=True, name='swin_unet')
    if model_type == 'trans-unet':
        model = models.transunet_2d((dim, dim, channel), filter_num=[64, 128, 256, 512], n_labels=num_class, stack_num_down=2, stack_num_up=2,
                                embed_dim=512, num_mlp=2048, num_heads=12, num_transformer=12,
                                activation='ReLU', mlp_activation='GELU', output_activation='Sigmoid',
                                batch_norm=True, pool=True, unpool='bilinear', name='transunet')
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
    print(model.summary())
    return model


class PerformancePlotCallback(keras.callbacks.Callback):
    def __init__(self, x_test, y_test, model_name):
        self.x_test = x_test
        self.y_test = y_test
        self.model_name = model_name

    def contour_plot(self, epoch, y_pred, i):
        levels = np.linspace(0.0, 1.0, 11)
        plt.contourf(y_pred, levels=levels, cmap=plt.cm.coolwarm)
        plt.colorbar()
        plt.tight_layout()
        plt.title(f'Prediction Visualization - Epoch: {epoch}')
        plt.savefig('pred_' + str(i) + "_" + self.model_name + "_" + str(epoch))
        plt.close()

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x_test)
        for i in range(len(y_pred)):
            self.contour_plot(epoch, y_pred[i, ..., 1], i)

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
    for i in range(len(model_type)):
        start = timer()
        model_t = model_type[i]

        model = model_selection(dim,model_t,num_class,backbone,LR,optimizer,loss,channel)

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

        val_performance = PerformancePlotCallback(test_input[0:num_sample], test_label[0:num_sample], str(model_t))

        callbacks_list = [early_stop, cp_callback, val_performance]

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
        print('Testing set Recall = {}'.format(np.mean(recall_m(test_label, y_pred))))
        print('Testing set Precision = {}'.format(np.mean(precision_m(test_label, y_pred))))
        print('Testing set F1 = {}'.format(np.mean(f1_m(test_label, y_pred))))
        print('Testing set MSE = {}'.format(np.mean(metrics.mean_squared_error(test_label, y_pred))))
        print('Testing set Hausdorff Distance = {}'.format(np.mean(metrics.hausdorff_distance(test_label,y_pred))))

        if num_class == 2:
            plots.test_viewer(test_input,test_label,y_pred)
        if num_class == 3:
            plots.multi_test_viewer(test_input,test_label,y_pred)
        end = timer()
        time = end - start
        print(model_type[i] + ' took ' + str(round(time, 2)) + ' seconds.\n')

def evaluate(weight_path,
             img_path,
             mask_path,
             eval_path,
             model_type,
             num_class,
             backbone, LR, optimizer, loss,
             channel, norm, scale, eval_dim):
    dim = eval_dim

    #Load dim model
    model = model_selection(dim,model_type,num_class,backbone,LR,optimizer,loss,channel)
    model.load_weights(weight_path)

    #For windowed reconstruction
    full_img_list = []
    big_imgs_path = sorted(glob(eval_path + '*.png'))
    big_img_path = big_imgs_path[0]
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
    print('Total Windows: ' + str(len(full_stack)))
    full_stack = full_stack.reshape(len(full_stack), dim, dim, C)
    recon = np.zeros((int(H / dim) * dim, int(W / dim) * dim))
    row_num = int(W / step)
    col_num = int(H / step)
    y_pred = model.predict([full_stack])

    #View some of y_pred
    plots.y_pred_viewer(y_pred)

    #Reconstruct from windows
    k = 0
    for i in range(col_num):
        for j in range(row_num):
            recon[i*dim:(i+1)*dim, j*dim:(j+1)*dim] = y_pred[k,...,1]
            k += 1

    plots.eval_viewer(recon)



