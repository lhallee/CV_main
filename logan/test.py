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


arr = np.array([
   [1., 2., 3., 4., 5.],
   [6., 7., 8., 9., 10.],
   [11., 12., 13., 14., 15.],
   [16., 17., 18., 19., 20.],
   [21., 22., 23., 24., 25.]])
arr = arr.reshape(5,5,1)

full_img_list = []
#big_imgs_path = sorted(glob(img_path + '*.png'))
#big_img_path = big_imgs_path[0]
#big_img = tf.io.read_file(big_img_path)
#big_img = tf.image.decode_png(big_img, channels=3)
#big_img = np.array(big_img)
#H, W, C = big_img.shape

step = 2
patch_imgs = skimage.util.view_as_windows(arr, (2, 2, 1), step=step)
for i in range(len(patch_imgs)):
    for j in range(len(patch_imgs[0])):
        full_img_list.append(patch_imgs[i][j])
full_stack = tf.stack(full_img_list)
full_stack = np.array(full_stack)
full_stack = full_stack.reshape(len(full_stack), 2, 2, 1)
print(full_stack.shape)
print(int(2.6))