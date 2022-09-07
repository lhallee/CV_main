import numpy as np
import tensorflow as tf
import skimage
import matplotlib.pyplot as plt
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
   [21., 22., 23., 24., 25.],
[26., 27., 28., 29., 30.],
[31., 32., 33., 34.,35.]])
arr = arr.reshape(7,5,1)
print(arr)
full_img_list = []


H, W, C= arr.shape
print(H,W,C)
dim =2
step = dim
patch_imgs = skimage.util.view_as_windows(arr, (dim, dim, 1), step=step)
for i in range(len(patch_imgs)):
    for j in range(len(patch_imgs[0])):
        full_img_list.append(patch_imgs[i][j])

full_stack = tf.stack(full_img_list)
full_stack = np.array(full_stack)
full_stack = full_stack.reshape(len(full_stack), dim, dim, 1)

recon = np.zeros((int(H / dim) * dim, int(W / dim) * dim))
row_num = int(W / step)
col_num = int(H / step)

k=0
for i in range(col_num):
    for j in range(row_num):
        recon[i * dim:(i + 1) * dim, j * dim:(j + 1) * dim] = full_stack[k, ..., 0]
        k += 1
print(recon)
plt.contourf(recon)
plt.show()