import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import skimage

def cols(H, W):
  x = np.array(range(W))
  y = np.array(range(H))
  return x,y
def ax_decorate_box(ax):
    [j.set_linewidth(0) for j in ax.spines.values()]
    ax.tick_params(axis="both", which="both", bottom=False, top=False,
               labelbottom=False, left=False, right=False, labelleft=False)
    return ax

def crop_viewer(img, mask):
    for i in range(3):
        fig, AX = plt.subplots(1, 2, figsize=(7, 3))
        plt.subplots_adjust(0, 0, 1, 1, hspace=0, wspace=0.1)
        for ax in AX:
            ax = ax_decorate_box(ax)
        AX[0].pcolormesh(np.mean(img[i, ...], axis=-1))
        AX[1].pcolormesh(mask[i, ..., 0], cmap=plt.cm.gray)
        AX[0].set_title("Original", fontsize=14)
        AX[1].set_title("Segmentation mask", fontsize=14)
        plt.show()

def multi_crop_viewer(img,mask):
    for i in range(3):
        fig, AX = plt.subplots(1,4,figsize=(10,3))
        plt.subplots_adjust(0,0,1,1,hspace=0,wspace=0.1)
        for ax in AX:
            ax = ax_decorate_box(ax)
        AX[0].pcolormesh(np.mean(img[i, ...], axis=-1))
        AX[1].pcolormesh(mask[i, ..., 0]>0, cmap=plt.cm.gray)
        AX[2].pcolormesh(mask[i, ..., 1]>0, cmap=plt.cm.gray)
        AX[3].pcolormesh(mask[i, ..., 2]>0, cmap=plt.cm.gray)

        AX[0].set_title("Original", fontsize=14);
        AX[1].set_title("Background", fontsize=14);
        AX[2].set_title("Lobule", fontsize=14);
        AX[3].set_title("HEV", fontsize=14);
        plt.show()
def toy_viewer(img, mask):
    for i in range(3):
        fig, AX = plt.subplots(1, 2, figsize=(7, 3))
        plt.subplots_adjust(0, 0, 1, 1, hspace=0, wspace=0.1)
        for ax in AX:
            ax = ax_decorate_box(ax)
        AX[0].pcolormesh(img[i, ..., 0], cmap=plt.cm.gray)
        AX[1].pcolormesh(mask[i, ..., 0], cmap=plt.cm.gray)
        AX[0].set_title("Original", fontsize=14)
        AX[1].set_title("Segmentation mask", fontsize=14)
        plt.show()

def sign_viewer(mask,dim):
    for i in range(3):
        sign = mask[i].reshape(dim,dim)
        levels = np.linspace(0,dim,1000)
        plt.contourf(sign, levels=levels, cmap=plt.cm.coolwarm)
        plt.colorbar()
        plt.show()

def test_viewer(img, mask, pred):
    for i in range(3):
        fig, AX = plt.subplots(1, 3, figsize=(20, 7))
        plt.subplots_adjust(0, 0, 1, 1, hspace=0, wspace=0.1)
        for ax in AX:
            ax = ax_decorate_box(ax)

        AX[0].pcolormesh(np.mean(img[i, ...,]*255, axis=-1))
        AX[1].pcolormesh(pred[i, ..., 1], cmap=plt.cm.jet)
        AX[2].pcolormesh(mask[i, ..., 1], cmap=plt.cm.jet)

        AX[0].set_title("Scaled", fontsize=14)
        AX[1].set_title("Prediction", fontsize=14)
        AX[2].set_title("Labeled truth", fontsize=14)
        plt.show()

def multi_test_viewer(img, mask, pred):
    for i in range(3):
        fig, AX = plt.subplots(1,7,figsize=(20,3))
        plt.subplots_adjust(0,0,1,1,hspace=0,wspace=0.1)
        for ax in AX:
            ax = ax_decorate_box(ax)
        AX[0].pcolormesh(np.mean(img[i, ...], axis=-1))
        AX[1].pcolormesh(pred[i, ..., 0], cmap=plt.cm.coolwarm)
        AX[2].pcolormesh(pred[i, ..., 1], cmap=plt.cm.coolwarm)
        AX[3].pcolormesh(pred[i, ..., 2], cmap=plt.cm.coolwarm)
        AX[4].pcolormesh(mask[i, ..., 0], cmap=plt.cm.gray)
        AX[5].pcolormesh(mask[i, ..., 1], cmap=plt.cm.gray)
        AX[6].pcolormesh(mask[i, ..., 2], cmap=plt.cm.gray)

        AX[0].set_title("Original", fontsize=14);
        AX[1].set_title("Pred Background", fontsize=14);
        AX[2].set_title("Pred Lobule", fontsize=14);
        AX[3].set_title("Pred HEV", fontsize=14);
        AX[4].set_title("Mask Background", fontsize=14);
        AX[5].set_title("Mask Lobule", fontsize=14);
        AX[6].set_title("Mask HEV", fontsize=14);
        plt.show()
def eval_viewer(img):
    H, W = img.shape
    x_col, y_col = cols(H,W)
    xfit = np.arange(0, W, 0.1)
    yfit = np.arange(0, H, 0.1)
    levels = np.linspace(0.0, 1.0, 21)
    set_func = scipy.interpolate.RectBivariateSpline(y_col,x_col,img)
    func = set_func(x_col,y_col)

    plt.contourf(func, levels=levels, cmap=plt.cm.coolwarm)
    plt.colorbar()
    plt.show()

    filt_img = skimage.filters.threshold_local(img)
    filt_set_func = scipy.interpolate.RectBivariateSpline(y_col,x_col,filt_img)
    filt_func = filt_set_func(x_col,y_col)

    plt.contourf(filt_func, levels=levels, cmap=plt.cm.coolwarm)
    plt.colorbar()
    plt.show()

def multi_eval_viewer(img, mask, pred):
    dim = len(img[:][0])
    x_col, y_col = cols(dim)
    xfit = np.arange(0, dim, 0.1)
    yfit = np.arange(0, dim, 0.1)
    for i in range(3):
        fig, AX = plt.subplots(1,7,figsize=(20,3))
        plt.subplots_adjust(0,0,1,1,hspace=0,wspace=0.1)
        for ax in AX:
            ax = ax_decorate_box(ax)
        AX[0].pcolormesh(np.mean(img[i, ...], axis=-1))
        AX[1].pcolormesh(pred[i, ..., 0], cmap=plt.cm.coolwarm)
        AX[2].pcolormesh(pred[i, ..., 1], cmap=plt.cm.coolwarm)
        AX[3].pcolormesh(pred[i, ..., 2], cmap=plt.cm.coolwarm)
        AX[4].pcolormesh(mask[i, ..., 0], cmap=plt.cm.gray)
        AX[5].pcolormesh(mask[i, ..., 1], cmap=plt.cm.gray)
        AX[6].pcolormesh(mask[i, ..., 2], cmap=plt.cm.gray)

        AX[0].set_title("Original", fontsize=14);
        AX[1].set_title("Pred Background", fontsize=14);
        AX[2].set_title("Pred Lobule", fontsize=14);
        AX[3].set_title("Pred HEV", fontsize=14);
        AX[4].set_title("Mask Background", fontsize=14);
        AX[5].set_title("Mask Lobule", fontsize=14);
        AX[6].set_title("Mask HEV", fontsize=14);
        plt.show()

    for i in range(3):
        img = pred[i,...,0]
        set_func = scipy.interpolate.RectBivariateSpline(x_col,y_col,img)
        detail_func = set_func(xfit,yfit)
        levels = np.linspace(0.0,1.0,21)
        plt.contourf(detail_func, levels=levels, cmap=plt.cm.coolwarm)
        plt.colorbar()
        plt.show()

