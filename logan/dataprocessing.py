import numpy as np
import tensorflow as tf
import scipy
from scipy.stats import beta
from scipy.ndimage import gaussian_filter
from matplotlib.path import Path as mpath
from logan import plots
from glob import glob
from tensorflow import keras
from sklearn.model_selection import train_test_split

def target_data_process(target_array,num_class):
    if num_class == 2:
        target_array[target_array>0]=1
    if num_class == 3:
        target_array[target_array==0]=1
        target_array = target_array - 1
    return keras.utils.to_categorical(target_array, num_classes=num_class)

def crop(dim,amt,num_class,img_path,mask_path,norm,scale):
    img_list = []
    mask_list = []
    big_imgs = sorted(glob(img_path + '*.png'))
    big_masks = sorted(glob(mask_path + '*.png'))
    for i in range(len(big_imgs)):
        img = tf.io.read_file(big_imgs[i])
        img = tf.image.decode_png(img, channels=3)
        mask = tf.io.read_file(big_masks[i])
        mask = tf.image.decode_png(mask, channels=1)
        mask = np.array(mask)
        if norm:
            img = keras.utils.normalize(np.array(img), axis=1)
        if scale:
            img = img / 255
        for j in range(amt):
            img_crop = tf.image.stateless_random_crop(img, size=[dim, dim, 3], seed=[42,j]) #deterministic crop
            mask_crop = tf.image.stateless_random_crop(mask, size=[dim, dim, 1], seed=[42,j])
            if ((np.count_nonzero(np.array(mask_crop)) / (dim * dim)) > 0.05):
            #If there are more than 5% nonblack pixels in the mask the crop is kept, keeps images better for training
                if (j % 3 == 0):
                    img_crop = tf.image.rot90(img_crop,k=1)
                    mask_crop = tf.image.rot90(mask_crop,k=1)
                if (j % 4 == 0):
                    img_crop = tf.image.flip_left_right(img_crop)
                    mask_crop = tf.image.flip_left_right(mask_crop)
                if (j % 5 == 0):
                    img_crop = tf.image.flip_up_down(img_crop)
                    mask_crop = tf.image.flip_up_down(mask_crop)
                img_list.append(img_crop)
                mask_list.append(mask_crop)
    img_stack = np.array(tf.stack(img_list))
    mask_stack = np.array(tf.stack(mask_list))
    mask_stack = target_data_process(mask_stack,num_class)
    train_input, test_input, train_label, test_label = train_test_split(img_stack, mask_stack, test_size=0.2)

    print(train_input.shape, train_label.shape, test_input.shape, test_label.shape)
    if num_class == 2:
        plots.crop_viewer(train_input,train_label)
    if num_class == 3:
        plots.multi_crop_viewer(train_input,train_label)
    return train_input,train_label,test_input,test_label

def eval_crops(dim,num_class,img_path,mask_path,norm,scale):
    img_list = []
    mask_list = []
    big_imgs = sorted(glob(img_path + '*.png'))
    big_masks = sorted(glob(mask_path + '*.png'))
    for i in range(len(big_imgs)):
        img = tf.io.read_file(big_imgs[i])
        img = tf.image.decode_png(img, channels=3)
        mask = tf.io.read_file(big_masks[i])
        mask = tf.image.decode_png(mask, channels=1)
        mask = np.array(mask)
        if norm:
            img = keras.utils.normalize(np.array(img), axis=1)
        if scale:
            img = img / 255
        img_crop = tf.image.stateless_random_crop(img, size=[dim, dim, 3], seed=[42,42]) #deterministic crop
        mask_crop = tf.image.stateless_random_crop(mask, size=[dim, dim, 1], seed=[42,42])
        img_list.append(img_crop)
        mask_list.append(mask_crop)
    img_stack = np.array(tf.stack(img_list))
    mask_stack = np.array(tf.stack(mask_list))
    mask_stack = target_data_process(mask_stack,num_class)
    return img_stack, mask_stack

def window_eval():
    big_imgs = sorted(glob(img_path + '*.png'))
    big_img = big_imgs[0]
    W = len(big_img[0,:,:])
    H = len(big_img[:,0,:])
    step = 64
    print(W,H)
    patch_imgs = skimage.util.view_as_windows(big_img, (512, 512, 1), step=step)
    reconstructed_img = np.zeros((W, H, 1))
    for x in range(patch_imgs.shape[0]):
        for y in range(patch_imgs.shape[1]):
            x_pos, y_pos = x * step, y * step
            reconstructed_img[x_pos:x_pos + (step * 2), y_pos:y_pos + (step * 2)] = patch_imgs[x, y, 0, ...]
def make(n, side_len, circ_rad, theta, ishift=0, jshift=0,
         sigma_smooth=0., sigma_noise=0., rs=None):
    """
    Create an illusory triangle contour [1] image with random
    size and orientation.

    [1]: https://en.wikipedia.org/wiki/Illusory_contours

    Parameters
    ----------
    n: int
        Image shape will be (n,n)

    side_len: float
        Side length of the triangle in pixels.

    circ_rad: float
        Radius of the circles at the vertices
        of the triangle in pixels.

    theta: float (radians)
        Rotation of the triangle. Zero points the triangle to the right.

    ishift,jshift: integers
        Translate the center of the triangle by ishift and jshift.

    sigma_smooth: float
        Gaussian smoothing parameter (make image borders more diffuse).

    sigma_noise: float
        Additive noise amplitude.

    rs: numpy.random.RandomState, default=None
        Include for reproducible results.
    """
    if circ_rad > 0.5 * side_len:
        raise ValueError(("Circle radius should be less "
                          "than one half the side length."))

    # Triangle height.
    height = 0.5 * np.sqrt(3) * side_len

    # Distance from center of triangle to a vertex.
    tri_rad = (2.0 / 3.0) * height

    # Rotation factor for triangle vertices.
    w = (2.0 / 3.0) * np.pi

    # Get extent of triangle plus outer circles for validation.
    extent = np.zeros((3, 2))
    for i in range(3):
        x = (tri_rad + circ_rad) * np.cos(i * w + theta) + n / 2 + jshift
        y = (tri_rad + circ_rad) * np.sin(i * w + theta) + n / 2 - ishift
        extent[i] = n - y, x

    for e in extent:
        if e[0] < 0 or e[0] > n - 1:
            raise ValueError(("Extent of triangle plus circles exceeds"
                              "image dimensions along axis 0."))
        if e[1] < 0 or e[1] > n - 1:
            raise ValueError(("Extent of triangle plus circles exceeds"
                              "image dimensions along axis 1."))

    vertices = np.zeros((3, 2))
    for i in range(3):
        x = tri_rad * np.cos(i * w + theta) + n / 2 + jshift
        y = tri_rad * np.sin(i * w + theta) + n / 2 - ishift
        vertices[i] = n - y, x

    tri_path = mpath(np.append(vertices, vertices[-1].reshape(1, 2), axis=0),
                     codes=[mpath.MOVETO, mpath.LINETO,
                            mpath.LINETO, mpath.CLOSEPOLY])

    ii, jj = np.indices((n, n))
    coords = np.c_[ii.flatten(), jj.flatten()]

    triangle = tri_path.contains_points(coords).reshape(n, n)

    ucircle = mpath.unit_circle()
    circles = np.zeros((n, n), dtype=np.bool)

    for v in vertices:
        circle = mpath(vertices=ucircle.vertices * circ_rad + v,
                       codes=ucircle.codes)
        circles = np.logical_or(circles,
                                circle.contains_points(coords).reshape(n, n))

    image = (~np.logical_and(circles, ~triangle)).astype(np.float)
    rs = rs if rs is not None else np.random.RandomState()

    if sigma_smooth > 0:
        image = gaussian_filter(image, sigma_smooth)

    if sigma_noise > 0:
        image += sigma_noise * rs.randn(n, n)

    return image, triangle


def make_dataset(N, n=101, slen=[40, 60], crad=[10, 20],
                 shift=[-0, 0], nsig=[0.05, 0.15], ssig=[1, 1],
                 theta=[0, 2 * np.pi / 3], random_state=None, verbose=False):
    """
    Make a randomly generated dataset of illusory triangle data.

    Parameters
    ----------
    N: int
        The number of examples.

    n: int
        The image size.

    slen: list, len=2
        Interval of triangle side lengths from which to sample.

    crad: list, len=2
        Interval of circle radii from which to sample.

    shift: list, len=2
        The interval of shift values from which to sample.

    nsig: list, len=2
        The interval of values from which to sample `sigma_noise`.

    ssig: list, len=2
        The interval of values from which to sample `sigma_smooth`.

    ctheta: list, len=2
        The interval of values form which to sample `theta`.

    return_meta: bool, default=False
        Return a list of meta data attributes for each example if True.

    random_state: numpy.random.RandomState, default=None
        Include a for reproducible results.

    verbose: bool, default=True
        Print progress.
    """
    random_state = random_state if random_state is not None else np.random.RandomState()

    def betarvs(**kwargs):
        return beta.rvs(3, 3, random_state=random_state, **kwargs)

    if verbose:
        q = len(str(N))
        pstr = "Creating dataset ... %%0%dd / %d" % (q, N)

    imgs = np.zeros((N, n, n))
    segs = np.zeros((N, n, n), dtype=np.bool)

    i = 0

    while i < N:
        try:
            sl = betarvs(loc=slen[0], scale=slen[1] - slen[0])
            cr = betarvs(loc=crad[0], scale=crad[1] - crad[0])

            ishift, jshift = betarvs(loc=shift[0],
                                     scale=shift[1] - shift[0], size=2)
            th = betarvs(loc=theta[0], scale=theta[1] - theta[0])

            sigma_noise = betarvs(loc=nsig[0], scale=nsig[1] - nsig[0])
            sigma_smooth = betarvs(loc=ssig[0], scale=ssig[1] - ssig[0])

            meta = dict(
                side_len=sl,
                circ_rad=cr,
                ishift=ishift,
                jshift=jshift,
                theta=th,
                sigma_smooth=sigma_smooth,
                sigma_noise=sigma_noise
            )

            img, seg = make(n, sl, cr, th, ishift, jshift,
                            sigma_smooth, sigma_noise, rs=random_state)
            imgs[i] = img
            segs[i] = seg
            i += 1

            if verbose: print(pstr % i)
        except ValueError:
            continue
    return imgs, segs

def toy(dim,num_crops):
    train, train_GT = make_dataset(num_crops,dim)
    train_input = np.array(train).reshape(num_crops,dim,dim,1)
    train_GT[train_GT == True] = 1
    train_GT[train_GT == False] = 0
    train_label = np.array(train_GT).reshape(num_crops,dim,dim,1)
    train_label = target_data_process(train_label, 2)
    train_input,test_input,train_label,test_label = train_test_split(train_input,train_label,test_size=0.2)

    print(train_input.shape, train_label.shape)
    plots.toy_viewer(train_input,train_label)
    return train_input,train_label,test_input,test_label

def signed_distance_transform(masks,dim):
    masks = np.copy(masks[:,:,:,1].reshape(len(masks),dim,dim,1))
    for i in range(len(masks)):
        inv = masks[i].reshape(dim,dim)
        inv = scipy.ndimage.morphology.distance_transform_edt(inv) #inward transform
        inv[inv == 0] = 1000 #set zeros for arbitrarily high value
        inv[inv != 1000] = 0
        inv[inv == 1000] = 1 #1 around the boundary of the masks
        dist = scipy.ndimage.morphology.distance_transform_edt(inv) #outward transform
        dist = dist.reshape(dim,dim,1)
        masks[i] = dist
    print(masks.shape)
    return masks

