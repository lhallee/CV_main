import os
from logan import dataprocessing
from logan import run_model
from logan import plots
from datetime import date

def initialize(
    #Paths
    img_path = '/img/', #Where your images are, string
    mask_path = '/mask/', #Where your masks are, string
    eval_path = '/eval/', #Where evaluation images are, string
    save_weight = '/saved weights/', #Where you want to save the weights of the model, -string
    same_weight = True, #if true, uses the weights just saved, -bool
    weight_path = 'string', #specify path to use if same_weight = False, -string
    #Parameters
    run_type = 'hev', #type or run: hev, lobule, combo, -string
    data_type = 'real', #Real to use real data, toy to generate fake data, -string
    dim = 128, #model dimension, crop size, toy data size, -int
    eval_dim = 1024, #the dimension of image data for evaluation, -int
    num_class = 2, #(num_class - 1) is the number of classes you are looking to segment, -int
    num_crops = 1000, #number of crops on each training image, -int
    train = True, #Will train on generated train and validation data, then test on generated test data, -bool
    evaluate = False, #When true, will load weights from weight_path and evaluate full size images for 3D reconstruction, -bool
    dist_transform = False, #When true, apply signed distance transform to masks, -bool
    #For model
    model_type = 'att-unet', #Model type of choice, -string
    #Options: unet, att-unet, r2-unet, self-unet, self-multiunet
    num_epoch = 1000, #number of epochs, -int
    num_batch = 100, #number of batches per epoch, -int
    num_sample = 32, #number of samples per batch, -int
    patience = 50, #the max-allowed early stopping patience, -int
    min_del = 0, #the lowest acceptable loss value reduction, -float
    backbone = None, #Pretrained backbone for certain models, -string
    #Options: None, VGG16, VGG19, ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2, DenseNet121, DenseNet169, DenseNet201, EfficientNetB[0,7]
    #All trained on imagenet dataset
    optimizer = 'Adam', #optimizer of choice, -string
    #Options: Adam, SGD, Adadelta
    LR = 1e-3, #enter initial learning rate or learning rate scheme, -float
    loss = 'categorical crossentropy', #desired loss function, -string
    #Options: categorical crossentropy, binary crossentropy, reconstruction, -jaccard
    unpool = False, #False or 2D interpolation in pooling layers, -bool
    norm = False, #If true, normalize the images by their mean and std, -bool
    scale = False, #If true, divide images by 255, -bool
):
    #working_dir = os.path.dirname(__file__)
    working_dir = '/content/drive/MyDrive/Logan/CV/IMG - Weights'
    #working_dir = 'C:/Users/Logan/Desktop/Logan/School/UD/Research/Gleghorn/Code/Dir'
    img_path = working_dir + img_path
    mask_path = working_dir + mask_path
    eval_path = working_dir + eval_path
    save_weight = working_dir + save_weight + str(dim) + '-' + str(backbone) + '-' + run_type + '-' + loss + '-' + optimizer + '-' + str(LR) + '-' + str(date.today()) + '.hdf5'

    channel = 3
    if train:
        if data_type == 'real':
            train_input, train_label, test_input, test_label = dataprocessing.crop(dim,
                                                                                   num_crops,
                                                                                   num_class,
                                                                                   img_path,
                                                                                   mask_path,
                                                                                   norm,
                                                                                   scale)
        if data_type == 'toy':
            train_input, train_label, test_input, test_label = dataprocessing.toy(dim,num_crops)
            channel = 1
        if dist_transform:
            train_label = dataprocessing.signed_distance_transform(train_label, dim)
            test_label = dataprocessing.signed_distance_transform(test_label, dim)
            plots.sign_viewer(train_label, dim)

        run_model.train(train_input, train_label, test_input, test_label, dim, model_type,
                        num_epoch, num_batch, num_sample, num_class, patience, min_del, backbone, LR, optimizer, loss,
                        save_weight, channel)

    if evaluate:
        if same_weight:
            weights = save_weight
        else:
            weights = weight_path

        run_model.evaluate(weights,
                           img_path,
                           mask_path,
                           eval_path,
                           model_type,
                           num_class,
                           backbone, LR, optimizer, loss,
                           channel, norm, scale, eval_dim
                           )
'''
initialize(norm=True,evaluate=True,eval_dim=224,train=False,same_weight=False,
           weight_path='C:/Users/Logan/Desktop/Logan/School/UD/Research/Gleghorn/Code/Dir/saved weights/512-None-lobule-jaccard-Adam-0.001-2022-09-02.hdf5')
'''