import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import os
import sys
import random
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from keras.models import Model, load_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add,Flatten,Dense,Reshape
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img#,save_img


# Set some parameters
im_width = 101
im_height = 101
im_chan = 1

img_size_ori = (320, 240)
img_size_target = (320, 240)


def upsample(img):  # not used
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
    # res = np.zeros((img_size_target, img_size_target), dtype=img.dtype)
    # res[:img_size_ori, :img_size_ori] = img
    # return res


def downsample(img):  # not used
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)
    # return img[:img_size_ori, :img_size_ori]


# Loading of training/testing ids and depths
# train_df = pd.read_csv("/storage/tgs_salt/train.csv", index_col="id", usecols=[0])
# depths_df = pd.read_csv("/storage/tgs_salt/depths.csv", index_col="id")
# train_df = train_df.join(depths_df)
# test_df = depths_df[~depths_df.index.isin(train_df.index)]
#
# len(train_df)

train_dir = '../data/train/'
images = [np.array(load_img(join(train_dir, f), grayscale=False)) / 255
                for f in listdir(train_dir) if isfile(join(train_dir, f))]
masks_dir = '../data/train_mask/'
#masks = [np.array(load_img(join(masks_dir, f), grayscale=True)) / 255
#               for f in listdir(masks_dir) if isfile(join(masks_dir, f))]

# Create train/validation split stratified by salt coverage
m = len(images)
train_obs = np.random.choice(range(m), size=round(0.8 * m), replace=False)
valid_obs = np.delete(range(m), train_obs)

# train_images = np.array([images[i] for i in train_obs])
# train_masks = np.array([masks[i].reshape(img_size_target + (1,)) for i in train_obs])
valid_images = np.array([images[i] for i in valid_obs])
# valid_masks = np.array([masks[i].reshape(img_size_target + (1,)) for i in valid_obs])


def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = Activation('relu')(x)
    return x


def residual_block(blockInput, num_filters=16):
    x = Activation('relu')(blockInput)
    x = BatchNormalization()(x)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    return x


# Build model
def build_model(input_layer, start_neurons, DropoutRatio=0.5):
    # (320, 240) -> (160, 120)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(input_layer)
    conv1 = residual_block(conv1, start_neurons * 1)
    conv1 = residual_block(conv1, start_neurons * 1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(DropoutRatio / 2)(pool1)

    # (160, 120) -> (80, 60)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2, start_neurons * 2)
    conv2 = residual_block(conv2, start_neurons * 2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # (80, 60) -> (40, 30)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3, start_neurons * 4)
    conv3 = residual_block(conv3, start_neurons * 4)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # (40, 30) -> (20, 15)
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
    conv4 = residual_block(conv4, start_neurons * 8)
    conv4 = residual_block(conv4, start_neurons * 8)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # (20, 15) -> (10, 7)
    conv5 = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(pool4)
    conv5 = residual_block(conv5, start_neurons * 16)
    conv5 = residual_block(conv5, start_neurons * 16)
    conv5 = Activation('relu')(conv5)
    pool5 = MaxPooling2D((2, 2))(conv5)
    pool5 = Dropout(DropoutRatio)(pool5)

    # Middle
    convm = Conv2D(start_neurons * 32, (3, 3), activation=None, padding="same")(pool5)
    convm = residual_block(convm, start_neurons * 32)
    convm = residual_block(convm, start_neurons * 32)
    convm = Activation('relu')(convm)

    col = Reshape((20, 256, 15))(conv5)
    col = Conv2D(1, 1, activation=None, padding='same')(col)
    col = Flatten()(col)

    # (20, 15) -> (40, 30)
    deconv5 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same", use_bias=True)(convm)
    deconv5 = Flatten()(deconv5)
    deconv5 = concatenate([deconv5, col])
    deconv5 = Reshape((20, 15, 256))(deconv5)
    uconv5 = concatenate([deconv5, conv5])
    uconv5 = Dropout(DropoutRatio)(uconv5)

    uconv5 = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(uconv5)
    uconv5 = residual_block(uconv5, start_neurons * 16)
    uconv5 = residual_block(uconv5, start_neurons * 16)
    uconv5 = Activation('relu')(uconv5)

    # (20, 15) -> (40, 30)
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv5)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)

    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4, start_neurons * 8)
    uconv4 = residual_block(uconv4, start_neurons * 8)
    uconv4 = Activation('relu')(uconv4)

    # 12 -> 25
    # deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(DropoutRatio)(uconv3)

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_neurons * 4)
    uconv3 = residual_block(uconv3, start_neurons * 4)
    uconv3 = Activation('relu')(uconv3)

    # 25 -> 50
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])

    uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_neurons * 2)
    uconv2 = residual_block(uconv2, start_neurons * 2)
    uconv2 = Activation('relu')(uconv2)

    # 50 -> 101
    # deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])

    uconv1 = Dropout(DropoutRatio)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, start_neurons * 1)
    uconv1 = residual_block(uconv1, start_neurons * 1)
    uconv1 = Activation('relu')(uconv1)

    uconv1 = Dropout(DropoutRatio / 2)(uconv1)
    output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)

    return output_layer


# Score the model and do a threshold optimization by the best IoU.

# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in

    true_objects = 2
    pred_objects = 2

    # Jiaxin fin that if all zeros, then, the background is treated as object
    temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=([0, 0.5, 1], [0, 0.5, 1]))
    #     temp1 = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))
    # print(temp1)
    intersection = temp1[0]
    # print("temp2 = ",temp1[1])
    # print(intersection.shape)
    # print(intersection)
    # Compute areas (needed for finding the union between all objects)
    # print(np.histogram(labels, bins = true_objects))
    area_true = np.histogram(labels, bins=[0, 0.5, 1])[0]
    # print("area_true = ",area_true)
    area_pred = np.histogram(y_pred, bins=[0, 0.5, 1])[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    intersection[intersection == 0] = 1e-9

    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)


def iou_metric_batch(y_true_in, y_pred_in):
    y_pred_in = y_pred_in > 0.5  # added by sgx 20180728
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    # print("metric = ",metric)
    return np.mean(metric)


def my_iou_metric(label, pred):
    metric_value = tf.py_func(iou_metric_batch, [label, pred], tf.float64)
    return metric_value


def dice_coef_K(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return -(2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_np(y_true, y_pred, smooth=1):
    intersection = (y_true.flatten() * y_pred.flatten()).sum()
    return -(2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)


def dice_coef_batch(y_true_in, y_pred_in):
    y_pred_in = (y_pred_in > 0.5).astype(np.float32)  # added by sgx 20180728
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = dice_coef_np(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)


def my_dice_metric(label, pred):
    metric_value = tf.py_func(dice_coef_batch, [label, pred], tf.float64)
    return metric_value


TRAIN = False
load_model_bool = True

if TRAIN:
    # Data augmentation
    x_train2 = np.append(train_images, [np.fliplr(x) for x in train_images], axis=0)
    y_train2 = np.append(train_masks, [np.fliplr(x) for x in train_masks], axis=0)

    # x_train2 = train_images
    # y_train2 = train_masks

    # sample = np.random.choice(range(x_train2.shape[0]), size=1000, replace=False)
    # x_train2 = np.array([x_train2[i, :, :, :] for i in sample])
    # y_train2 = np.array([y_train2[i, :, :, :] for i in sample])

    print(x_train2.shape)
    print(valid_masks.shape)


    if load_model_bool:
        model = load_model("resnet_weights.01--0.84.hdf5.model",custom_objects={'my_dice_metric': my_dice_metric})
    else:
        # model
        input_layer = Input(img_size_target + (3,))
        # input_layer2 = Input((img_size_target, img_size_target, 1))
        output_layer = build_model(input_layer, 16, 0.5)

        # del model
        model = Model(input_layer, output_layer)
        model.compile(loss='binary_crossentropy', optimizer="adam", metrics=[my_dice_metric])

        model.summary()

    # early_stopping = EarlyStopping(monitor='val_my_dice_metric', mode = 'max',patience=20, verbose=1)
    model_checkpoint = ModelCheckpoint("resnet_weights.{epoch:02d}-{val_my_dice_metric:.2f}.hdf5.model", save_best_only=True, verbose=1)
    # reduce_lr = ReduceLROnPlateau(monitor='val_my_dice_metric', mode='max', factor=0.2, patience=10, min_lr=0.00001, verbose=1)
    #reduce_lr = ReduceLROnPlateau(factor=0.2, patience=5, min_lr=0.00001, verbose=1)

    epochs = 50
    batch_size = 32
    verbose = 1

    print('VERBOSE={}'.format(verbose))
    print(x_train2.shape)
    history = model.fit(x_train2, y_train2,
                        validation_data=[[valid_images], [valid_masks]],
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[model_checkpoint],
                        verbose=verbose)
    print('Fitted!')
else:
    model = load_model("../models/resnet_weights.17--0.95.hdf5.model",custom_objects={'my_dice_metric': my_dice_metric})
    def predict_result(model,x_test,img_size_target): # predict both orginal and reflect x
        x_test_reflect = np.array([np.fliplr(x) for x in x_test])
        x_test_reflect = x_test_reflect
        print('Predicting...')
        preds_test1 = model.predict(x_test, verbose=1)#.reshape(-1, img_size_target[0], img_size_target[1])
        #preds_test2_refect = model.predict(x_test_reflect).reshape(-1, img_size_target[0], img_size_target[1])
        #preds_test2 = np.array([ np.fliplr(x) for x in preds_test2_refect] )
        #preds_avg = (preds_test1 + preds_test2)/2
        #return preds_avg
        return preds_test1

    from PIL import Image
    valid_images = valid_images[:1]

    # save initial image
    initial_im = Image.fromarray((valid_images[0] * 255).astype(np.uint8))
    initial_im.save("../output/your_initial_im.png")

    pred_masks = predict_result(model, valid_images, img_size_target)
    # как наложить маску на фото? сложить фото и 0.2 * маску

    pred_mask = (pred_masks[0]).reshape((320, 240))
    pred_mask = np.round(pred_mask) * 255
    pred_mask = pred_mask.astype(np.uint8)
    # try_one_image = try_one_image.reshape((try_one_image.shape[0], try_one_image.shape[1], 1))
    new_p = Image.fromarray(pred_mask)
    new_p.save("../output/your_mask.png")


    # image_with_mask = valid_images[0] * 255
    # # image_with_mask[:, :, 0] = image_with_mask[:, :, 0] + 0.2 * try_one_image
    # # image_with_mask[:, :, 1] = image_with_mask[:, :, 1] + 0.2 * try_one_image
    # image_with_mask[:, :, 2] = image_with_mask[:, :, 2] + 0.8 * pred_mask
    #
    # image_with_mask = image_with_mask.astype(np.uint8)
    # #image_with_mask = valid_images[0] + 0.2 * try_one_image
    # im_mask = Image.fromarray(image_with_mask)
    # im_mask.save("image_plus_mask.png")

    val_image = valid_images[0].copy()
    val_image = np.round(val_image * 255, 0).astype(np.uint8)

    # Возможно, это особенности работы функций opencv. В этом пакете кодировка BGR вместо RGB
    val_image = np.concatenate([val_image[:, :, 2].reshape(val_image.shape[:2] + (1,)),
                                val_image[:, :, 1].reshape(val_image.shape[:2] + (1,)),
                                val_image[:, :, 0].reshape(val_image.shape[:2] + (1,))], axis=2)
    pred_mask_red = np.zeros(pred_mask.shape + (3,), np.uint8)
    pred_mask_red[:, :, 2] = pred_mask.copy()
    blended_image = cv2.addWeighted(pred_mask_red, 1, val_image, 1, 0)
    cv2.imwrite('../output/image_plus_mask.png', blended_image)

    raise ValueError

    preds_valid2 = np.array([downsample(x) for x in preds_valid])
    y_valid2 = np.array([downsample(x) for x in valid_masks])

    ## Scoring for last model
    thresholds = np.linspace(0.3, 0.7, 31)
    ious = np.array([dice_coef_batch(y_valid2, np.int32(preds_valid2 > threshold)) for threshold in tqdm_notebook(thresholds)])

    threshold_best_index = np.argmax(ious)
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]

    """
    used for converting the decoded image to rle mask
    Fast compared to previous one
    """
    def rle_encode(im):
        '''
        im: numpy array, 1 - mask, 0 - background
        Returns run length as string formated
        '''
        pixels = im.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)


    test_dir = '../data/test/'
    test_images = np.array([np.array(load_img(join(test_dir, f), grayscale=False)) / 255
                           for f in listdir(test_dir) if isfile(join(test_dir, f))])
    test_file_names = [f[:f.find('.')] for f in listdir(test_dir) if isfile(join(test_dir, f))]
    preds_test = predict_result(model,test_images,img_size_target)

    import time
    t1 = time.time()
    pred_dict = {idx: rle_encode(np.round(preds_test[i] > threshold_best))
                 for i, idx in enumerate(tqdm_notebook(test_file_names))}
    t2 = time.time()

    print('Used tume = {}'.format(t2-t1))
    #print(f"Usedtime = {t2-t1} s")

    sub = pd.DataFrame.from_dict(pred_dict,orient='index')
    sub.index.names = ['image']
    sub.columns = ['rle_mask']
    sub.to_csv('submission_un2.csv')
