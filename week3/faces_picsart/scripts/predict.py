import os
from PIL import Image
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
from keras import backend as K
from keras.models import Model, load_model
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img#,save_img
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def dice_coef_K(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


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

def predict_result(model ,x_test ,img_size_target): # predict both orginal and reflect x
    x_test_reflect = np.array([np.fliplr(x) for x in x_test])
    x_test_reflect = x_test_reflect
    print('Predicting...')
    preds_test1 = model.predict(x_test, verbose=1)  # .reshape(-1, img_size_target[0], img_size_target[1])
    # preds_test2_refect = model.predict(x_test_reflect).reshape(-1, img_size_target[0], img_size_target[1])
    # preds_test2 = np.array([ np.fliplr(x) for x in preds_test2_refect] )
    # preds_avg = (preds_test1 + preds_test2)/2
    # return preds_avg
    return preds_test1

# Set some parameters
im_width = 101
im_height = 101
im_chan = 1
img_size_ori = (320, 240)
img_size_target = (320, 240)

test_dir = '../data/new_test/'
test_images = np.array([np.array(load_img(join(test_dir, f), grayscale=False)) / 255
                        for f in listdir(test_dir) if isfile(join(test_dir, f))])
test_file_names = [f[:f.find('.')] for f in listdir(test_dir) if isfile(join(test_dir, f))]
model = load_model("../models/unet_weights.11-val_loss0.12--0.95.hdf5.model",
                   custom_objects={'dice_coef_K': dice_coef_K, 'my_dice_metric': my_dice_metric})
valid_images = test_images[:10]

pred_masks = predict_result(model, valid_images, img_size_target)
# save initial image
for i, mask in enumerate(pred_masks):
    # Initial image
    initial_im = Image.fromarray((valid_images[i] * 255).astype(np.uint8))
    initial_im.save("../output/{}_initial.png".format(test_file_names[i]))

    # Mask
    pred_mask = (pred_masks[i]).reshape((320, 240))
    pred_mask = np.round(pred_mask) * 255
    pred_mask = pred_mask.astype(np.uint8)
    # try_one_image = try_one_image.reshape((try_one_image.shape[0], try_one_image.shape[1], 1))
    new_p = Image.fromarray(pred_mask)
    new_p.save("../output/{}_mask.png".format(test_file_names[i]))

    val_image = valid_images[i].copy()
    val_image = np.round(val_image * 255, 0).astype(np.uint8)

    # Возможно, это особенности работы функций opencv. В этом пакете кодировка BGR вместо RGB
    val_image = np.concatenate([val_image[:, :, 2].reshape(val_image.shape[:2] + (1,)),
                                val_image[:, :, 1].reshape(val_image.shape[:2] + (1,)),
                                val_image[:, :, 0].reshape(val_image.shape[:2] + (1,))], axis=2)
    pred_mask_red = np.zeros(pred_mask.shape + (3,), np.uint8)
    pred_mask_red[:, :, 2] = pred_mask.copy()
    blended_image = cv2.addWeighted(pred_mask_red, 1, val_image, 1, 0)
    cv2.imwrite('../output/{}_image_plus_mask.png'.format(test_file_names[i]), blended_image)
