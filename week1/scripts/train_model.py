import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import keras
from keras.applications import vgg16, inception_v3, resnet50, mobilenet, xception, vgg19, InceptionResNetV2, mobilenetv2
from keras.applications.densenet import DenseNet121
from keras.applications.nasnet import NASNetMobile
from keras.models import Sequential
from keras.preprocessing import image
from keras.layers.core import Activation, Reshape, Dense, Flatten
from keras.layers import Conv2D
from keras.models import Model
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import DepthwiseConv2D
#from keras.applications.mobilenet import relu6
from keras.utils.generic_utils import CustomObjectScope
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
PROJECT_PATH = os.path.dirname(os.getcwd())
#PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_PATH)
from engine.tools.calculate_usage_memory import get_model_memory_usage
from engine.logger import TFLogger
from engine.tools.filesystem_functions import count_folders, get_barcode_class
from engine.tools.add_metrics import precision, recall


from keras.metrics import categorical_crossentropy
import PIL
print(PIL.__version__)

print(keras.__version__)
import tensorflow
import numpy as np
print(np.__version__)
print(tensorflow.__version__)

args = {'batch_size': 16, 'data_path': os.path.join(PROJECT_PATH, 'data', 'TEMP_CODE'), 'previous_model': ''}

""" Define barcode class and underlying classes number from file structure """
NUM_CLASSES = count_folders(os.path.join(args['data_path'], 'train'))
BARCODE = get_barcode_class(args['data_path'])
SUPPORT_FILES_PATH = os.path.join(PROJECT_PATH, 'resource', BARCODE, 'support_files')

""" Check if previously trained model is used """
if args['previous_model'] == '':
    TRAIN_FROM_ZERO = True
else:
    TRAIN_FROM_ZERO = False

""" Modify existing architecture for actual number of classes """
if TRAIN_FROM_ZERO:
    #model = vgg16.VGG16(weights='imagenet')
    #model = inception_v3.InceptionV3(weights='imagenet')
    #model = resnet50.ResNet50(weights='imagenet')
    model_temp = mobilenet.MobileNet(weights='imagenet') # попробовать разные alpha
    #model = InceptionResNetV2(weights='imagenet')
    model_name = model_temp.name
    #model_name = model.name
    #model_temp.summary()



    model1 = Sequential()
    model1.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),
                     input_shape=(224, 224, 3), activation='relu'))
    model1.add(Conv2D(filters=32, kernel_size=(7, 7), strides=(2, 2), activation='relu'))
    model1.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model1.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2),  activation='relu'))
    model1.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2),  activation='relu'))
    model1.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2),  activation='relu'))
    model1.add(Flatten())
    #model1.add(Dense(units=2 * NUM_CLASSES, activation='relu'))
    #model1.add(Dense(units=8, activation='relu'))
    model1.add(Dense(units=NUM_CLASSES, activation='softmax'))
    model1.summary()
    ### end custom model


    '''### Start custom model:
    model1 = Sequential()
    model1.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2),
                     input_shape=(224, 224, 3)))
    model1.add(Conv2D(filters=32, kernel_size=(7, 7), strides=(2, 2)))
    model1.add(Conv2D(filters=32, kernel_size=(5, 5)))
    model1.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2)))
    model1.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2)))
    model1.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2)))
    model1.add(Flatten())
    #model1.add(Dense(units=2 * NUM_CLASSES, activation='relu'))
    #model1.add(Dense(units=8, activation='relu'))
    model1.add(Dense(units=NUM_CLASSES, activation='softmax'))
    model1.summary()
    ### end custom model'''

    '''### Start block for change last Dense layer:
    model.layers.pop()
    kernel_init = keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform', seed=None)
    o = Dense(units=NUM_CLASSES, name='predictions_final', trainable=True, activation='softmax', use_bias=True, kernel_initializer=kernel_init, bias_initializer='zeros')
    o = o(model.layers[-1].output)
    ### end block for change last Dense layer'''

    #o = Reshape(target_shape=(1, 1, model.layers[-1].output_shape[1]))(model.layers[-1].output)
    #o = Conv2D(filters=NUM_CLASSES, kernel_size=(1, 1))(o)
    #o = Activation('softmax')(o)
    #o = Reshape((NUM_CLASSES,))(o)

    ### start block for Mobile Net:
    model_temp.layers.pop()
    model_temp.layers.pop()
    model_temp.layers.pop()
    model_temp.get_layer(name='reshape_1').name = 'reshape_0'
    o = Conv2D(filters=NUM_CLASSES, kernel_size=(1, 1))(model_temp.layers[-1].output)
    o = Activation('softmax')(o)
    o = Reshape((NUM_CLASSES,))(o)
    ### end block for Mobile Net'''

    '''### start block for VGG Net:
    model.layers.pop()
    kernel_init = keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform', seed=None)
    o = Dense(units=NUM_CLASSES, name='predictions_final', trainable=True, activation='softmax', use_bias=True, kernel_initializer=kernel_init, bias_initializer='zeros')
    o = o(model.layers[-1].output)
    #o = Reshape((NUM_CLASSES,))(o)
    ### end block for VGG Net'''

    '''### start block for Inception Net:
    #model.layers[-1].output.set_shape = (None, 2)
    #model.layers.pop()
    kernel_init = keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform', seed=None)
    layer = Dense(units=NUM_CLASSES, name='predictions', trainable=True, activation='softmax', use_bias=True, kernel_initializer=kernel_init, bias_initializer='zeros')
    model.layers.pop()
    o = layer(model.layers[-1].output)
    #new_conf = model.layers[-1].get_config()
    #new_conf['units'] = 2
    #model.layers[-1] = model.layers[-1].from_config(new_conf)

    #o = Dense(units=NUM_CLASSES, input_shape=model.layers[-1].output_shape, activation='softmax')(model.layers[-1].output)
    #o = Activation('softmax')(o)
    ### end block for Inception Net'''

    '''### start block for ResNet:
    #a = model.layers[-1].get_config()
    model.layers.pop()
    kernel_init = keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform', seed=None)
    o = Dense(units=NUM_CLASSES, name='predictions_final', trainable=True, activation='softmax', use_bias=True, kernel_initializer=kernel_init, bias_initializer='zeros')
    o = o(model.layers[-1].output)
    #o = Conv2D(filters=NUM_CLASSES, kernel_size=(1, 1))(o)
    #o = Activation('softmax')(o)
    #o = Reshape((NUM_CLASSES,))(o)
    ### end block for ResNet '''

    # o = Flatten()(vgg_model.layers[-1].output)
    # o = Dense(units=NUM_CLASSES, input_shape=(NUM_CLASSES,))(o)
    # o = Dense(2 * NUM_CLASSES, input_shape=(None, mobilenet_model.output_shape))(mobilenet_model.layers[-1].output)
    # o = Dense(NUM_CLASSES, input_shape=(None, 2 * NUM_CLASSES))(mobilenet_model.layers[-1].output)

    #model = Model(model_temp.input, o)

    model = model1
    # model_name = model.name
    TRAINABLE_LAYERS = True
    for layer in model.layers: #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        layer.trainable = TRAINABLE_LAYERS

    model.summary()
    print('###########################')
    print('Usage of memory {} gygabytes'.format(get_model_memory_usage(64, model)))
    print('###########################')
else:
    with CustomObjectScope({'relu6': keras.layers.ReLU(6.), 'DepthwiseConv2D': DepthwiseConv2D}):
        TRAINABLE_LAYERS = True
        path = os.path.join(PROJECT_PATH, 'models', BARCODE, args['previous_model'] + '.h5')
        model = keras.models.load_model(path, custom_objects={'precision': precision, 'recall': recall})
        model_name = model.name

""" Define data path and output path  """
DATA_PATH = args['data_path']
OUTPUT_PATH = os.path.join(PROJECT_PATH, 'models', BARCODE, model_name + '.h5')
i = 1
while os.path.exists(OUTPUT_PATH):
    OUTPUT_PATH = os.path.join(PROJECT_PATH, 'models', BARCODE, model_name + '_' + str(i) + '.h5')
    i += 1

""" Data generators initialization: for train and validation sets """
train_datagen = image.ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=False,
    zca_whitening=False)
train_generator = train_datagen.flow_from_directory(
    directory=os.path.join(args['data_path'], 'train'), #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=args['batch_size'],
    class_mode="categorical",
    shuffle=True,
    seed=42
)

valid_datagen = image.ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=0,
    zca_whitening=False)

valid_generator = valid_datagen.flow_from_directory(
    directory=os.path.join(args['data_path'], 'val'),
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=args['batch_size'],
    class_mode="categorical",
    shuffle=True,
    seed=42
)

""" Set train parameters for choosen model """
#sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
optimizer = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
#optimize = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', precision, recall])

# // 10 => train_step ~ 13 min, val_step ~
STEP_SIZE_TRAIN = (train_generator.n // train_generator.batch_size) # поменять!
STEP_SIZE_VALID = (valid_generator.n // valid_generator.batch_size)
print(STEP_SIZE_TRAIN, STEP_SIZE_VALID)

""" Training """
checkpointer = ModelCheckpoint(OUTPUT_PATH, monitor='val_loss', verbose=0, save_best_only=False,
                               save_weights_only=False, mode='auto', period=1)

""" Read percantage parameters of train-val datasets """
with open(os.path.join(SUPPORT_FILES_PATH, 'percentage_train_val.txt'), 'r') as f:
    percentage_train_val = f.readline().split(' ')
percentage_train_val = [round(float(i), 2) for i in percentage_train_val]

""" Enable logging for Tensorboard """
tf_logger = TFLogger(PROJECT_PATH, model_name + '.h5', args['batch_size'], STEP_SIZE_TRAIN, STEP_SIZE_VALID, percentage_train_val[0], percentage_train_val[1], TRAINABLE_LAYERS, BARCODE, log_every=1, VERBOSE=0,  histogram_freq=0, write_graph=True,
                       write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                       embeddings_metadata=None, embeddings_data=None)
#tf_logger.start()

history = model.fit_generator(generator=train_generator,
                                  #steps_per_epoch=STEP_SIZE_TRAIN,
                                  validation_data=valid_generator,
                                  #validation_steps=STEP_SIZE_VALID,
                                  epochs=4,
                                  callbacks=[checkpointer, tf_logger],
                                  verbose=1
                                  )

""" Save logged entries """
#tf_logger.save_local()
history.model.save(OUTPUT_PATH)
