import os
import numpy as np
import sys
import keras
from keras.preprocessing import image
from keras import optimizers
from keras.layers import DepthwiseConv2D
from keras_applications.mobilenet import relu6
from keras.utils.generic_utils import CustomObjectScope
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_PATH)
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from engine.tools.make_labels_csv import generate_labels_from_train
from engine.tools.filesystem_functions import get_cardname, get_barcode_class
from engine.tools.make_consolidation_df import consolidation_df_for_predictions
from engine.tools.add_metrics import precision, recall

LOAD_MODEL_FROM_LOGS = True
name_of_models = ['mobilenet_1.00_224_1.h5']
args = {'input_path': os.path.join(PROJECT_PATH, 'data', 'TEMP_CODE', 'test'),
        'train_path': os.path.join(PROJECT_PATH, 'data', 'TEMP_CODE')}

""" Set paths for project, model to be used, input data, train data and output data"""
TRAIN_DATA_PATH = args['train_path']
barcode = get_barcode_class(TRAIN_DATA_PATH)


for name_of_model in name_of_models:
    if LOAD_MODEL_FROM_LOGS:
        args['model'] = name_of_model
        MODEL_PATH = os.path.join(PROJECT_PATH, 'models', barcode, name_of_model) #!!!!!!!!!!!!!!!!
    else:
        MODEL_PATH = os.path.join(PROJECT_PATH, 'models', barcode, args['model'])
    INPUT_PATH = args['input_path']
    OUTPUT_PATH = os.path.join(PROJECT_PATH, 'resource', barcode, 'results')
    SUPPORT_FILES_PATH = os.path.join(PROJECT_PATH, 'resource', barcode, 'support_files')

    """ Get labels names from the folder with train samples """
    if not os.path.exists(os.path.join(SUPPORT_FILES_PATH, 'labels.csv')):
        generate_labels_from_train(TRAIN_DATA_PATH, PROJECT_PATH)

    """ Load and define model """
    with CustomObjectScope({'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D}):
        model = keras.models.load_model(MODEL_PATH, custom_objects={'precision': precision, 'recall': recall})
    sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    """ Initialize data generator for input data """
    test_datagen = image.ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    test_generator = test_datagen.flow_from_directory(
        directory=INPUT_PATH,
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=1,
        class_mode=None,
        shuffle=False,
        seed=42
    )

    """ Save input cardnames """
    test_generator.reset()
    filenames = test_generator.filenames
    cardnames = [get_cardname(filename) for filename in filenames]
    with open(os.path.join(SUPPORT_FILES_PATH, 'cardnames.txt'), "w") as output:
        output.write(' '.join(cardnames))

    TRAIN_SOURCE = 'train_source' in args['input_path']
    if TRAIN_SOURCE:
        flag = os.path.exists(os.path.join(OUTPUT_PATH, 'predictions_with_all_probabilities_train_source_{}.csv'.format(args['model'][:-3])))
    else:
        flag = os.path.exists(os.path.join(OUTPUT_PATH, 'predictions_with_all_probabilities_{}.csv'.format(args['model'][:-3])))
    """ Make predictions with the selected model """
    if not flag:

        pred = model.predict_generator(test_generator, verbose=1)

        """ Save prediction results """
        if TRAIN_SOURCE:
            np.savetxt(os.path.join(OUTPUT_PATH, 'predictions_with_all_probabilities_train_source_{}.csv'.format(args['model'][:-3])),pred,
                delimiter=",")
        else:
            np.savetxt(os.path.join(OUTPUT_PATH, 'predictions_with_all_probabilities_{}.csv'.format(args['model'][:-3])), pred,
                       delimiter=",")

    """ Generate files with processed results """
    consolidation_df_for_predictions(args['model'], OUTPUT_PATH, SUPPORT_FILES_PATH, TRAIN_SOURCE)
