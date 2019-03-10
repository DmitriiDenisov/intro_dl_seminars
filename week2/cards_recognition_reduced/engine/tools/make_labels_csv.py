import os

import pandas as pd
from keras.preprocessing import image

from engine.tools.filesystem_functions import get_barcode_class

def generate_labels_from_train(data_path, project_path):
    """
    Returns labels names to use in prediction
    :param path: path to train data
    :return:
    pandas DataFrame with labels names
    """
    train_datagen = image.ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
        directory=os.path.join(data_path, 'train'),
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )

    barcode = get_barcode_class(data_path)

    labels = (train_generator.class_indices)
    labels = dict((v, k) for k, v in labels.items())

    df = pd.DataFrame(list(labels.items()), columns=['class_index', 'class_name'])
    df.to_csv(os.path.join(project_path, 'resource', barcode, 'support_files', 'labels.csv'), index=False)

    return df
