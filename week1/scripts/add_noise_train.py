import pandas as pd
import tqdm
import numpy as np
import glob
import shutil
import random
import os

path_to_consolidation_error = '/home/ddenisov/cardsmobile_recognition/resource/_VOICE_NEW/results/consolidation_error_df__VOICE_NEW_mobilenet_1.00_224_1.xlsx'
path_to_train = '/home/cardsmobile_data/_VOICE_NEW/train'
path_to_train_source = '/home/cardsmobile_data/_VOICE_NEW/train_source'
error_df = pd.read_excel(path_to_consolidation_error)
unique_true_type = error_df['true_type'].unique()
for unique in tqdm.tqdm(unique_true_type):
    # select dataframe:
    selected_error_df = error_df[(error_df['true_type'] == unique) & (error_df['Predictions'] != 'rejected')]
    if len(selected_error_df) == 0 or unique == 'rejected':
        continue

    # Define path from where to where copy
    class_to_add = selected_error_df.iloc[0]['Predictions']
    path_where_to_copy = os.path.join(path_to_train, unique)

    # Random selection:
    file_list = np.array(glob.glob(os.path.join(path_to_train, class_to_add, '*')))
    #file_list = np.array(glob.glob(os.path.join(folder, '*')))
    num_of_taken_as_noise = int(len(file_list) * 0.1) + 1 # Процент!
    train_indexes = set(random.sample(range(len(file_list)), num_of_taken_as_noise))
    train_images_noise = file_list[list(train_indexes)]

    # Copy
    for train_image_noise in train_images_noise:
        #path_what_to_copy = os.path.join(path_to_train, class_to_add, file)
        shutil.copy(train_image_noise, path_where_to_copy)

