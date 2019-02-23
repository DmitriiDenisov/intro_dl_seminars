import glob
import shutil
import random
import os
import numpy as np

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = r'/home/cardsmobile_data/'
BARCODE_TYPE = 'TEMP_CODE'
SUPPORT_FILES_PATH = os.path.join(PROJECT_PATH, 'resource', BARCODE_TYPE, 'support_files')
classes_folders = glob.glob(os.path.join(DATA_PATH, BARCODE_TYPE, 'train_source', '*'))

if not os.path.exists(os.path.join(DATA_PATH, BARCODE_TYPE, 'train')):
    os.mkdir(os.path.join(DATA_PATH, BARCODE_TYPE, 'train'))
if not os.path.exists(os.path.join(DATA_PATH, BARCODE_TYPE, 'val')):
    os.mkdir(os.path.join(DATA_PATH, BARCODE_TYPE, 'val'))

perc_taken_train = 2/3
perc_taken_val = 1
with open(os.path.join(SUPPORT_FILES_PATH, 'percentage_train_val.txt'), "w") as output:
    output.write(' '.join([str(perc_taken_train), str(perc_taken_val)]))

for idx, folder in enumerate(classes_folders):
    class_name = os.path.basename(folder)
    print(idx, class_name)

    train_target_folder = os.path.join(DATA_PATH, BARCODE_TYPE, 'train', class_name)
    test_target_folder = os.path.join(DATA_PATH, BARCODE_TYPE, 'val', class_name)
    os.mkdir(train_target_folder)
    os.mkdir(test_target_folder)

    file_list = np.array(glob.glob(os.path.join(folder, '*')))
    num_of_taken_in_train = int(len(file_list) * perc_taken_train)
    num_of_taken_in_val = int(len(file_list) * (1 - perc_taken_train) * perc_taken_val)

    train_indexes = set(random.sample(range(len(file_list)), num_of_taken_in_train))
    all_rest = set(range(len(file_list))) - train_indexes
    test_indexes = set(random.sample(all_rest, num_of_taken_in_val))

    if len(set(train_indexes).intersection(test_indexes)) > 0:
        raise ValueError('Intersection of Test and Train is not empty!')

    for filename in file_list[list(train_indexes)]:
        shutil.copy(filename, train_target_folder)
    for filename in file_list[list(test_indexes)]:
        shutil.copy(filename, test_target_folder)