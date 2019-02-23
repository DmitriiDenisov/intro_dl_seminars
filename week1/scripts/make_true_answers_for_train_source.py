import os
import pandas as pd
from os import listdir
from pandas import ExcelWriter
from os.path import isfile, join

# Структура выходного файла: две колонки, card_id, true_type;

BARCODE = 'TEMP_CODE'
PATH_TO_TRAIN_SOURCE = os.path.join('/home/cardsmobile_data', BARCODE, 'train_source')

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_PATH = os.path.join(PROJECT_PATH, 'resource', BARCODE, 'support_files')
list_of_directories = os.listdir(PATH_TO_TRAIN_SOURCE)

true_types_train_source = pd.DataFrame(columns=['card_id', 'true_type'])
print('Num of Directories:', len(list_of_directories))
for i, directory in enumerate(list_of_directories):
    if i % 100 == 0:
        print('Passed {} iterations!'.format(i))
    directory_path = os.path.join(PATH_TO_TRAIN_SOURCE, directory)
    onlyfiles = [f[:-4] for f in listdir(directory_path) if isfile(join(directory_path, f))]
    temp_df = pd.DataFrame({'true_type': directory, 'card_id': onlyfiles}, columns=['true_type', 'card_id'])
    frames = [true_types_train_source, temp_df]
    true_types_train_source = pd.concat(frames)


writer = ExcelWriter(os.path.join(RESULT_PATH, 'true_types_train_source.xlsx'))
true_types_train_source.to_excel(writer, 'Sheet1', index=False)
writer.save()