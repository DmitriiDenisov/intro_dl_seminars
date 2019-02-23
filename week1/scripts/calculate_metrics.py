import os
import pandas as pd
from pandas import ExcelWriter
from engine.tools.make_consolidation_df import consolidation_df_for_predictions_with_all_metrics


args = {'barcode': 'TEMP_CODE', 'model': 'mobilenet_1.00_224_1.h5'}

""" Set paths for project, model to be used, input data, train data and output data"""
#!!!barcode = '_' + args.barcode
barcode = args['barcode']

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(PROJECT_PATH, 'resource', barcode, 'results')
SUPPORT_FILES_PATH = os.path.join(PROJECT_PATH, 'resource', barcode, 'support_files')

# Create classes_in_set.xlsx file:
DATA_PATH = os.path.join(PROJECT_PATH, 'data', barcode, 'train')
list_of_dirs = [name for name in os.listdir(DATA_PATH) if not os.path.isfile(name)]
df = pd.DataFrame({'class_name_in_training_set': list_of_dirs})
result_path = os.path.join(PROJECT_PATH, 'resource', barcode, 'support_files')
writer = ExcelWriter(os.path.join(result_path, 'classes_in_set.xlsx'))
df.to_excel(writer, 'Sheet1', index=False)
writer.save()

""" Calculate metrics """
TRAIN_SOURCE = False
consolidation_df_for_predictions_with_all_metrics(args['model'], OUTPUT_PATH, SUPPORT_FILES_PATH, TRAIN_SOURCE)

