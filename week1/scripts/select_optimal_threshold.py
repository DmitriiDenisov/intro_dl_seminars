import os
import argparse
import numpy as np
import pandas as pd

from engine.tools.select_best_threshold import select_best_threshold

# TODO: to select threshold based on validation data, not test

""" Initialize argument parser """
parser = argparse.ArgumentParser(description='Script for selecting the best threshold')
parser.add_argument('-m', '--model', action='store', type=str, default='',
                    help='Name of the model used for obtaining processing results')
parser.add_argument('-b', '--barcode', action='store', type=str, default='',
                    help='Define barcode class which results are analyzed')
args = parser.parse_args()

""" Define paths """
barcode = '_' + args.barcode
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPPORT_FILES_PATH = os.path.join(PROJECT_PATH, 'resource', barcode, 'support_files')
RESULT_PATH = os.path.join(PROJECT_PATH, 'resource', barcode, 'results')

PATH_LABELS = os.path.join(SUPPORT_FILES_PATH, 'labels.csv')
PATH_TEST_ANSWERS = os.path.join(SUPPORT_FILES_PATH, 'true_answers.xlsx')
PATH_PREDICTIONS = os.path.join(RESULT_PATH, 'predictions_with_all_probabilities_{}.csv'.format(args.model[:-3]))
PATH_FILES_NAMES = os.path.join(SUPPORT_FILES_PATH, 'cardnames.txt')
PATH_CLASSES_IN_SET = os.path.join(SUPPORT_FILES_PATH, 'classes_in_set.xlsx')

""" Read cardnames list and launch the selection """
with open(PATH_FILES_NAMES) as f:
    content = f.readlines()
cardnames = content[0].split(' ')

select_best_threshold(np.linspace(0, 1, 21), cardnames, PATH_PREDICTIONS, PATH_TEST_ANSWERS,
                      PATH_LABELS, PATH_CLASSES_IN_SET)