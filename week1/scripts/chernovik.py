# Статистика Mobile Net на _EAN_13:
# 1) 0.66 train, rest test => 0.922 accuracy
# 2) 0.5 train, 0.6*rest test => 0.9153 accuracy
# 3) 0.15 train, 0.4*rest test => 0.9056 accuracy
# 4) 0.05 train, 0.35*rest test => 0.88 accuracy
#------------------
# Статистика разных предобученных моделей на _VOICE:
# Если веса зафиксированы везде, кроме последнего слоя:
# VGG - не учится
# Xception, InceptionResNetV2 - учатся на уровне Inception
# MobileNetV2 - лучше Inception, но хуже ResNet и MobileNet
# DenseNet121 - на трейне лучше кач-во, чем у MobileNet, но на val качество плохое, переобучается
# NASNetMobile - учится на уровне Inception

# Если все слои не фиксированы:
# VGG19Б 16 - не учится
# Xception - не запускается. Нехватка памяти? Chunks
# InceptionResNetV2 - очень долгая, но в итоге лучше, чем mobile net
# MobileNetV2 - Переобучается сильно
# DenseNet121 - не запускаяется. Нехватка памяти? Chunks
# NASNetMobile - на уровне Inception

import numpy as np
import pandas as pd
import os

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


results = pd.read_excel(os.path.join(PROJECT_PATH, 'results_EAN_13_close.xlsx')).set_index('Filename')
true_answers = pd.read_csv(os.path.join(PROJECT_PATH, 'close_data_true_labels.csv')).set_index('filename')
df = pd.concat([results, true_answers], axis=1, join_axes=[results.index])
df['Indicator'] = df['Pred_class_with_thresh_0.45'] == df['true_type']

df_temp = df[df['Pred_class_with_thresh_0.45'] != 'rejected']
bad_proportion = len(df_temp[df_temp['Indicator'] == False]) / len(df_temp)
print(bad_proportion)
pass