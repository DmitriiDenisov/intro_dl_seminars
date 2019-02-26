import os
import numpy as np
import pandas as pd
from pandas import ExcelWriter
from engine.tools.select_best_threshold import count_TP_and_FP_for_df


def consolidation_df_for_predictions(model, result_path, support_files_path, TRAIN_SOURCE):
    # Read all needed files: cardnames, best_threshold, labels, predictions_all_probabilities
    with open(os.path.join(support_files_path, 'cardnames.txt')) as f:
        content = f.readlines()
    filenames = content[0].split(' ')
    with open(os.path.join(result_path, 'best_threshold.txt'), 'r') as f:
        best_thresh = float(f.readline())
    labels = pd.read_csv(os.path.join(support_files_path, 'labels.csv'))
    labels = dict(zip(labels['class_index'], labels['class_name']))
    labels[-1] = 'rejected'
    if TRAIN_SOURCE:
        print('Reading')
        pred = pd.read_csv(os.path.join(result_path, 'predictions_with_all_probabilities_train_source_{}.csv'.format(model[:-3])), header=None)
        print('Read!')
    else:
        pred = pd.read_csv(os.path.join(result_path, 'predictions_with_all_probabilities_{}.csv'.format(model[:-3])), header=None)

    if isinstance(pred, pd.DataFrame):
        pred = pred.values

    # Make predictions with respect to threshold
    predicted_class_indices = np.argmax(pred, axis=1)
    predictions_without_thresh = [labels[k] for k in predicted_class_indices]
    df = pd.DataFrame({"Filename": filenames,
                            "Predictions": predictions_without_thresh}).set_index('Filename')
    #df = pd.concat([results, true_answers], axis=1, join_axes=[results.index])
    df['Pred_proba_without_thresh'] = np.max(pred, axis=1)
    #df['indicator'] = df['Predictions'] == df['true_type']
    bool_pred = pred > best_thresh
    rows_does_not_contain_more_than_thresh = np.argwhere(bool_pred.any(1) == False)
    predicted_class_indices[rows_does_not_contain_more_than_thresh] = -1
    predictions_with_thresh = [labels[k] for k in predicted_class_indices]
    df['Pred_class_with_thresh_' + str(best_thresh)] = predictions_with_thresh

    # write into file
    if TRAIN_SOURCE:
        print('Writing')
        writer = ExcelWriter(os.path.join(result_path, 'consolidation_df_train_source_{}.xlsx'.format(model[:-3])))
        print('Written!')
    else:
        writer = ExcelWriter(os.path.join(result_path, 'consolidation_df_{}.xlsx'.format(model[:-3])))
    df.to_excel(writer, 'Sheet1')
    writer.save()

def consolidation_df_for_predictions_with_all_metrics(model, result_path, support_files_path, TRAIN_SOURCE):
    # Read cardnames, true_answers, classes_in_train_set, labels and predictions_with_all_probabilities:
    if TRAIN_SOURCE:
        true_answers = pd.read_excel(os.path.join(support_files_path, 'true_types_train_source.xlsx')).set_index('card_id')
    else:
        true_answers = pd.read_excel(os.path.join(support_files_path, 'true_answers.xlsx')).set_index('card_id')
    classes_in_train_set = pd.read_excel(os.path.join(support_files_path, 'classes_in_set.xlsx'))['class_name_in_training_set'].values
    true_answers.loc[~true_answers['true_type'].isin(classes_in_train_set), 'true_type'] = 'rejected'
    labels = pd.read_csv(os.path.join(support_files_path, 'labels.csv'))
    labels = dict(zip(labels['class_index'], labels['class_name']))
    labels[-1] = 'rejected'

    if TRAIN_SOURCE:
        print('Reading')
        df = pd.read_excel(os.path.join(result_path, 'consolidation_df_train_source_{}.xlsx'.format(model[:-3])), header=0).set_index('Filename')
        print('Read!')
    else:
        df = pd.read_excel(os.path.join(result_path, 'consolidation_df_{}.xlsx'.format(model[:-3])), header=0).set_index('Filename')
    df = df[[df.columns[-1]]]
    df.rename(columns={df.columns[-1]: 'Predictions'}, inplace=True)
    df = pd.concat([df, true_answers], axis=1, join_axes=[df.index])
    df['indicator'] = df['Predictions'] == df['true_type']

    ##
    writer1 = ExcelWriter(os.path.join(result_path, 'df_pred_on_test_{}.xlsx'.format(model[:-3])))
    df.to_excel(writer1, 'Sheet1')
    writer1.save()
    ##


    # Creating score df and calculate all metrics:
    df_results = pd.DataFrame({'Unique true_type': df['true_type'].unique()})
    num_els_in_test = df['true_type'].value_counts()
    df_results['num_els_in_test'] = num_els_in_test.loc[df_results['Unique true_type']].values
    if TRAIN_SOURCE:
        print('Calculating TP, FP, FN, TN')
    TP_list, FP_list, FN_list, TN_list = count_TP_and_FP_for_df(df)
    if TRAIN_SOURCE:
        print('Calculated!')
    df_results['TP_list'] = TP_list
    df_results['FP_list'] = FP_list
    df_results['FN_list'] = FN_list
    df_results['TN_list'] = TN_list
    df_results['Precision'] = df_results['TP_list'] / (df_results['TP_list'] + df_results['FP_list'])
    df_results['Precision'].fillna(0, inplace=True)
    df_results['Recall'] = df_results['TP_list'] / (df_results['TP_list'] + df_results['FN_list'])
    df_results['Weighted_Pr'] = df_results['Precision'] * df_results['num_els_in_test'] / df_results['num_els_in_test'].sum()
    df_results['Weighted_Re'] = df_results['Recall'] * df_results['num_els_in_test'] / df_results['num_els_in_test'].sum()
    df_results['bad_classification_proportion'] = ''

    df_temp = df[df['Predictions'] != 'rejected']
    bad_proportion = len(df_temp[df_temp['indicator'] == False]) / len(df)

    scores = ['Scores:', '', df_results['TP_list'].sum(), df_results['FP_list'].sum(), df_results['FN_list'].sum(),
              df_results['TN_list'].sum(), df_results['TP_list'].mean() / (df_results['TP_list'].mean() + df_results['FP_list'].mean()),
              df_results['TP_list'].mean() / (df_results['TP_list'].mean() + df_results['FN_list'].mean()),
              df_results['Weighted_Pr'].sum(), df_results['Weighted_Re'].sum(), bad_proportion]
    df_results.loc[len(df_results)] = [''] * df_results.shape[1]
    df_results.loc[len(df_results)] = scores

    # Save file with metrics:
    if TRAIN_SOURCE:
        print('Writing')
        writer = ExcelWriter(os.path.join(result_path, 'consolidation_df_all_metrics_train_source_{}.xlsx'.format(model[:-3])))
        print('Written!')
    else:
        writer = ExcelWriter(os.path.join(result_path, 'consolidation_df_all_metrics_{}.xlsx'.format(model[:-3])))
    df_results.to_excel(writer, 'Sheet1', index=False)
    writer.save()

def make_consolidation_for_errors(model, result_path, support_files_path, TRAIN_SOURCE):
    # Read all needed files:
    if TRAIN_SOURCE:
        print('Reading')
        true_answers = pd.read_excel(os.path.join(support_files_path, 'true_types_train_source.xlsx')).set_index('card_id')
        PATH_CONSOLIDATION_DF = os.path.join(PROJECT_PATH, 'resource', barcode, 'results', 'consolidation_df_train_source_{}.xlsx'.format(model[:-3]))
        print('Read!')
    else:
        true_answers = pd.read_excel(os.path.join(support_files_path, 'true_answers.xlsx')).set_index('card_id')
        PATH_CONSOLIDATION_DF = os.path.join(PROJECT_PATH, 'resource', barcode, 'results', 'consolidation_df_{}.xlsx'.format(model[:-3]))
    classes_in_train_set = pd.read_excel(os.path.join(support_files_path, 'classes_in_set.xlsx'))['class_name_in_training_set'].values
    true_answers.loc[~true_answers['true_type'].isin(classes_in_train_set), 'true_type'] = 'rejected'

    results = pd.read_excel(PATH_CONSOLIDATION_DF).set_index('Filename')
    results = results[[results.columns[-1]]]
    results.rename(columns={results.columns[-1]: 'Predictions'}, inplace=True)

    # Concatenate predictions and true
    df = pd.concat([results, true_answers], axis=1, join_axes=[results.index])
    df['indicator'] = df['Predictions'] == df['true_type']

    # Group by and aggregate true_type & indicator
    count_in_test_of_types = df[['true_type', 'indicator']].groupby('true_type').agg(['count'])['indicator']['count']
    count_in_test_of_types = count_in_test_of_types.to_dict()
    consolidation_error_df = df[df['indicator'] == False].groupby(['true_type', 'Predictions']).agg('count').reset_index()
    consolidation_error_df.rename(columns={'indicator': 'count'}, inplace=True)
    consolidation_error_df['percentage'] = consolidation_error_df['count'] / consolidation_error_df['count'].sum()
    consolidation_error_df['num_appearance_in_test'] = consolidation_error_df['true_type'].map(count_in_test_of_types)

    # Write file
    if TRAIN_SOURCE:
        writer = ExcelWriter(os.path.join(result_path, 'consolidation_error_df_train_source_{}_{}.xlsx'.format(barcode, model[:-3])))
    else:
        writer = ExcelWriter(os.path.join(result_path, 'consolidation_error_df_{}_{}.xlsx'.format(barcode, model[:-3])))
    consolidation_error_df.to_excel(writer, 'Sheet1', index=False)
    writer.save()

if __name__ == '__main__':
    barcode = 'TEMP_CODE'
    TRAIN_SOURCE = False
    PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    OUTPUT_PATH = os.path.join(PROJECT_PATH, 'resource', barcode, 'results')
    SUPPORT_FILES_PATH = os.path.join(PROJECT_PATH, 'resource', barcode, 'support_files')

    """ Calculate metrocs """
    make_consolidation_for_errors('mobilenet_1.00_224.h5', OUTPUT_PATH, SUPPORT_FILES_PATH, TRAIN_SOURCE)