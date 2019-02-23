import numpy as np
import tqdm
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def count_TP_and_FP_for_df(df):
    """
    For given dataframe calculates for all unique classses metrics TP, FP, FN, TN
    :param df: pandas Dataframe, should contain columns 'true_type' (true value), 'Predictions' (predicted value),
    'indicator' (if pred == true)
    :return: 4 lists of TP, FP, TN and FN
    """
    # Select labels for confusion_matrix function:
    labels = list(df['true_type'].unique())
    labels.extend([x for x in df['Predictions'].unique() if x not in labels])

    # run confusion_matrix function and count TP, FP, FN, TN
    # Code taken from here (also visualizations are provided): https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    A = confusion_matrix(df['true_type'], df['Predictions'], labels=labels)
    TP_A = np.diag(A)
    FP_A = A.sum(axis=0) - np.diag(A)
    FN_A = A.sum(axis=1) - np.diag(A)
    TN_A = np.diag(A).sum() - TP_A # TN code in Stackoverflow is incorrect, this variant is correct one

    cut = len(df['true_type'].unique())
    TP_A = TP_A[:cut]
    FP_A = FP_A[:cut]
    FN_A = FN_A[:cut]
    TN_A = TN_A[:cut]

    return TP_A, FP_A, FN_A, TN_A

def select_best_threshold(threshold_list, filenames, PATH_PREDICTIONS, PATH_TEST_ANSWERS, PATH_LABELS, PATH_CLASSES_IN_SET):
    labels = pd.read_csv(PATH_LABELS)
    labels = dict(zip(labels['class_index'], labels['class_name']))
    labels[-1] = 'rejected'

    pred = pd.read_csv(PATH_PREDICTIONS, header=None)
    if isinstance(pred, pd.DataFrame):
        pred = pred.values

    best_pr = np.nan
    true_answers = pd.read_excel(PATH_TEST_ANSWERS).set_index('card_id')
    classes_in_train_set = pd.read_excel(PATH_CLASSES_IN_SET)['class_name_in_training_set'].values
    true_answers.loc[~true_answers['true_type'].isin(classes_in_train_set), 'true_type'] = 'rejected'

    for thresh in threshold_list:
        bool_pred = pred > thresh
        predicted_class_indices = np.argmax(pred, axis=1)
        rows_does_not_contain_more_than_thresh = np.argwhere(bool_pred.any(1) == False)
        predicted_class_indices[rows_does_not_contain_more_than_thresh] = -1

        # labels = (train_generator.class_indices)
        # labels = dict((v,k) for k,v in labels.items())
        predictions = [labels[k] for k in predicted_class_indices]

        results = pd.DataFrame({"Filename": filenames,
                                "Predictions": predictions}).set_index('Filename')
        df = pd.concat([results, true_answers], axis=1, join_axes=[results.index])
        df['indicator'] = df['Predictions'] == df['true_type']

        TP_list, FP_list, _, _ = count_TP_and_FP_for_df(df)
        mean_pr = np.mean(TP_list) / (np.mean(TP_list) + np.mean(FP_list))

        print('For thresh {0}, mean precision: {1}'.format(thresh, mean_pr))
        if mean_pr > best_pr or best_pr is np.nan:
            best_pr = mean_pr
            best_thresh = thresh
            #best_df = df.copy()

    with open(os.path.join(os.path.dirname(PATH_PREDICTIONS), 'best_threshold.txt'), 'w') as f:
        f.write(str(best_thresh))

    return best_pr, best_thresh

if __name__ == '__main__':
    d = {'Predictions': ['a', 'b', 'd', 'g'], 'true_type': ['a', 'c', 'd', 'a']}
    df = pd.DataFrame(d)
    df['indicator'] = df['Predictions'] == df['true_type']
    count_TP_and_FP_for_df(df)