import numpy as np
import os
import csv

def compute_multiclass_multilabel_metrics(y_target: np.ndarray,
                                          y_pred: np.ndarray,
                                          threshold: float,
                                          file_name: str) -> None:
    """
    Inputs:
        y_target: real labels
        y_pred: predicted labels (probabilities)
        threshold: threshold for classification
    Outputs:
        .csv with metrics
    """
    # TP, TN, FN, FP per class -> 6 classes
    TP_0, TN_0, FN_0, FP_0 = 0, 0, 0, 0
    TP_1, TN_1, FN_1, FP_1 = 0, 0, 0, 0
    TP_2, TN_2, FN_2, FP_2 = 0, 0, 0, 0
    TP_3, TN_3, FN_3, FP_3 = 0, 0, 0, 0
    TP_4, TN_4, FN_4, FP_4 = 0, 0, 0, 0
    TP_5, TN_5, FN_5, FP_5 = 0, 0, 0, 0

    # Compare per instance target and prediction
    for target, pred in zip(y_target, y_pred):
        # Per instance, compute per class
        # Class 0: NF
        if target[0] == 1 and pred[0] >= threshold:
            TP_0 = TP_0 + 1
        if target[0] == 1 and pred[0] < threshold:
            FN_0 = FN_0 + 1
        if target[0] == 0 and pred[0] >= threshold:
            FP_0 = FP_0 + 1
        if target[0] == 0 and pred[0] < threshold:
            TN_0 = TN_0 + 1
        # Class 1: RM
        if target[1] == 1 and pred[1] >= threshold:
            TP_1 = TP_1 + 1
        if target[1] == 1 and pred[1] < threshold:
            FN_1 = FN_1 + 1
        if target[1] == 0 and pred[1] >= threshold:
            FP_1 = FP_1 + 1
        if target[1] == 0 and pred[1] < threshold:
            TN_1 = TN_1 + 1
        # Class 2: DPF
        if target[2] == 1 and pred[2] >= threshold:
            TP_2 = TP_2 + 1
        if target[2] == 1 and pred[2] < threshold:
            FN_2 = FN_2 + 1
        if target[2] == 0 and pred[2] >= threshold:
            FP_2 = FP_2 + 1
        if target[2] == 0 and pred[2] < threshold:
            TN_2 = TN_2 + 1
        # Class 3: NPF
        if target[3] == 1 and pred[3] >= threshold:
            TP_3 = TP_3 + 1
        if target[3] == 1 and pred[3] < threshold:
            FN_3 = FN_3 + 1
        if target[3] == 0 and pred[3] >= threshold:
            FP_3 = FP_3 + 1
        if target[3] == 0 and pred[3] < threshold:
            TN_3 = TN_3 + 1
        # Class 4: DWF
        if target[4] == 1 and pred[4] >= threshold:
            TP_4 = TP_4 + 1
        if target[4] == 1 and pred[4] < threshold:
            FN_4 = FN_4 + 1
        if target[4] == 0 and pred[4] >= threshold:
            FP_4 = FP_4 + 1
        if target[4] == 0 and pred[4] < threshold:
            TN_4 = TN_4 + 1
        # Class 5: NWF
        if target[5] == 1 and pred[5] >= threshold:
            TP_5 = TP_5 + 1
        if target[5] == 1 and pred[5] < threshold:
            FN_5 = FN_5 + 1
        if target[5] == 0 and pred[5] >= threshold:
            FP_5 = FP_5 + 1
        if target[5] == 0 and pred[5] < threshold:
            TN_5 = TN_5 + 1

    # Compute Metrics Per Class
    accuracy_0 = (TP_0+TN_0)/(TP_0+TN_0+FP_0+FN_0)
    precision_0 = (TP_0)/(TP_0+FP_0+1e-16)
    recall_0 = (TP_0)/(TP_0+FN_0+1e-16)
    f1_0 = 2*precision_0*recall_0/(precision_0+recall_0)

    accuracy_1 = (TP_1+TN_1)/(TP_1+TN_1+FP_1+FN_1)
    precision_1 = (TP_1)/(TP_1+FP_1+1e-16)
    recall_1 = (TP_1)/(TP_1+FN_1+1e-16)
    f1_1 = 2*precision_1*recall_1/(precision_1+recall_1)

    accuracy_2 = (TP_2 + TN_2) / (TP_2 + TN_2 + FP_2 + FN_2)
    precision_2 = (TP_2) / (TP_2 + FP_2 + 1e-16)
    recall_2 = (TP_2) / (TP_2 + FN_2 + 1e-16)
    f1_2 = 2 * precision_2 * recall_2 / (precision_2 + recall_2)

    accuracy_3 = (TP_3 + TN_3) / (TP_3 + TN_3 + FP_3 + FN_3)
    precision_3 = (TP_3) / (TP_3 + FP_3 + 1e-16)
    recall_3 = (TP_3) / (TP_3 + FN_3 + 1e-16)
    f1_3 = 2 * precision_3 * recall_3 / (precision_3 + recall_3)

    accuracy_4 = (TP_4 + TN_4) / (TP_4 + TN_4 + FP_4 + FN_4)
    precision_4 = (TP_4) / (TP_4 + FP_4 + 1e-16)
    recall_4 = (TP_4) / (TP_4 + FN_4 + 1e-16)
    f1_4 = 2 * precision_4 * recall_4 / (precision_4 + recall_4)

    accuracy_5 = (TP_5 + TN_5) / (TP_5 + TN_5 + FP_5 + FN_5)
    precision_5 = (TP_5) / (TP_5 + FP_5 + 1e-16)
    recall_5 = (TP_5) / (TP_5 + FN_5 + 1e-16)
    f1_5 = 2 * precision_5 * recall_5 / (precision_5 + recall_5)

    f1_avg = (f1_0+f1_1+f1_2+f1_3+f1_4+f1_5)/6

    variable_names = [
        'threshold',
        'TP_0', 'TN_0', 'FP_0', 'FN_0',
        'TP_1', 'TN_1', 'FP_1', 'FN_1',
        'TP_2', 'TN_2', 'FP_2', 'FN_2',
        'TP_3', 'TN_3', 'FP_3', 'FN_3',
        'TP_4', 'TN_4', 'FP_4', 'FN_4',
        'TP_5', 'TN_5', 'FP_5', 'FN_5',
        'accuracy_0', 'precision_0', 'recall_0', 'f1_0',
        'accuracy_1', 'precision_1', 'recall_1', 'f1_1',
        'accuracy_2', 'precision_2', 'recall_2', 'f1_2',
        'accuracy_3', 'precision_3', 'recall_3', 'f1_3',
        'accuracy_4', 'precision_4', 'recall_4', 'f1_4',
        'accuracy_5', 'precision_5', 'recall_5', 'f1_5',
        'f1_avg'
    ]

    variable_values = [
        threshold,
        TP_0, TN_0, FP_0, FN_0,
        TP_1, TN_1, FP_1, FN_1,
        TP_2, TN_2, FP_2, FN_2,
        TP_3, TN_3, FP_3, FN_3,
        TP_4, TN_4, FP_4, FN_4,
        TP_5, TN_5, FP_5, FN_5,
        accuracy_0, precision_0, recall_0, f1_0,
        accuracy_1, precision_1, recall_1, f1_1,
        accuracy_2, precision_2, recall_2, f1_2,
        accuracy_3, precision_3, recall_3, f1_3,
        accuracy_4, precision_4, recall_4, f1_4,
        accuracy_5, precision_5, recall_5, f1_5,
        f1_avg
    ]

    if not os.path.isfile(file_name):
        with open(file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(variable_names)
            writer.writerow(variable_values)
    else:
        with open(file_name, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(variable_values)