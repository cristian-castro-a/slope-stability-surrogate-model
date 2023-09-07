from pathlib import Path
from pickle import dump
from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model

from multilabel_multiclass_metrics import compute_multiclass_multilabel_metrics
from utils.loader import load_yaml
from utils.plotting import plot_multiple_learning_curves, plot_confusion_matrix, plot_multiple_confusion_matrices

CONFIG = 'config.yaml'
BINARY_LABELS = ['No_Failure', 'Failure']
MULTICLASS_LABELS = ['NF', 'RM', 'DPF', 'NPF', 'DWF', 'NWF']
NUMBER_OF_PREDICTORS = 26
EPOCHS = 100

THRESHOLD_1 = 0.05
THRESHOLD_2 = 0.10
THRESHOLD_3 = 0.5


@tf.function
def macro_soft_f1(y, y_hat):
    """Computes the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.

    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    # reduce 1 - soft-f1 in order to increase soft-f1
    cost = 1 - soft_f1
    # average on all labels
    macro_cost = tf.reduce_mean(cost)
    return macro_cost


@tf.function
def macro_f1(y, y_hat, thresh=0.5):
    """Computes the macro F1-score on a batch of observations (average F1 across labels)

    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive

    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1


def architecture_1() -> Model:
    # Builds best architecture for the binary model as multioutput model
    inputs = tf.keras.Input(shape=(NUMBER_OF_PREDICTORS,))

    x = Dense(units=500, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(inputs)
    x = Dropout(rate=0.2)(x)
    x = Dense(units=300, activation='tanh', kernel_initializer=tf.keras.initializers.HeNormal())(x)
    bifurcation = Dropout(rate=0.1)(x)

    bo = Dense(units=100, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(bifurcation)
    bo = Dropout(rate=0.05)(bo)
    bo = Dense(units=10, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(bo)
    binary_output = Dense(units=1, activation='sigmoid', name='binary_output')(bo)

    mo = Dense(units=200, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(bifurcation)
    mo = Dropout(rate=0.05)(mo)
    mo = Dense(units=100, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(mo)
    mo = Dense(units=50, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(mo)
    multiclass_output = Dense(units=6, activation='sigmoid', name='multiclass_output')(mo)

    return Model(inputs=inputs, outputs=[binary_output, multiclass_output])


def architecture_2() -> Model:
    # Builds best architecture for the binary model as multioutput model
    inputs = tf.keras.Input(shape=(NUMBER_OF_PREDICTORS,))

    x = Dense(units=500, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(inputs)
    x = Dropout(rate=0.2)(x)
    x = Dense(units=250, activation='tanh', kernel_initializer=tf.keras.initializers.HeNormal())(x)
    bifurcation = Dropout(rate=0.1)(x)

    bo = Dense(units=10, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(bifurcation)
    bo = Dropout(rate=0.05)(bo)
    bo = Dense(units=10, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(bo)
    binary_output = Dense(units=1, activation='sigmoid', name='binary_output')(bo)

    mo = Dense(units=100, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(bifurcation)
    mo = Dropout(rate=0.05)(mo)
    mo = Dense(units=50, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(mo)
    mo = Dense(units=10, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(mo)
    multiclass_output = Dense(units=6, activation='sigmoid', name='multiclass_output')(mo)

    return Model(inputs=inputs, outputs=[binary_output, multiclass_output])


def architecture_3() -> Model:
    # Builds best architecture for the binary model as multioutput model
    inputs = tf.keras.Input(shape=(NUMBER_OF_PREDICTORS,))

    x = Dense(units=450, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(inputs)
    x = Dropout(rate=0.2)(x)
    x = Dense(units=230, activation='tanh', kernel_initializer=tf.keras.initializers.HeNormal())(x)
    bifurcation = Dropout(rate=0.05)(x)

    bo = Dense(units=10, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(bifurcation)
    bo = Dropout(rate=0.05)(bo)
    bo = Dense(units=10, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(bo)
    bo = Dense(units=10, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(bo)
    binary_output = Dense(units=1, activation='sigmoid', name='binary_output')(bo)

    mo = Dense(units=100, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(bifurcation)
    mo = Dropout(rate=0.05)(mo)
    mo = Dense(units=50, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(mo)
    mo = Dense(units=10, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(mo)
    multiclass_output = Dense(units=6, activation='sigmoid', name='multiclass_output')(mo)

    return Model(inputs=inputs, outputs=[binary_output, multiclass_output])


def architecture_4() -> Model:
    # Builds best architecture for the binary model as multioutput model
    inputs = tf.keras.Input(shape=(NUMBER_OF_PREDICTORS,))

    x = Dense(units=450, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(inputs)
    x = Dropout(rate=0.2)(x)
    x = Dense(units=230, activation='tanh', kernel_initializer=tf.keras.initializers.HeNormal())(x)
    bifurcation = Dropout(rate=0.05)(x)

    bo = Dense(units=10, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(bifurcation)
    bo = Dropout(rate=0.05)(bo)
    bo = Dense(units=10, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(bo)
    bo = Dense(units=10, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(bo)
    binary_output = Dense(units=1, activation='sigmoid', name='binary_output')(bo)

    mo = Dense(units=100, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(bifurcation)
    mo = Dropout(rate=0.05)(mo)
    mo = Dense(units=10, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(mo)
    mo = Dense(units=10, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(mo)
    multiclass_output = Dense(units=6, activation='sigmoid', name='multiclass_output')(mo)

    return Model(inputs=inputs, outputs=[binary_output, multiclass_output])


def architecture_5() -> Model:
    # Builds best architecture for the binary model as multioutput model
    inputs = tf.keras.Input(shape=(NUMBER_OF_PREDICTORS,))

    x = Dense(units=450, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(inputs)
    x = Dropout(rate=0.2)(x)
    x = Dense(units=230, activation='tanh', kernel_initializer=tf.keras.initializers.HeNormal())(x)
    bifurcation = Dropout(rate=0.05)(x)

    bo = Dense(units=10, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(bifurcation)
    bo = Dropout(rate=0.05)(bo)
    bo = Dense(units=10, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(bo)
    bo = Dense(units=10, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(bo)
    binary_output = Dense(units=1, activation='sigmoid', name='binary_output')(bo)

    mo = Dense(units=90, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(bifurcation)
    mo = Dropout(rate=0.05)(mo)
    mo = Dense(units=10, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(mo)
    mo = Dense(units=10, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(mo)
    multiclass_output = Dense(units=6, activation='sigmoid', name='multiclass_output')(mo)

    return Model(inputs=inputs, outputs=[binary_output, multiclass_output])


def architecture_6() -> Model:
    # Builds best architecture for the binary model as multioutput model
    inputs = tf.keras.Input(shape=(NUMBER_OF_PREDICTORS,))

    x = Dense(units=450, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(inputs)
    x = Dropout(rate=0.2)(x)
    x = Dense(units=230, activation='tanh', kernel_initializer=tf.keras.initializers.HeNormal())(x)
    bifurcation = Dropout(rate=0.05)(x)

    bo = Dense(units=10, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(bifurcation)
    bo = Dropout(rate=0.05)(bo)
    bo = Dense(units=10, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(bo)
    bo = Dense(units=10, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(bo)
    binary_output = Dense(units=1, activation='sigmoid', name='binary_output')(bo)

    mo = Dense(units=120, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(bifurcation)
    mo = Dropout(rate=0.05)(mo)
    mo = Dense(units=10, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(mo)
    mo = Dense(units=10, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(mo)
    multiclass_output = Dense(units=6, activation='sigmoid', name='multiclass_output')(mo)

    return Model(inputs=inputs, outputs=[binary_output, multiclass_output])


def run_multiclass_multilabel_model_tuning(config: Dict) -> None:
    """
    Trains a two output neural network:
    Output 1. Binary classification
    Output 2. Multiclass multilabel classification
    """
    data_dict = {}
    for set in ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']:
        path_to_set = Path.cwd().parent / config['data'][set]
        if set.split('_')[0] == 'y':
            data = pd.read_csv(path_to_set)
            bina = np.array(data['Failure'])
            mult = np.array(data[['NF', 'RM', 'DPF', 'NPF', 'DWF', 'NWF']])
            data_dict[set] = (bina, mult)
        else:
            data_dict[set] = pd.read_csv(path_to_set).to_numpy()

    # Scale the features
    scaler = StandardScaler()
    x_training_scaled = scaler.fit_transform(data_dict['X_train'])
    x_val_scaled = scaler.transform(data_dict['X_val'])
    x_test_scaled = scaler.transform(data_dict['X_test'])
    dump(scaler, open('scaler.pkl', 'wb'))

    # Build and Train Different Architectures
    architectures = [architecture_1, architecture_2, architecture_3, architecture_4, architecture_5, architecture_6]
    for idx, architecture in enumerate(architectures):
        model = architecture()

        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0033),
            loss = {
                'binary_output': 'binary_crossentropy',
                'multiclass_output': macro_soft_f1
            },
            metrics = {
                'binary_output': ["accuracy", "Recall", "AUC"],
                'multiclass_output': ["accuracy", "Recall", macro_f1]
            }
        )

        history = model.fit(
            x_training_scaled,
            data_dict['y_train'],
            epochs=EPOCHS,
            batch_size=256,
            validation_data=(x_val_scaled, data_dict['y_val']),
            callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')
        )

        # Print Metrics
        plot_multiple_learning_curves(history=history.history, name=f'Architecture{idx+1}')

        # Test Binary Model
        y_val = data_dict['y_val'][0]
        y_pred_val = model.predict(x_val_scaled)[0].round(0)
        cm = confusion_matrix(y_val, y_pred_val, labels=[0,1])
        plot_confusion_matrix(cm=cm, labels=BINARY_LABELS, name=f'Binary_Architecture{idx+1}')

        # Test Multiclass Model
        y_val = data_dict['y_val'][1]
        y_pred_val = model.predict(x_val_scaled)[1].round(0)
        cm = multilabel_confusion_matrix(y_val, y_pred_val)
        plot_multiple_confusion_matrices(cms=cm, labels=MULTICLASS_LABELS, name=f'Multiclass_Architecture{idx+1}')

        # Test on Test Set
        # Test Binary Model
        y_val = data_dict['y_test'][0]
        y_pred = model.predict(x_test_scaled)[0]
        # Threshold 1
        y_pred_thresholded_labels = (y_pred > THRESHOLD_1).astype('int')
        cm = confusion_matrix(y_val, y_pred_thresholded_labels, labels=[0, 1])
        plot_confusion_matrix(cm=cm, labels=BINARY_LABELS, name=f'Test_Binary_Architecture{idx+1}_1')
        # Threshold 2
        y_pred_thresholded_labels = (y_pred > THRESHOLD_2).astype('int')
        cm = confusion_matrix(y_val, y_pred_thresholded_labels, labels=[0, 1])
        plot_confusion_matrix(cm=cm, labels=BINARY_LABELS, name=f'Test_Binary_Architecture{idx+1}_2')
        # Threshold 3
        y_pred_thresholded_labels = (y_pred > THRESHOLD_3).astype('int')
        cm = confusion_matrix(y_val, y_pred_thresholded_labels, labels=[0, 1])
        plot_confusion_matrix(cm=cm, labels=BINARY_LABELS, name=f'Test_Binary_Architecture{idx+1}_3')

        # Test Multiclass Model
        y_val = data_dict['y_test'][1]
        y_pred_val = model.predict(x_test_scaled)[1].round(0)
        for thresh in [0.05,0.10,0.15,0.20,0.25,0.30]:
            compute_multiclass_multilabel_metrics(y_target=y_val,
                                                  y_pred=model.predict(x_test_scaled)[1],
                                                  threshold=thresh,
                                                  file_name=f'Test_Multiclass_Architecture{idx+1}_Metrics.csv')
        cm = multilabel_confusion_matrix(y_val, y_pred_val)
        plot_multiple_confusion_matrices(cms=cm, labels=MULTICLASS_LABELS, name=f'Test_Multiclass_Architecture{idx+1}')

        model.save(f'multioutpput_model_architecture{idx+1}.h5')


if __name__ == '__main__':
    """
    Trains a multiclass multilabel model for slope stability based on the best architecture for binary model
    Labels:
    - Failure vs. No-Failure
    - No Failure, Rock Mass Failure, Daylighting Planar Failure, Non-daylighting Planar Failure, Daylighting Wedge 
      Failure and Non-daylighting Wedge Failure
    """
    # Reset
    tf.keras.backend.clear_session()

    # Load Config
    config = load_yaml(CONFIG)

    # Tune Model
    run_multiclass_multilabel_model_tuning(config=config)