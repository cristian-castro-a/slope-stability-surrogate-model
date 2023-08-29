import csv
from pathlib import Path

import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, load_model

from utils.plotting import plot_learning_curves, plot_confusion_matrix

X_TRAIN = 'X_train.csv'
X_VAL = 'X_val.csv'
Y_TRAIN = 'y_train.csv'
Y_VAL = 'y_val.csv'

NUMBER_OF_PREDICTORS = 26
LABELS = ['No_Failure', 'Failure']


def binary_base_model() -> tf.keras.Model:
    """
    Creates the architecture for a Feed-forward Neural Network
    """
    model = Sequential()
    model.add(Dense(100, input_dim=NUMBER_OF_PREDICTORS, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=['accuracy', 'Recall', 'AUC']
    )

    return model


def run_binary_base_model(X_train_dir: Path, y_train_dir: Path, X_val_dir: Path, y_val_dir: Path) -> None:
    """
    Trains a binary base model: Failure vs. No-Failure
    """
    X_train = pd.read_csv(X_train_dir).to_numpy()
    y_train = pd.read_csv(y_train_dir).to_numpy()

    X_val = pd.read_csv(X_val_dir).to_numpy()
    y_val = pd.read_csv(y_val_dir).to_numpy()

    # Metrics to Store
    metrics_list = []

    # Scale the features
    scaler = StandardScaler()
    x_training_scaled = scaler.fit_transform(X_train)
    x_val_scaled = scaler.transform(X_val)

    # Tensorflow feed-forward neural network
    model = binary_base_model()
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(x_training_scaled,
                        y_train,
                        validation_data=(x_val_scaled, y_val),
                        epochs=200,
                        verbose=2,
                        batch_size=10,
                        callbacks=[callback])

    # Save Model
    model.save('base_model.h5')

    # Learning Curves
    history_df = pd.DataFrame.from_dict(history.history).rename(columns={'accuracy': 'acc', 'val_accuracy': 'val_acc'})
    plot_learning_curves(history=history_df, validation=True)

    # Store Metrics
    metrics_list.append([v[-1] for k, v in history.history.items()])

    # Confusion Matrix
    y_pred = model.predict(x_val_scaled).round(0).astype(int)
    cm = confusion_matrix(y_val, y_pred, labels=[0,1])
    plot_confusion_matrix(cm=cm, labels=LABELS)

    fields = history.history.keys()
    with open('summary_base_model.csv', 'w', newline='') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(metrics_list)


if __name__ == '__main__':
    """
    Trains a base binary surrogate model for slope stability.
    Labels: Failure vs. No-Failure
    """

    # Reset
    tf.keras.backend.clear_session()

    # Data
    data_source_path = Path.cwd().parent / '03_01_data_split'
    X_train_dir = data_source_path / X_TRAIN
    y_train_dir = data_source_path / Y_TRAIN
    X_val_dir = data_source_path / X_VAL
    y_val_dir = data_source_path / Y_VAL

    # Model
    # run_binary_base_model(X_train_dir=X_train_dir,
    #                       y_train_dir=y_train_dir,
    #                       X_val_dir=X_val_dir,
    #                       y_val_dir=y_val_dir)

    # Model Information
    model = load_model('base_model.h5')
    model.build(input_shape=(None, NUMBER_OF_PREDICTORS))
    model.summary()