import shutil
from pathlib import Path
from pickle import dump

import keras_tuner as kt
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential

from utils.plotting import plot_learning_curves, plot_confusion_matrix

X_TRAIN = 'X_train.csv'
X_VAL = 'X_val.csv'
Y_TRAIN = 'y_train.csv'
Y_VAL = 'y_val.csv'

NUMBER_OF_PREDICTORS = 26
LABELS = ['No_Failure', 'Failure']
OUTPUT_BIAS = 0.01

DELETE_PREVIOUS_TRIALS = False
TENSORBOARD_LOG_DIR = 'he_auc_tensorboard'
TUNER_DIR = 'he_auc_tuner'
BEST_MODEL_DIR = 'he_auc_best_model'
BEST_MODEL_NAME = 'he_auc_best_model'

OBJECTIVE = 'val_auc'
OBJECTIVE_DIRECTION = 'max'
EPOCHS = 200
MAX_TRIALS = 50


class HyperModel(kt.HyperModel):
    """
    Hypermodel for Keras Tuner
    """
    def build(self, hp):
        model = Sequential()
        model.add(
            Dense(
                units=hp.Int("units_0", min_value=100, max_value=600, step=10),
                activation=hp.Choice(f"activation_layer_0", ["relu", "tanh"]),
                kernel_initializer=tf.keras.initializers.HeNormal(),
                #bias_initializer=tf.keras.initializers.Constant(OUTPUT_BIAS),
                name=f"Dense_0"
            )
        )
        if hp.Boolean("dropout_layer_0"):
            model.add(Dropout(rate=hp.Float("rate_layer_0", min_value=0.05, max_value=0.2, step=0.05)))
        for i in range(1, hp.Int("num_layers", 1, 5)):
            if hp.Boolean(f"batch_norm_{i}"):
                model.add(BatchNormalization())
            model.add(
                Dense(
                    units=hp.Int(f"units_{i}", min_value=10, max_value=model.get_layer(f"Dense_{i-1}").units, step=10),
                    activation=hp.Choice(f"activation_layer_{i}", ["relu", "tanh"]),
                    kernel_initializer=tf.keras.initializers.HeNormal(),
                    #bias_initializer=tf.keras.initializers.Constant(OUTPUT_BIAS),
                    name=f"Dense_{i}"
                )
            )
            if hp.Boolean(f"dropout_layer_{i}"):
                model.add(Dropout(rate=hp.Float(f"rate_layer_{i}", min_value=0.05, max_value=0.2, step=0.05)))
        model.add(Dense(1, activation="sigmoid"))
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy", "Recall", "AUC"],
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size = hp.Choice("batch_size", [2,6,10,14]),
            class_weight = {0: hp.Float("0_weight", min_value=0.5, max_value=1.0, step=0.1),
                            1: hp.Float("1_weight", min_value=1.0, max_value=1.5, step=0.1)},
            **kwargs,
        )


def extract_history(best_trial):
    """
    Extracts history for tuner best trail, to plot learning curves.
    """
    acc = []
    val_acc = []
    loss = []
    val_loss = []

    size_guidance = {
        'scalars': 0,
        'images': 0,
        'audio': 0,
        'histograms': 0,
        'compressedHistograms': 0,
        'tensors': EPOCHS,
    }

    for set_data in ['train', 'validation']:
        if set_data == 'train':
          ea = EventAccumulator(Path.cwd() / TENSORBOARD_LOG_DIR / best_trial / 'execution0' / set_data,
                                size_guidance=size_guidance)
          ea.Reload()
          for i in range(len(ea.Tensors('epoch_loss'))):
              acc.append(float(tf.make_ndarray(ea.Tensors('epoch_accuracy')[i].tensor_proto)))
              loss.append(float(tf.make_ndarray(ea.Tensors('epoch_loss')[i].tensor_proto)))

        if set_data == 'validation':
          ea = EventAccumulator(Path.cwd() / TENSORBOARD_LOG_DIR / best_trial / 'execution0' / set_data,
                                size_guidance=size_guidance)
          ea.Reload()
          for i in range(len(ea.Tensors('epoch_loss'))):
            val_acc.append(float(tf.make_ndarray(ea.Tensors('epoch_accuracy')[i].tensor_proto)))
            val_loss.append(float(tf.make_ndarray(ea.Tensors('epoch_loss')[i].tensor_proto)))

    return acc, val_acc, loss, val_loss


def run_binary_keras_tuner(X_train_dir: Path, y_train_dir: Path, X_val_dir: Path, y_val_dir: Path) -> None:
    """
    Trains a binary base model: Failure vs. No-Failure
    """
    X_train = pd.read_csv(X_train_dir).to_numpy()
    y_train = pd.read_csv(y_train_dir).to_numpy()

    X_val = pd.read_csv(X_val_dir).to_numpy()
    y_val = pd.read_csv(y_val_dir).to_numpy()

    # Scale the features
    scaler = StandardScaler()
    x_training_scaled = scaler.fit_transform(X_train)
    x_val_scaled = scaler.transform(X_val)
    dump(scaler, open('scaler.pkl', 'wb'))

    # Keras Tuner
    callbacks = [
        tf.keras.callbacks.TensorBoard(TENSORBOARD_LOG_DIR),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')]

    tuner = kt.RandomSearch(
        HyperModel(),
        objective=kt.Objective(OBJECTIVE, direction=OBJECTIVE_DIRECTION),
        max_trials=MAX_TRIALS,
        executions_per_trial=3,
        overwrite=True,
        directory=TUNER_DIR
    )

    tuner.search(x_training_scaled,
                 y_train,
                 validation_data=(x_val_scaled, y_val),
                 epochs=EPOCHS,
                 callbacks=callbacks)

    # Best Trial: Learning Curves
    best_trial = tuner.oracle.get_best_trials()[0].trial_id
    acc, val_acc, loss, val_loss = extract_history(best_trial=best_trial)
    history = pd.DataFrame.from_dict({'acc': acc, 'val_acc': val_acc, 'loss': loss, 'val_loss': val_loss})
    plot_learning_curves(history=history, validation=True)

    # Best Model Validation: Confusion Matrix
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.build(input_shape=(None, NUMBER_OF_PREDICTORS))
    best_model.summary()
    best_model.save(f'{BEST_MODEL_DIR}/{BEST_MODEL_NAME}.h5')

    y_pred_val = best_model.predict(x_val_scaled).round(0)
    cm = confusion_matrix(y_val, y_pred_val, labels=[0,1])
    plot_confusion_matrix(cm=cm, labels=LABELS)


if __name__ == '__main__':
    """
    Trains a base binary surrogate model for slope stability.
    Labels: Failure vs. No-Failure
    """
    # Reset
    tf.keras.backend.clear_session()

    # Delete previous trials
    if DELETE_PREVIOUS_TRIALS:
        tuner_dir = Path.cwd() / TUNER_DIR
        if tuner_dir.is_dir():
            shutil.rmtree(tuner_dir)

        log_dir = Path.cwd() / TENSORBOARD_LOG_DIR
        if log_dir.is_dir():
            shutil.rmtree(log_dir)

    # Data
    data_source_path = Path.cwd().parent / '03_01_data_split'
    X_train_dir = data_source_path / X_TRAIN
    y_train_dir = data_source_path / Y_TRAIN
    X_val_dir = data_source_path / X_VAL
    y_val_dir = data_source_path / Y_VAL

    # Model
    #run_binary_keras_tuner(X_train_dir=X_train_dir,
    #                       y_train_dir=y_train_dir,
    #                       X_val_dir=X_val_dir,
    #                       y_val_dir=y_val_dir)