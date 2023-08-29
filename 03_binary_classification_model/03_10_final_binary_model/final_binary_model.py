import json
import os
from pathlib import Path
from pickle import dump
from typing import Dict, Tuple

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

from utils.loader import load_yaml
from utils.plotting import plot_learning_curves

CONFIG_FILE = 'config.yaml'

NUMBER_OF_PREDICTORS = 26
LABELS = ['No_Failure', 'Failure']


def gather_data(config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    # Predictors
    X_train = pd.read_csv(Path.cwd().parent / config['data']['X_train']).to_numpy()
    X_val = pd.read_csv(Path.cwd().parent / config['data']['X_val']).to_numpy()
    X_test = pd.read_csv(Path.cwd().parent / config['data']['X_test']).to_numpy()
    predictors = np.concatenate((X_train, X_val, X_test), axis=0)
    # Targets
    y_train = pd.read_csv(Path.cwd().parent / config['data']['y_train']).to_numpy()
    y_val = pd.read_csv(Path.cwd().parent / config['data']['y_val']).to_numpy()
    y_test = pd.read_csv(Path.cwd().parent / config['data']['y_test']).to_numpy()
    targets = np.concatenate((y_train, y_val, y_test), axis=0)
    return predictors, targets


def search_weights(lr: float, directory: Path) -> Tuple[float, float]:
    weight_0, weight_1 = 1, 1
    for subdir, dirs, files in os.walk(directory):
        for dir in dirs:
            path_to_json = directory / dir / 'trial.json'
            with open(path_to_json) as jsonfile:
                trial_dict = json.load(jsonfile)
            if round(lr,8) == round(trial_dict['hyperparameters']['values']['lr'],8):
                weight_0 = trial_dict['hyperparameters']['values']['0_weight']
                weight_1 = trial_dict['hyperparameters']['values']['1_weight']
                break
    return weight_0, weight_1


def run_final_binary_model(config: Dict) -> None:
    predictors, targets = gather_data(config=config)
    scaler = StandardScaler()
    predictors_scaled = scaler.fit_transform(predictors)
    dump(scaler, open('scaler.pkl', 'wb'))

    model_dir = Path.cwd().parent / config['best_model']
    model = load_model(model_dir)

    # Re-train on the entire
    lr = float(model.optimizer.lr)
    dir = Path.cwd().parent / '03_06_keras_tuner' / 'he_loss_tuner' / 'untitled_project'
    weight_0, weight_1 = search_weights(lr=lr, directory=dir)

    model.compile(
        loss=config['to_compile']['loss'],
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=config['to_compile']['metrics']
    )

    history = model.fit(
                        predictors_scaled,
                        targets,
                        batch_size=2,
                        epochs=config['to_train']['epochs'],
                        class_weight={0: weight_0, 1: weight_1}
                    )

    model.save('binary_model.h5')

    # Learning Curves
    history_df = pd.DataFrame.from_dict(history.history).rename(columns={'accuracy': 'acc', 'val_accuracy': 'val_acc'})
    plot_learning_curves(history=history_df)


if __name__ == '__main__':
    """
    Trains a base binary surrogate model for slope stability.
    Labels: Failure vs. No-Failure
    """
    tf.keras.backend.clear_session()
    config = load_yaml(CONFIG_FILE)

    # Model
    #run_final_binary_model(config=config)

    # Model Information
    model = load_model('binary_model.h5')
    model.build(input_shape=(None, NUMBER_OF_PREDICTORS))
    model.summary()

    model_info = h5py.File('binary_model.h5', 'r')
    model_config_json = json.loads(model_info.attrs['model_config'])

    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, show_layer_activations=True)