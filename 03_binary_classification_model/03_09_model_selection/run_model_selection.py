from pathlib import Path
from pickle import load
from typing import Dict

import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix
from tensorflow.keras.models import load_model

from utils.loader import load_yaml
from utils.plotting import plot_roc_curves, plot_confusion_matrix

CONFIG_FILE = 'config.yaml'

THRESHOLD = 0.5
LABELS = ['No_Failure', 'Failure']


def run(config: Dict) -> None:
    """
    ROC Curve
    """
    # Load data
    X_test = pd.read_csv(Path.cwd().parent / config['data']['X_test'])
    y_test = pd.read_csv(Path.cwd().parent / config['data']['y_test']).to_numpy()

    # Scaler
    scaler = load(open(Path.cwd().parent / config['scalers']['nn_scaler'], 'rb'))
    X_test_scaled_nn = scaler.transform(X_test.to_numpy())

    # Data
    data = {}

    # Models and Predictions
    for model, model_path in config['models'].items():
        model_dir = Path.cwd().parent / model_path

        if model_dir.suffix == '.h5':
            loaded_model = load_model(model_dir)
            y_test_pred = loaded_model.predict(X_test_scaled_nn)

        else:
            loaded_model = load(open(model_dir, 'rb'))
            y_test_pred = loaded_model.predict_proba(X_test)
            y_test_pred = np.array([[i[1]] for i in y_test_pred])

        y_test_thresholded_labels = (y_test_pred > THRESHOLD).astype('int')
        cm = confusion_matrix(y_test, y_test_thresholded_labels, labels=[0, 1])
        plot_confusion_matrix(cm=cm, labels=LABELS, name=model)

        fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)
        data[f"{model}"] = [fpr, tpr, [round(i, 2) for i in thresholds]]

    # Plot ROC Curves
    plot_roc_curves(data=data)


if __name__ == '__main__':
    """
    Compare models in ROC Curve
    """
    config = load_yaml(CONFIG_FILE)
    run(config=config)