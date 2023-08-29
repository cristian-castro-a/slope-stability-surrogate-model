from typing import Dict
from pathlib import Path

import pandas as pd
from pickle import dump, load
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from utils.loader import load_yaml
from utils.plotting import plot_confusion_matrix

CONFIG_FILE = 'config.yaml'

NUMBER_OF_PREDICTORS = 26
LABELS = ['No_Failure', 'Failure']


def run_binary_random_forest(config: Dict) -> None:
    # Load data
    X_train = pd.read_csv(Path.cwd().parent / config['data']['X_train'])
    y_train = pd.read_csv(Path.cwd().parent / config['data']['y_train'])
    X_val = pd.read_csv(Path.cwd().parent / config['data']['X_val'])
    y_val = pd.read_csv(Path.cwd().parent / config['data']['y_val'])

    X_training = pd.concat([X_train, X_val])
    y_training = pd.concat([y_train, y_val])

    dtc = DecisionTreeClassifier(class_weight='balanced', random_state=42)
    abc = AdaBoostClassifier(base_estimator=dtc, random_state=42)

    param_grid = {
        'base_estimator__criterion': ['gini', 'entropy'],
        'base_estimator__splitter': ['best', 'random'],
        'base_estimator__max_depth': [5, 10, 15, 20, 25, 30, 40],
        'base_estimator__max_features': ['auto', 'sqrt', 'log2'],
        'n_estimators': [100, 200, 300, 400, 500, 600, 700],
        'algorithm': ['SAMME', 'SAMME.R']
    }

    CV_abc = GridSearchCV(estimator=abc,
                          param_grid=param_grid,
                          cv=5,
                          scoring='accuracy',
                          verbose=3,
                          return_train_score=True)
    CV_abc.fit(X_training, y_training)

    # Best Model
    print(CV_abc.best_params_)

    best_model = CV_abc.best_estimator_
    dump(best_model, open('best_abc.pkl', 'wb'))

    cross_valscore = cross_val_score(estimator=best_model, X=X_training, y=y_training, cv=5)
    print(f"Cross Val Score: {cross_valscore}")

    y_pred = best_model.predict(pd.read_csv(Path.cwd().parent / config['data']['X_test']))
    y_test = pd.read_csv(Path.cwd().parent / config['data']['y_test'])
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    plot_confusion_matrix(cm=cm, labels=LABELS)


if __name__ == '__main__':
    """
    Trains a binary classifier using Random Forest
    """
    config = load_yaml(CONFIG_FILE)
    #run_binary_random_forest(config=config)

    X_train = pd.read_csv(Path.cwd().parent / config['data']['X_train'])
    y_train = pd.read_csv(Path.cwd().parent / config['data']['y_train'])
    X_val = pd.read_csv(Path.cwd().parent / config['data']['X_val'])
    y_val = pd.read_csv(Path.cwd().parent / config['data']['y_val'])

    X_training = pd.concat([X_train, X_val])
    y_training = pd.concat([y_train, y_val])

    reload_model = load(open('best_abc.pkl', 'rb'))
    cross_valscore = cross_val_score(estimator=reload_model, X=X_training, y=y_training, cv=5)
    print(f"Cross Val Score: {cross_valscore}")