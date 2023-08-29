from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET = 'dataset_7_classes_v2.xlsx'
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.2

FAILURE_MECHANISMS = ['NF', 'RM', 'DPF', 'NPF', 'DWF', 'NWF']
BINARY_LABELS = ['No_Failure', 'Failure']


def make_binary_classification(df: pd.DataFrame,  binary_labels: List) -> pd.DataFrame:
    """
    Transform a dataframe of multiclass classication into a binary classification one
    """
    map_dict = {}
    for fm in FAILURE_MECHANISMS:
        if fm == 'NF':
            map_dict[fm] = BINARY_LABELS[0]
        else:
            map_dict[fm] = BINARY_LABELS[1]

    df = df.replace({'fm': map_dict})

    # Invert the logic for binary classification
    df = df.rename(columns={'NF': 'Failure'})
    df['Failure'] = 1 - df['Failure']

    return df.drop(columns=FAILURE_MECHANISMS[1:])


def run(dataset_path: Path) -> None:
    """
    Takes the dataset, make the labels binary (failure / no failure), encodes the angle features into cos/sin
    and splits it into train, validation and test sets.
    """
    df = pd.read_excel(dataset_path)

    # Encode the angles to better representation
    df['slope_ira_cos'] = np.cos(np.radians(df.slope_ira))
    df['slope_ira_sin'] = np.sin(np.radians(df.slope_ira))
    df['int_1_dip_cos'] = np.cos(np.radians(df.interface_1_dip))
    df['int_1_dip_sin'] = np.sin(np.radians(df.interface_1_dip))
    df['int_1_dd_cos'] = np.cos(np.radians(df.interface_1_dd))
    df['int_1_dd_sin'] = np.sin(np.radians(df.interface_1_dd))
    df['int_1_fri_cos'] = np.cos(np.radians(df.interface_1_fri))
    df['int_1_fri_sin'] = np.sin(np.radians(df.interface_1_fri))
    df['int_2_dip_cos'] = np.cos(np.radians(df.interface_2_dip))
    df['int_2_dip_sin'] = np.sin(np.radians(df.interface_2_dip))
    df['int_2_dd_cos'] = np.cos(np.radians(df.interface_2_dd))
    df['int_2_dd_sin'] = np.sin(np.radians(df.interface_2_dd))
    df['int_2_fri_cos'] = np.cos(np.radians(df.interface_2_fri))
    df['int_2_fri_sin'] = np.sin(np.radians(df.interface_2_fri))

    # Add distance between mapping points
    df['distance'] = np.sqrt((df.interface_1_x - df.interface_2_x)**2 + (df.interface_1_y - df.interface_2_y)**2 +
                             (df.interface_1_z - df.interface_2_z)**2)

    # Get the ratio of the distance between mapping points and slope height
    df['ratio'] = df.distance / df.slope_height

    # Add one column to the dataframe transforming labels from one-hot-encoding to a column of str names
    df['fm'] = df[FAILURE_MECHANISMS].idxmax(axis=1)

    # Make binary classification
    df = make_binary_classification(df=df, binary_labels=BINARY_LABELS)

    # Dataset is highly imbalanced
    train, test = train_test_split(df,
                                   test_size=TEST_SIZE,
                                   random_state=42,
                                   shuffle='True',
                                   stratify=df[['fm']])

    train, validation = train_test_split(train,
                                   test_size=VALIDATION_SIZE,
                                   random_state=42,
                                   shuffle='True',
                                   stratify=train[['fm']])

    labels = ['Failure', 'fm']

    others = ['uuid', 'server', 'folder', 'expansion_factor', 'zone_size_ref', 'min_zones_in_slope', 'slope_length',
              'slope_width', 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax', 'max_dist_d', 'max_d_factor',
              'interface_1_x', 'interface_1_y', 'interface_1_z', 'interface_2_x', 'interface_2_y', 'interface_2_z',
              'interface_1_dip', 'interface_1_dd', 'interface_2_dip', 'interface_2_dd', 'slope_ira',
              'interface_1_fri', 'interface_2_fri']

    predictors = [i for i in df.columns if i not in [k for k in labels + others]]

    y_train, y_val, y_test = train[labels[:-1]], validation[labels[:-1]], test[labels[:-1]]
    X_train, X_val, X_test = train[predictors], validation[predictors], test[predictors]

    dataframes_dict = {'y_train': y_train,
                       'y_val': y_val,
                       'y_test': y_test,
                       'X_train': X_train,
                       'X_val': X_val,
                       'X_test': X_test}

    for k, v in dataframes_dict.items():
        if k.split('_')[0] == 'y':
            for label in BINARY_LABELS:
                if label == 'No_Failure':
                    proportion = v.loc[v['Failure'] == 0].count().values[0]/v.shape[0]
                    print(f"For {k}, {label} has {v.loc[v['Failure'] == 0].count().values[0]} cases which is: "
                          f"{round(proportion,2)}")
                else:
                    proportion = v.loc[v['Failure'] == 1].count().values[0]/v.shape[0]
                    print(f"For {k}, {label} has {v.loc[v['Failure'] == 1].count().values[0]} cases which is: "
                          f"{round(proportion,2)}")
            print(f"For {k}, the total amount of cases is: {v.shape[0]}")
            print("-"*50)

        path_to_save = Path.cwd() / f"{k}.csv"
        v.to_csv(path_to_save, index=False)


if __name__ == '__main__':
    """
    Data Split
    """
    dataset_path = Path().cwd().parent.parent / '01_dataset_generation' / DATASET
    run(dataset_path=dataset_path)