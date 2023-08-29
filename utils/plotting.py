from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_learning_curves(history: pd.DataFrame, validation: bool=False) -> None:
    """
    Two axes plot for learning curves
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=[i for i in range(history.shape[0]+1)],
                   y=history['acc'],
                   mode='lines+markers',
                   name='Training Accuracy'),
        secondary_y=True)
    if validation:
        fig.add_trace(
            go.Scatter(x=[i for i in range(history.shape[0]+1)],
                       y=history['val_acc'],
                       mode='lines+markers',
                       name='Validation Accuracy'),
            secondary_y=True)
    fig.add_trace(
        go.Scatter(x=[i for i in range(history.shape[0]+1)],
                   y=history['loss'],
                   mode='lines+markers',
                   name='Training Loss'),
        secondary_y=False)
    if validation:
        fig.add_trace(
            go.Scatter(x=[i for i in range(history.shape[0]+1)],
                       y=history['val_loss'],
                       mode='lines+markers',
                       name='Validation Loss'),
            secondary_y=False)

    fig.update_layout(
        title_text="Learning Curves"
    )
    fig.update_xaxes(title_text="Epochs")
    fig.update_yaxes(title_text='Loss', secondary_y=False, range=[0,1])
    fig.update_yaxes(title_text='Accuracy', secondary_y=True, range=[0,1])
    fig.write_html('learning_curves.html')


def plot_confusion_matrix(cm: np.ndarray, labels: List, name: Optional[int] = None) -> None:
    fig = px.imshow(cm,
                    text_auto=True,
                    x=labels,
                    y=labels,
                    labels=dict(x='Predicted Label', y='True Label'))
    if name is not None:
        fig.update_layout(title_text=f'Confusion Matrix for {name}')
        fig.write_html(f'confusion_matrix_{name}.html')
    else:
        fig.update_layout(title_text='Confusion Matrix')
        fig.write_html('confusion_matrix.html')


def plot_roc_curves(data: Dict) -> None:
    """
    Data is a dictionary.
    Data[model] = [fpr, tpr, thresholds]
    """
    fig = go.Figure()

    for k, v in data.items():
        fig.add_trace(
            go.Scatter(x=v[0],
                       y=v[1],
                       mode='lines+markers',
                       name=k,
                       hovertext=[f"Threshold: {i}" for i in v[2]]))

    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_layout(
        title=dict(
            text='ROC Curves over Test Set',
            font=dict(size=30)
        ),
        font=dict(size=18)
    )

    fig.update_xaxes(title_text="False Positive Rate", range=[0,1])
    fig.update_yaxes(title_text='True Positive Rate', range=[0, 1])
    fig.write_html('roc_curves.html')


def plot_multiple_learning_curves(history: Dict, name: Optional[int] = None) -> None:
    """
    Two axes plot for learning curves for binary output (loss and accuracy) and multiclass output (loss and f1)
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    length = len(history['binary_output_accuracy'])+1
    fig.add_trace(
        go.Scatter(x=[i for i in range(length)],
                   y=history['binary_output_accuracy'],
                   mode='lines+markers',
                   name='Training Binary Output Accuracy'),
        secondary_y=True)
    fig.add_trace(
        go.Scatter(x=[i for i in range(length)],
                   y=history['val_binary_output_accuracy'],
                   mode='lines+markers',
                   name='Validation Binary Output Accuracy'),
        secondary_y=True)
    fig.add_trace(
        go.Scatter(x=[i for i in range(length)],
                   y=history['binary_output_loss'],
                   mode='lines+markers',
                   name='Training Binary Output Loss'),
        secondary_y=False)
    fig.add_trace(
        go.Scatter(x=[i for i in range(length)],
                   y=history['val_binary_output_loss'],
                   mode='lines+markers',
                   name='Validation Binary Output Loss'),
        secondary_y=False)

    fig.add_trace(
        go.Scatter(x=[i for i in range(length)],
                   y=history['multiclass_output_macro_f1'],
                   mode='lines+markers',
                   name='Training Multiclass Output Macro F1'),
        secondary_y=True)
    fig.add_trace(
        go.Scatter(x=[i for i in range(length)],
                   y=history['val_multiclass_output_macro_f1'],
                   mode='lines+markers',
                   name='Validation Multiclass Output Macro F1'),
        secondary_y=True)
    fig.add_trace(
        go.Scatter(x=[i for i in range(length)],
                   y=history['multiclass_output_loss'],
                   mode='lines+markers',
                   name='Training Multiclass Output Loss'),
        secondary_y=False)
    fig.add_trace(
        go.Scatter(x=[i for i in range(length)],
                   y=history['val_multiclass_output_loss'],
                   mode='lines+markers',
                   name='Validation Multiclass Output Loss'),
        secondary_y=False)

    fig.update_layout(
        title_text="Learning Curves"
    )
    fig.update_xaxes(title_text="Epochs")
    fig.update_yaxes(title_text='Loss', secondary_y=False, range=[0,1])
    fig.update_yaxes(title_text='Accuracy / Macro F1', secondary_y=True, range=[0,1])

    if name is not None:
        fig.write_html(f'learning_curves_{name}.html')
    else:
        fig.write_html('learning_curves.html')


def plot_multiple_confusion_matrices(cms: np.ndarray, labels: List, name: Optional[int] = None) -> None:
    fig = make_subplots(rows=2, cols=3, subplot_titles=[f'Class {i}' for i in labels])

    for i in range(6):
        cm_i = cms[i]
        class_name = labels[i]
        tp, fn, fp, tn = cm_i.ravel()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        fig_cm = px.imshow(cm_i,
                           color_continuous_scale='Blues',
                           text_auto=True,
                           x=['Other', class_name],
                           y=['Other', class_name],
                           labels=dict(x='Predicted Label', y='True Label')
                           )

        fig_cm.update_layout(title_text=f'Class {i} - Precision: {precision:.2f}, Recall: {recall:.2f}')

        fig.update_xaxes(title_text='', row=1, col=1)
        fig.update_yaxes(title_text='', row=1, col=1)

        fig.add_trace(fig_cm.data[0], row=(i // 3) + 1, col=(i % 3) + 1)

    fig.update_layout(title_text='Multi-Label Confusion Matrices')

    if name is not None:
        fig.write_html(f'confusion_matrix_{name}.html')
    else:
        fig.update_layout(title_text='Confusion Matrix')
        fig.write_html('confusion_matrices.html')