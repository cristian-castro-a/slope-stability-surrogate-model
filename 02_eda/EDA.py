from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DATASET = 'final_dataset.xlsx'
PNG_FOLDER = 'pngs_final'
HTML_FOLDER = 'htmls_final'
GENERAL_FOLDER = 'general'
MOF_FOLDER = 'mof'


class EDAConfig:
    def __init__(self, png_folder: str, html_folder: str, general_folder: str, mof_folder: str) -> None:
        self.cwd = Path.cwd()
        self.png_folder_path = self.cwd / png_folder
        self.html_folder_path = self.cwd / html_folder
        self.general_png_folder_path = self.png_folder_path / general_folder
        self.mof_png_folder_path = self.png_folder_path / mof_folder
        self.general_html_folder_path = self.html_folder_path / general_folder
        self.mof_html_folder_path = self.html_folder_path / mof_folder

        if not self.cwd.is_dir():
            self.cwd.mkdir()

        if not self.png_folder_path.is_dir():
            self.png_folder_path.mkdir()

        if not self.html_folder_path.is_dir():
            self.html_folder_path.mkdir()

        if not self.general_html_folder_path.is_dir():
            self.general_html_folder_path.mkdir()

        if not self.mof_html_folder_path.is_dir():
            self.mof_html_folder_path.mkdir()

        if not self.general_png_folder_path.is_dir():
            self.general_png_folder_path.mkdir()

        if not self.mof_png_folder_path.is_dir():
            self.mof_png_folder_path.mkdir()


def plot_histogram_total(df: pd.DataFrame, var_name: str) -> None:
    """
    Plots the var_name from a dataframe
    """
    file_name = f"{var_name}_histogram"
    file_path_png = directories.general_png_folder_path / f"{file_name}.png"
    file_path_html = directories.general_html_folder_path / f"{file_name}.html"
    title = f"{var_name} - Histogram"

    fig = px.histogram(
        df,
        x=var_name,
        nbins=20,
        marginal='box',
        title=title
    )

    fig.update_layout(
        title=dict(
            font=dict(size=30)),
        yaxis=dict(tickfont=dict(size=20)),
        xaxis=dict(tickfont=dict(size=20)),
        font=dict(size=25)
    )

    fig.write_html(file_path_html)
    fig.write_image(file_path_png, width=1600, height=1000, scale=1)


def plot_histogram(df: pd.DataFrame, var_name: str, subset: str=None) -> None:
    """
    Plots the var_name from a dataframe for a given subset
    """
    df = df.loc[df[subset]==1]

    if subset is not None:
        file_name = f"{var_name}_{subset}_histogram"
        file_path_png = directories.mof_png_folder_path / f"{file_name}.png"
        file_path_html = directories.mof_html_folder_path / f"{file_name}.html"
        title = f"{var_name} - {subset} - Histogram"
    else:
        file_name = f"{var_name}_histogram"
        file_path_png = directories.mof_png_folder_path / f"{file_name}.png"
        file_path_html = directories.mof_html_folder_path / f"{file_name}.html"
        title = f"{var_name} - Histogram"

    fig = px.histogram(
        df,
        x=var_name,
        nbins=20,
        marginal='box',
        title=title
    )

    fig.update_layout(
        title=dict(
            font=dict(size=30)),
        yaxis=dict(tickfont=dict(size=20)),
        xaxis=dict(tickfont=dict(size=20)),
        font=dict(size=25)
    )

    fig.write_html(file_path_html)
    fig.write_image(file_path_png, width=1600, height=1000, scale=1)


def plot_bars(df: pd.DataFrame, var_names: List[str]) -> None:
    """
    Plot the percentages and counts of failure mechanisms in the dataset
    """
    mof = df[var_names].sum()
    total_mof = sum(mof)
    percentages = 100 * mof / total_mof

    text = []
    for count, percentage in zip(mof.values, percentages):
        text.append(f"{round(percentage,1)}% <br> {count} Cases")

    percentages_df = pd.DataFrame({'Failure Mechanism': mof.index, 'Counts': mof.values, 'Percentage': percentages, 'Text': text})

    fig = px.bar(percentages_df,
                 x='Failure Mechanism',
                 y='Counts',
                 text="Text",
                 title='Distribution of Failure Mechanisms in Dataset')
    fig.update_traces(textposition='inside', textfont_size=25)
    fig.update_layout(
        title=dict(
            font=dict(size=40)),
        yaxis=dict(tickfont=dict(size=25)),
        xaxis=dict(tickfont=dict(size=25)),
        font=dict(size=30)
    )
    file_path_png = directories.general_png_folder_path / "Distribution_of_mof.png"
    file_path_html = directories.general_html_folder_path / "Distribution_of_mof.html"
    fig.write_html(file_path_html)
    fig.write_image(file_path_png, width=1600, height=1000, scale=1)


def plot_table(df: pd.DataFrame, var_names: List[str]) -> None:
    """
    Plot table with descriptive statistics
    """
    table = df[var_names].describe().round(1).T.reset_index()
    table = table.rename(columns={'index': 'Measure'})
    file_name = "Descriptive_statistics"
    file_path_png = directories.general_png_folder_path / f"{file_name}.png"
    file_path_html = directories.general_html_folder_path / f"{file_name}.html"

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=table.columns,
                    font=dict(size=30),
                    align="center",
                    height=30),
                cells=dict(
                    values=[table[k].tolist() for k in table.columns],
                    font=dict(size=30),
                    align="center",
                    height=30)
            )
        ])

    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")

    fig.write_html(file_path_html)
    fig.write_image(file_path_png, width=1600, height=1000, scale=1)


def plot_table_per_var_name(df: pd.DataFrame, var_name: str) -> None:
    """
    Plot table with descriptive statistics per variable
    """
    table = df[var_name].describe().T
    file_name = f"{var_name}_descriptive_statistics"
    file_path_png = directories.general_png_folder_path / f"{file_name}.png"
    file_path_html = directories.general_html_folder_path / f"{file_name}.html"

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Measure", "Value"],
                    font=dict(size=30),
                    align="center",
                    height=40),
                cells=dict(
                    values=[table.index.values, table.values.round(1)],
                    font=dict(size=30),
                    align="center",
                    height=40)
            )
        ])

    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")

    fig.write_html(file_path_html)
    fig.write_image(file_path_png, width=800, height=550, scale=1)


def plot_scatter_matrix(df: pd.DataFrame, var_names: List[str], var_names_group: str) -> None:
    """
    Plots a scatter matrix with the variables listed
    """
    # Change the dataframe from one-hot-encoding to str names
    df['Failure_Mechanism'] = df[['NF','RM','DPF','NPF','DWF','NWF']].idxmax(axis=1)

    # File names
    file_path_png = directories.general_png_folder_path / f"Scatter_matrix_{var_names_group}.png"
    file_path_html = directories.general_html_folder_path / f"Scatter_matrix_{var_names_group}.html"


    fig = px.scatter_matrix(df,
                            dimensions=var_names,
                            color="Failure_Mechanism",
                            title=var_names_group)
    fig.update_layout(
        title=dict(
            font=dict(size=35)),
        yaxis=dict(tickfont=dict(size=20)),
        xaxis=dict(tickfont=dict(size=20)),
        font=dict(size=22)
    )
    fig.write_html(file_path_html)
    fig.write_image(file_path_png, width=1600, height=1000, scale=1)


def run(df: pd.DataFrame) -> None:
    """
    EDA
    """
    mof = ['NF','RM','DPF','NPF','DWF','NWF']
    plot_bars(df=df, var_names=mof)

    list_var = [
        'slope_height',
        'slope_ira',
        'interface_1_dip',
        'interface_1_dd',
        'interface_1_coh',
        'interface_1_fri',
        'interface_2_dip',
        'interface_2_dd',
        'interface_2_coh',
        'interface_2_fri',
        'rock_density',
        'young_modulus',
        'poisson_ratio',
        'UCS',
        'phreatic_level',
        'GSI',
        'mi',
        'ratio',
        'distance'
    ]

    # Descriptive statistics
    plot_table(df=df, var_names=list_var)

    # Make scatter plot per group of variables
    physical_elastic_props = ['rock_density', 'young_modulus', 'poisson_ratio']
    plot_scatter_matrix(df=df, var_names=physical_elastic_props, var_names_group='Physical and Elastic Rock Properties')

    strength_props = ['UCS', 'GSI', 'mi']
    plot_scatter_matrix(df=df, var_names=strength_props, var_names_group='Strength and Rock Mass Properties')

    geological_orientations = ['interface_1_dip', 'interface_1_dd', 'interface_2_dip', 'interface_2_dd']
    plot_scatter_matrix(df=df, var_names=geological_orientations, var_names_group='Orientations of Geological Structures')

    geological_strengths = ['interface_1_coh', 'interface_1_fri', 'interface_2_coh', 'interface_2_fri']
    plot_scatter_matrix(df=df, var_names=geological_strengths, var_names_group='Strength of Geological Structures')

    slope = ['slope_height', 'slope_ira', 'phreatic_level']
    plot_scatter_matrix(df=df, var_names=slope, var_names_group='Slope Geometry Parameters')

    mapping = ['ratio', 'distance']
    plot_scatter_matrix(df=df, var_names=mapping, var_names_group='Mapping Parameters')

    for var in list_var:
        plot_table_per_var_name(df=df, var_name=var)
        plot_histogram_total(df=df, var_name=var)

    for item in mof:
        for var in list_var:
            plot_histogram(df=df, var_name=var, subset=item)


if __name__ == '__main__':
    """
    EDA for Generated Dataset
    """
    directories = EDAConfig(png_folder=PNG_FOLDER, html_folder=HTML_FOLDER, general_folder=GENERAL_FOLDER,
                            mof_folder=MOF_FOLDER)
    df = pd.read_excel(DATASET)

    # Add distance between mapping points
    df['distance'] = np.sqrt((df.interface_1_x - df.interface_2_x)**2 + (df.interface_1_y - df.interface_2_y)**2 +
                         (df.interface_1_z - df.interface_2_z)**2)

    # Get the ratio of the distance between mapping points and slope height
    df['ratio'] = df.distance / df.slope_height

    run(df)