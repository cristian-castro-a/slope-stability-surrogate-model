# Surrogate Model for Slope Stability Analysis [WIP]
This repository builds up a surrogate model aimed at predicting the failure mechanisms for an idealised continuous slope 
that exhibit two geological structures.

Two steps are involved:
1. **Dataset Generation**: It creates a dataset of results of multiple *FLAC3D* runs ('01_dataset_generation'). 
These runs include a generic continuous slope and two interfaces, that represent the occurrence of geological 
discontinuities. The setup of these runs are random, as each parameter that describes the model, the rock mass and 
the interfaces is treated as a random variable. Each variable is thought as a normal distribution, whose parameters 
are set by the user. The results of the runs are stored in a .csv file, named by the user.
2. **Model Selection and Assessment**: The generated dataset is used to train a surrogate model and give estimations 
for the probability of failure for different failure mechanisms, which are: No Failure (NF), Rock Mass Failure (RM), 
Daylighting Plane Failure (DPF), Non-daylighting Plane Failure (NPF), Daylighting Wedge Failure (DWF) and 
Non-daylighting Wedge Failure (NWF) (rest of the folder in the repo).

## Installation
This project uses the following frameworks and software:
- Python v3.10
- FISH in FLAC3D v9.0 [(Itasca Consulting Group)](https://www.itascacg.com/software/new-in-flac3d-9)
- FISH in 3DEC v9.0 [(Itasca Consulting Group)](https://www.itascacg.com/software/3dec)

The use of *FLAC3D v9.0* and *3DEC v9.0* involve the acquisition of license keys, these software are not open source.

To set a Conda virtual environment with the required dependencies in Windows:
```bash
conda create --name myenv --file requirements.txt
```

## Project Structure
1. **Dataset Generation:** The user must intervene only two files, before generating the dataset. These files are:
   * [main.py](./main.py): Entry point for the Python program. Two parameters must be set here:
     * **ITERATIONS**: Number of models to run
     * **FILE_NAME**: Name of the .csv file that will summarize all the runs
   * [input_parameters.py](./input_parameters.py): This is where all the parameters involved in the runs (random variables) are described. The user must declare each of the global variables stated in capital letters before running 'main.py'. 
2. **EDA:** Exploratory data analysis for the generated dataset.
3. **Binary Classification Model:** Builds a model for predicting Stable/Unstable classes.
4. **Multiclass Multilabel Model:** Build a multioutput model with Functional API of TensorFlow.
5. **Model Deployment:** Script for deploying the model on OnRender.com

## License
[MIT](https://choosealicense.com/licenses/mit/)