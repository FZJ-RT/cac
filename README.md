# Chip Analyzer and Calculator (cac)

An open-source Python library for applying data-driven and process-based machine learning methods to analyze microfluidics experiments.

## Introduction

`cac` provides data-driven and process-based machine learning methods for enabling efficient and real-time analysis of microfluidics experiments.
The package uses [Keras](https://keras.io/) and [Tensorflow](https://www.tensorflow.org/) for the implementation of data-driven machine learning methods, and [OpenCV](https://opencv.org/) for image processing.
It enables training with GPU, built on [Horovod](https://horovod.ai/).

The current version (v1.0) contains only the implementation for data-driven machine learning methods.
In the future, we will add more functionalities for the implementation of process-based machine learning methods.
Stay tuned!

## Installation

### Manual installation

#### Creation of a virtual environment

It is recommended to install this package in a Python [virtual environment](https://docs.python.org/3/library/venv.html).
This will allow you to manage the package and its dependencies separately from other Python packages on your system.


To create a virtual environment and activate it, use the following commands:

```bash
python -m venv env
source env/bin/activate
```

Replace `env` with the desired name of your virtual environment.
If you succeeded, the prompt will show now `(env) $` instead of `$`.
To deactivate the virtual environment use the `deactivate` command.

#### Installation of dependencies

Install manually the requirements of the package, which vary based on if you intend to use a GPU or not.
Consult the corresponding `yaml` file to learn about the requirements:

- [requirements with GPU](https://github.com/FZJ-RT/cac/blob/main/with_gpu/environment.devenv.yml)
- [requirements without GPU](https://github.com/FZJ-RT/cac/blob/main/no_gpu/environment.devenv.yml)

#### Install the package itself

To install the latest version of the package, use pip:

```bash
# Via https
pip install git+https://github.com/FZJ-RT/cac.git

# Via ssh
pip install git+ssh://git@github.com:FZJ-RT/cac.git
```

### Installation via Anaconda or Miniforge

1. Ensure [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) or [Miniforge](https://github.com/conda-forge/miniforge) (recommended) on your machine.
2. Once conda environment installed, install conda-devenv: conda install conda-devenv
3. Clone cac
4. Go to cac folder
5. If you have GPU, go to "with_gpu" folder. If you do not have GPU, go to "no_gpu" folder
6. Enter in your terminal: conda devenv. This process will create a new environment called (cac)
7. Activate cac environment: conda activate cac
8. Go back to main cac folder by writing "cd .." in your terminal
8. Build cac library: python setup.py bdist_wheel
9. Install build: pip install -e . (don't forget dot!!!)
10. Enjoy the library

## Contact us

For further questions or inquiries, please contact Ryan Santoso - r.santoso@fz-juelich.de
