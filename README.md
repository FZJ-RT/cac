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

Follow the steps listed underneath to install this package.

1. Install [anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) or [Miniforge](https://github.com/conda-forge/miniforge) (recommended) on your machine.
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
