# Chip Analyzer and Calculator (cac)
An open-source Python library for applying data-driven and process-based machine learning methods to analyze microfluidics experiments

## Introduction
`cac` provides data-driven and process-based machine learning methods for enabling efficient and real-time analysis of microfluidics experiments. It uses [Keras](https://keras.io/) and [Tensorflow](https://www.tensorflow.org/) for the implementation of data-driven machine learning methods, and [OpenCV](https://opencv.org/) for image processing. It enables training with GPU, built on [Horovod](https://horovod.ai/). 

The process-based machine learning implementation uses [Keras](https://keras.io/) and [Tensorflow](https://www.tensorflow.org/) library. It automatically applied optimization using Bayesian Optimization with Hyperband (BOHB) algorithm from [bohb-hpo](https://github.com/goktug97/bohb-hpo.git) repository. Users can use the process-based machine learning method with minimum changes.

## Simulator for microfluidics experiments
To provide training samples for constructing machine learning models, we frequently use Lattice-Boltzmann method. The Lattice-Boltzmann code can be downloaded through this link: [lbm](git@github.com:FZJ-RT/lbm.git) and should be used in conjuction with `cac`. 

## Getting started
Please check INSTALLATION.txt for detailed steps for installing the library.

## Contact us
For further questions or inquiries, please contact Ryan Santoso - r.santoso@fz-juelich.de
