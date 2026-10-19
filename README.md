# Chip Analyzer and Calculator (cac)
An open-source Python library for applying data-driven and process-based machine learning methods to analyze microfluidics experiments

## Introduction
`cac` provides data-driven and process-based machine learning methods for enabling efficient and real-time analysis of microfluidics experiments. It uses [Keras](https://keras.io/) and [Tensorflow](https://www.tensorflow.org/) for the implementation of data-driven machine learning methods, and [OpenCV](https://opencv.org/) for image processing. It enables training with GPU, built on [Horovod](https://horovod.ai/). 

The process-based machine learning implementation uses [Keras](https://keras.io/) and [Tensorflow](https://www.tensorflow.org/) library. It automatically applied optimization using Bayesian Optimization with Hyperband (BOHB) algorithm from [bohb-hpo](https://github.com/goktug97/bohb-hpo.git) repository. Users can use the process-based machine learning method with minimum changes.

## Simulator for microfluidics experiments
To provide training samples for constructing machine learning models, we frequently use Lattice-Boltzmann method. The Lattice-Boltzmann code can be downloaded through this link: [lbm](https://github.com/FZJ-RT/lbm.git) and should be used in conjuction with `cac`. 

## Getting started
Please check INSTALLATION.txt for detailed steps for installing the library.

## Tutorials
The tutorials are located in the `examples` folder.

| Tutorial | Content | Functions used |
|---|---|---|
| `tutorial_1` | Image segmentation of microfluidics images using U-Net | `utilities.read_single_image`<br>`utilities.read_single_label`<br>`utilities.read_images_from_folder`<br>`utilities.read_labels_from_folder`<br>`utilities.create_set`<br>`model.create_vanilla_unet`<br>`trainer.train_with_generator`<br>`trainer.calculate_image_metric`<br>`trainer.calculate_jaccard_score` |
| `tutorial_2` | Gas phase distribution in porous media using saturation-conditioned U-Net | `model.create_tfunet`<br>`trainer.train_tfunet`<br>`trainer.calculate_image_metric_tfunet` |
| `tutorial_3` | Non-intrusive reduced basis (NI-RB) surrogate model for a 2D spatio-temporal problem | `trainer.NIRB`<br>`NIRB.define_set`<br>`NIRB.reduce_output`<br>`NIRB.execute`<br>`model.online_nirb`<br>`online_nirb.load_existing_model`<br>`online_nirb.load_output_basis`<br>`online_nirb.load_scaling_factor`<br>`online_nirb.predict` |
| `tutorial_4` | Image reduction using a convolutional autoencoder for user image datasets | `model.create_cae2D`<br>`trainer.train_cae2D`<br>`trainer.calculate_cae2D_metric`<br>`trainer.calculate_cae2D_jaccard`<br>`model.load_existing_cae2D` |

## Contact us
For further questions or inquiries, please contact Ryan Santoso - r.santoso@fz-juelich.de
