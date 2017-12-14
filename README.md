# UNOFFICIAL code for the paper [Certifiable Distributional Robustness with Principled Adversarial Training](https://openreview.net/forum?id=Hk6kPgZA-)

UNOFFICIAL implementation of the work submitted to ICLR 2018. Work in progress, not completely sure of the correctness of the implementation at present.  

## Overview 

The code allows to train a ConvNet on MNIST using the adversarial training proposed in the paper. 

### Prerequisites

Python 2.7, Tensorflow 1.3 

### Files

Model.py: class to build and train a ConvNet.

TrainOps.py: class to train and test the model. 

run_exp.sh: to lunch an experiment.

exp_configuration: configuration file for an experiment, set here the hyperparameters. 

## How it works

Run

```
python download_and_process_mnist.py
```

to download and process MNIST dataset, then run 

```
sh run_exp.sh
```

to start the training procedure, after having set the desired hyperparameters/configurations in exp_config and run_exp.
