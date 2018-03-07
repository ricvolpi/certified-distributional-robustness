# Unofficial code for the paper [Certifying Some Distributional Robustness with Principled Adversarial Training](https://openreview.net/forum?id=Hk6kPgZA-)
 
 ## Overview 
 
 The code allows to train a ConvNet on MNIST using the adversarial training proposed in the paper. 
 
 ### Prerequisites
 
 Python 2.7, Tensorflow 1.3 
 
 ### Files
 
 model.py: class to build and train a ConvNet.
 
 trainOps.py: class to train and test the model. 
 
 run_exp.sh: to launch an experiment.
 
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
