# Code for Consistency with Infinitely Wide and Deep Networks

## Dependencies
The code depends on the following standard libraries: 

1. numpy - 1.21.6 
2. scikit-learn - 1.0.2 
3. scipy - 1.7.3
4. pytorch - 1.11.0  

We use a GPU enabled version of PyTorch for our experiments, but all code is runnable without GPUs (just set use_gpu = False in main.py).

## File Description

1. main.py - contains example code for training deep NTKs using a variety of activation functions to classify data distributed according to a normalized Dirichlet distribution.  Uncommenting lines 90-98 includes comparison with deep NTKs with ReLU activation, but we note this computation will require increasing depth past 50000 for greater than 1000 samples.   Dirichlet distribution parameters are encoded with variables alpha1 and alpha2 and other settings for higher dimensional data are provided as comments in lines 22-29.

2. dataset.py - contains helper functions to sample data from Dirichlet distributions, which are then normalized to the unit sphere. 

3. models.py - code for implementing Neural Network Gaussian Processes (NNGPs) and NTKs for networks with custom activations considered in our work, the majority vote classifier, the nearest neighbor classifier, the Hilbert estimate for data distributed on the unit sphere, the Bayes optimal classifier given Dirichlet distribution parameters.  
