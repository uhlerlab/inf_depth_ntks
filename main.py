import numpy as np
import random
import time
import dataset
import models


def main():
    SEED = 1717
    n_train = 10000
    n_test = 10000
    prior1 = .4

    use_gpu = True

    random.seed(SEED)
    np.random.seed(SEED)

    alpha1 = np.array([1, 2])
    alpha2 = np.array([2, 1])

    #alpha1 = np.array([.5, .5, 2.5])
    #alpha2 = np.array([2.5, .1, .5])

    #alpha1 = np.array([3, 1, 2, 1, 1])
    #alpha2 = np.array([1, 2, 2, 2, 3])

    #alpha1 = np.array([1, 2, 1, 2, 1, 1, 1, 1, 1])
    #alpha2 = np.array([1, 1, 1, 2, 3, 4, 2, 2, 1])

    DIM = alpha1.shape

    bad_sample = True
    while bad_sample:
        X, y, SX = dataset.get_dirichlet_data(n_train, n_test, alpha1,
                                              alpha2, prior1)
        test = X @ X.T - np.eye(len(X))
        # Re-sample if two points are too close in training set
        if test.max() >= 1 or test.min() == 0:
            bad_sample = True
        else:
            bad_sample = False
    print("Max Angle Between Training Samples", test.max())

    X, y, SX = dataset.shuffle(X, y, SX)

    X_train = X[:n_train]
    y_train = y[:n_train]

    X_test = X[n_train:]
    SX_test = SX[n_train:]
    y_test = y[n_train:]

    y_1 = np.sum(np.where(y_train > 0, 1, 0))
    y_2 = len(y_train) - y_1
    print("Training set: ", X_train.shape, y_train.shape, y_1, y_2)
    y_1 = np.sum(np.where(y_test > 0, 1, 0))
    y_2 = len(y_test) - y_1
    print("Test set: ", X_test.shape, y_test.shape, y_1, y_2)

    bayes_acc = models.dirichlet_bayes_classifier(alpha1, alpha2, prior1,
                                                      SX_test, y_test)
    nn_acc = models.nn_classifier(X_train, y_train, X_test, y_test)
    devroye_acc = models.devroye_classifier(X_train, y_train, X_test, y_test)
    maj_acc = models.majority_vote(y_train, y_test)

    perfs = {}

    kernel_fn = 'nngp'
    # List of activations used in experiments
    act_names = ['erf', 'sine_1',
                 '2d_opt',
                 'sine_2.676', '3d_opt',  # Consistent for 3d data
                 'sine_6.13539', '5d_opt',  # Consistent for 5d data
                 'sine_13.8258', '9d_opt']  # Consistent for 9d data

    # Depths considered for experiments in paper
    depths = [200, 100, 100, 100, 100, 50, 100, 50, 100]

    for idx, act_name in enumerate(act_names):
        depth = depths[idx]
        kernel_acc = models.iterative_kernel(X_train, y_train, X_test, y_test,
                                             fn=kernel_fn, act_name=act_name,
                                             depth=depth,
                                             use_gpu=use_gpu)
        perfs[act_name] = kernel_acc

    # Uncomment to run ReLU NTK Experiments
    # Setting N to be large requires more depth to achieve majority vote
    """
    N = 100
    relu_acc = models.iterative_kernel(X_train[:N], y_train[:N],
                                       X_test, y_test,
                                       fn='ntk', act_name='relu',
                                       depth=50000,
                                       use_gpu=use_gpu)
    perfs['relu'] = relu_acc
    #"""
    print("==================================================")
    print("Summary of Results:")
    print("Bayes Acc: ", bayes_acc)
    print("1-NN Acc:", nn_acc)
    print("Devroye Acc:", devroye_acc)
    print("Maj Vote Acc: " , maj_acc)
    for k in perfs:
        print(k, perfs[k])


if __name__ == "__main__":
    main()
