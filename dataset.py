import numpy as np
import random
from scipy.stats import dirichlet
from numpy.linalg import norm


def get_dirichlet_data(n_train, n_test, alpha1, alpha2, prior1=.5):

    total = n_train + n_test
    size1 = int(total * prior1)

    X1 = dirichlet.rvs(alpha1, size=size1)
    X2 = dirichlet.rvs(alpha2, size=total-size1)
    SX = np.concatenate([X1, X2], axis=0)
    X = SX / norm(SX, axis=-1).reshape(-1, 1)

    y1 = np.ones(size1)
    y2 = np.ones(total-size1) * -1
    y = np.concatenate([y1, y2]).reshape(-1, 1)
    return X, y, SX


def shuffle(X, y, angles):

    shuf = list(zip(X,y, angles))
    random.shuffle(shuf)
    X, y, angles = zip(*shuf)

    return np.array(X), np.array(y), np.array(angles)
