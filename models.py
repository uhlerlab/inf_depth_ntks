import numpy as np
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from numpy.linalg import norm, inv, solve, svd
import time
from tqdm import tqdm
import torch
import gc
from scipy.stats import dirichlet


def get_acc(out, y):
    preds = np.where(out < 0, -1, 1).reshape(-1,1)
    acc = np.mean(preds == y)
    return acc


def get_multiclass_acc(out, y):
    print(out.shape, y.shape)
    preds = np.argmax(out, axis=1)
    acc = np.mean(preds == np.argmax(y, axis=1))
    return acc


def dirichlet_bayes_classifier(alpha1, alpha2, prior1,
                               SX, y):

    pdf1 = dirichlet.pdf(SX.T, alpha1) * prior1
    pdf2 = dirichlet.pdf(SX.T, alpha2) * (1-prior1)
    pred = np.where(pdf1 > pdf2, 1, -1)

    bayes_acc = get_acc(pred, y)
    print("Bayes Opt Acc: ", bayes_acc)
    return bayes_acc


def majority_vote(y_train, y_test):
    y_1 = np.sum(np.where(y_train > 0, 1, 0))
    y_2 = len(y_train) - y_1
    if y_1 > y_2:
        preds = np.ones(y_test.shape)
    else:
        preds = np.ones(y_test.shape)*-1
    maj_acc = get_acc(preds, y_test)
    print("Maj Vote Acc: ", maj_acc)
    return maj_acc


def nn_classifier(X_train, y_train, X_test, y_test):
    model = KNeighborsRegressor(n_neighbors=1, n_jobs=16)
    model.fit(X_train, y_train)
    out = model.predict(X_test)

    nn_test_acc = get_acc(out, y_test)
    print("1NN Test Acc: ", nn_test_acc)
    return nn_test_acc


def devroye_classifier(X_train, y_train, X_test, y_test,
                        multiclass=False):
    angles = X_train @ X_test.T

    n, d = X_train.shape
    d = d - 1
    alpha = d / 2

    angles = (2 - 2*angles)**alpha
    angles = 1 / angles

    naive_preds = (y_train.T @ angles)

    if not multiclass:
        test_acc = get_acc(naive_preds.reshape(-1, 1), y_test)
    else:
        test_acc = get_multiclass_acc(naive_preds.T, y_test)
    print("Devroye Classifier Acc: ", test_acc, " Dimension corrected by subtracting 1")
    return test_acc


def dual_act(angles, act_name):
    if 'sine' in act_name:
        a = eval(act_name.strip().split("_")[-1])
        return torch.sinh(a * angles) / np.sinh(a)
    elif act_name == 'erf':
        return 1 / np.arcsin(2/3) * torch.arcsin(2 * angles / 3)
    elif act_name == 'relu':
        return 1/np.pi * (angles * (np.pi - torch.arccos(angles)) \
                          + torch.sqrt(1 - torch.pow(angles, 2)))
    elif act_name == '2d_opt':
        return angles**7/2 + angles/2
    elif act_name == '3d_opt':
        return .5 * angles**3 + .5 * angles
    elif act_name == '5d_opt':
        return angles**3/4 + .5 * angles**2 + 1/4 * angles
    elif act_name == '9d_opt':
        return angles**3/16 + 7/8 * angles**2 + 1/16 * angles
    elif 'hermite' in act_name:
        vals = act_name.strip().split("_")
        degree = eval(vals[-1])
        return angles**degree
    else:
        print("ERROR: Activation name not found")


def d_dual_act(angles, act_name):
    if act_name == "sine":
        return 2 * np.exp(1) / (np.exp(2) - 1) * torch.cosh(angles)
    elif act_name == 'erf':
        return 1 / np.arcsin(2/3) * 1/torch.sqrt(1 - 4 * angles**2 / 9) * 2/3
    elif act_name == 'relu':
        return 1/np.pi * (np.pi - torch.arccos(angles))
    elif act_name == '2d_opt':
        return 7 * angles**6/2 + 1/2
    elif act_name == '3d_opt':
        return 3 * .5 * angles**2 + .5
    elif act_name == '5d_opt':
        return 3/4 * angles**2 + angles + 1/4
    elif act_name == '9d_opt':
        return 3/16 * angles**2 + 7/4 * angles + 1/16
    elif 'hermite' in act_name:
        vals = act_name.strip().split("_")
        degree = eval(vals[-1])
        return degree * angles**(degree - 1)
    else:
        print("ERROR: Activation name not found")


def nngp_kernel(X1, X2, depth=1, act_name='erf', use_gpu=False):
    # Assuming X1 = num samples x features
    angles = X1 @ X2.T
    angles = np.clip(angles, -1, 1)
    angles = torch.from_numpy(angles)
    if use_gpu:
        angles = angles.cuda()

    for i in tqdm(range(depth)):
        angles = dual_act(angles, act_name)
        angles = torch.clip(angles, -1, 1)

    torch.cuda.empty_cache()
    angles = angles.cpu().numpy()

    return angles


def ntk_kernel(X1, X2, depth=1, act_name='erf', use_gpu=False):
    angles = X1 @ X2.T
    angles = np.clip(angles, -1, 1)
    K = np.copy(angles)
    angles = torch.from_numpy(angles)
    K = torch.from_numpy(K)
    if use_gpu:
        angles = angles.cuda()
        K = K.cuda()
    for i in tqdm(range(depth)):
        angles_ = dual_act(angles, act_name)
        K = angles_ + K * d_dual_act(angles, act_name)
        angles = angles_
    del angles
    torch.cuda.empty_cache()
    gc.collect()
    K = K.cpu().numpy()
    return K


def iterative_kernel(X_train, y_train, X_test, y_test,
                     fn='nngp', depth=1, act_name='sine',
                     multiclass=False, use_gpu=False):
    name = fn + "_" + act_name + "_" + str(depth)
    start = time.time()
    if fn == 'nngp':
        K_train = nngp_kernel(X_train, X_train, depth=depth, act_name=act_name,
                              use_gpu=use_gpu)
    else:
        K_train = ntk_kernel(X_train, X_train, depth=depth, act_name=act_name,
                             use_gpu=use_gpu)
    end = time.time()
    print("Time for K_train: ", end - start)
    start = time.time()
    if fn == 'nngp':
        K_test = nngp_kernel(X_train, X_test, depth=depth, act_name=act_name,
                             use_gpu=use_gpu)
    else:
        K_test = ntk_kernel(X_train, X_test, depth=depth, act_name=act_name,
                            use_gpu=use_gpu)
    end = time.time()
    print("Time for K_test: ", end - start)
    test_acc = solve_kr(K_train, y_train, K_test, y_test, name, multiclass=multiclass,
                        fn=fn)
    return test_acc


def solve_kr(K_train, y_train, K_test, y_test, name, multiclass=False, fn='nngp'):
    # Ensure NNGP has 1s on diagonal
    if fn == 'nngp':
        for i in range(len(K_train)):
            K_train[i, i] = 1.

    sol = solve(K_train, y_train).T
    out = sol @ K_test
    train_out = sol @ K_train

    if not multiclass:
        train_acc = get_acc(train_out, y_train)
    else:
        train_acc = get_multiclass_acc(train_out.T, y_train)
    print(name + " Train Acc: ", train_acc)
    if not multiclass:
        test_acc = get_acc(out, y_test)
    else:
        test_acc = get_multiclass_acc(out.T, y_test)
    print(name + " Test Acc: ", test_acc)

    # Used to check whether there are numerical issues
    #print("KERNEL TRAIN MATRIX shape and elementwise bounds: ")
    #print("MAX OFF DIAGONAL: ", (K_train - np.eye(len(K_train))).max())
    #print(K_train.max(), K_train.min(), K_train.shape)
    #print("K Test Bounds: ", K_test.max(), K_test.min(), K_test.shape)
    return train_acc, test_acc
