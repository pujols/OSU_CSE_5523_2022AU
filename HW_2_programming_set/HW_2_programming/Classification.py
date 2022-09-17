"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.
"""

import argparse
import os
import os.path as osp
import numpy as np
import json
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


## Data loader and data generation functions
def data_loader(args):
    """
    Output:
        X_train: the data matrix (numpy array) of size D-by-N_train
        Y_train: the label matrix (numpy array) of size N_train-by-1
        X_val: the data matrix (numpy array) of size D-by-N_val
        Y_val: the label matrix (numpy array) of size N_val-by-1
        X_test: the data matrix (numpy array) of size D-by-N_test
        Y_test: the label matrix (numpy array) of size N_test-by-1
    """
    if args.data == "linear":
        print("Using linear")
        X_train, Y_train, X_val, Y_val, X_test, Y_test = data_linear(args.feature)
    elif args.data == "noisy_linear":
        print("Using noisy linear")
        X_train, Y_train, X_val, Y_val, X_test, Y_test = data_noisy_linear(args.feature)
    elif args.data == "quadratic":
        print("Using quadratic")
        X_train, Y_train, X_val, Y_val, X_test, Y_test = data_quadratic(args.feature)
    elif args.data == "mnist":
        print("Using mnist")
        X_train, Y_train, X_val, Y_val, X_test, Y_test = data_mnist()

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def data_linear(feature):
    """
    N = 3000  # number of samples
    X = np.random.uniform(-1, 1, (2, N))
    Y = np.zeros((N, 1))
    Y[np.matmul(X.transpose(), np.ones((2,1))) > 0.2] = 1.0
    Y[np.matmul(X.transpose(), np.ones((2,1))) < -0.2] = -1.0
    X = X[:, Y.reshape(-1)!=0]
    Y = Y[Y!=0].reshape(-1, 1)
    N = X.shape[1]
    X += np.random.uniform(-0.1, 0.1, (2, N))
    print(np.sum(Y))
    print(N)
    X_train = X[:, :600]
    Y_train = Y[:600, :]
    X_val = X[:, 600:800]
    Y_val = Y[600:800, :]
    X_test = X[:, 800:1000]
    Y_test = Y[800:1000, :]
    np.savez(osp.join(args.path, 'Linear.npz'), X_train = X_train, Y_train = Y_train,
             X_val = X_val, Y_val = Y_val, X_test = X_test, Y_test = Y_test)
    """
    data = np.load(osp.join(args.path, 'Linear.npz'))
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_val = data['X_val']
    Y_val = data['Y_val']
    X_test = data['X_test']
    Y_test = data['Y_test']

    if feature == "quadratic":
        X_train = quadratic_transform(X_train)
        X_val = quadratic_transform(X_val)
        X_test = quadratic_transform(X_test)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def data_noisy_linear(feature):
    """
    N = 3000  # number of samples
    X = np.random.uniform(-1, 1, (2, N))
    Y = np.zeros((N, 1))
    Y[np.matmul(X.transpose(), np.ones((2,1))) > 0.0] = 1.0
    Y[np.matmul(X.transpose(), np.ones((2,1))) < -0.0] = -1.0
    X = X[:, Y.reshape(-1)!=0]
    Y = Y[Y!=0].reshape(-1, 1)
    N = X.shape[1]
    print(np.sum(Y))
    print(N)
    X_train = X[:, :60] + np.random.uniform(-1.0, 1.0, (2, 60))
    Y_train = Y[:60, :]
    X_val = X[:, 60:80]
    Y_val = Y[60:80, :]
    X_test = X[:, 80:100]
    Y_test = Y[80:100, :]
    np.savez(osp.join(args.path, 'Noisy_Linear.npz'), X_train = X_train, Y_train = Y_train,
             X_val = X_val, Y_val = Y_val, X_test = X_test, Y_test = Y_test)
    """
    data = np.load(osp.join(args.path, 'Noisy_Linear.npz'))
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_val = data['X_val']
    Y_val = data['Y_val']
    X_test = data['X_test']
    Y_test = data['Y_test']

    if feature == "quadratic":
        X_train = quadratic_transform(X_train)
        X_val = quadratic_transform(X_val)
        X_test = quadratic_transform(X_test)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def data_quadratic(feature):
    """
    N = 3000  # number of samples
    X = np.random.uniform(-1, 1, (2, N))
    Y = np.zeros((N, 1))
    Y[(np.sum(X**2, 0) * np.pi) > 2.2] = 1.0
    Y[(np.sum(X ** 2, 0) * np.pi) < 1.8] = -1.0
    X = X[:, Y.reshape(-1) != 0]
    Y = Y[Y != 0].reshape(-1, 1)
    N = X.shape[1]
    X += np.random.uniform(-0.1, 0.1, (2, N))
    print(np.sum(Y))
    print(N)
    X_train = X[:, :600]
    Y_train = Y[:600, :]
    X_val = X[:, 600:800]
    Y_val = Y[600:800, :]
    X_test = X[:, 800:1000]
    Y_test = Y[800:1000, :]
    np.savez(osp.join(args.path, 'Quadratic.npz'), X_train = X_train, Y_train = Y_train,
             X_val = X_val, Y_val = Y_val, X_test = X_test, Y_test = Y_test)
    """
    data = np.load(osp.join(args.path, 'Quadratic.npz'))
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_val = data['X_val']
    Y_val = data['Y_val']
    X_test = data['X_test']
    Y_test = data['Y_test']

    if feature == "quadratic":
        X_train = quadratic_transform(X_train)
        X_val = quadratic_transform(X_val)
        X_test = quadratic_transform(X_test)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def data_mnist():
    X = np.loadtxt(osp.join(args.path, "mnist_test.csv"), delimiter=",")
    X = X.astype('float64').transpose()
    N = X.shape[1]

    """
    np.random.seed(1)
    permutation = np.random.permutation(N)
    np.savez(osp.join(args.path, 'permutation.npz'), permutation=permutation)
    """

    data = np.load(osp.join(args.path, 'permutation.npz'))
    permutation = data['permutation']

    X = X[:, permutation]
    Y = X[0, :].reshape(-1,1)
    X = X[1:, :]
    Y[Y < 5] = -1.0
    Y[Y >= 5] = 1.0
    X_train = X[:, :int(0.6 * N)]
    Y_train = Y[:int(0.6 * N), :]
    X_val = X[:, int(0.6 * N):int(0.8 * N)]
    Y_val = Y[int(0.6 * N):int(0.8 * N), :]
    X_test = X[:, int(0.8 * N):]
    Y_test = Y[int(0.8 * N):, :]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def quadratic_transform(X):
    D, N = X.shape
    X_new = np.zeros((5, N))
    X_new[:2, :] = X
    X_new[2, :] = X[0, :] ** 2
    X_new[3, :] = X[1, :] ** 2
    X_new[4, :] = X[0, :] * X[1, :]
    return X_new

def display_data(X, Y):
    phi = Y.reshape(-1)
    plt.scatter(X[0, :], X[1, :], c=phi, cmap=plt.cm.Spectral)
    # plt.savefig('data.png', format='png')
    plt.show()
    plt.close()


##### Utility #####
def inti_parameter(X):
    # You may adjust this
    D, N = X.shape
    w = np.zeros((D, 1))
    b = 0.0
    return w, b

def linear_model_accuracy(X, Y, w, b):
    """
    Input:
        X: a D-by-N matrix (numpy array) of the input data
        Y: a N-by-1 matrix (numpy array) of the label (labels are either +1 or -1)
        w: the weight vector of a linear model, which is a D-by-1 matrix (numpy array)
        b: the bias of of a linear model (a scalar)
    Output:
        accuracy: a scalar between 0 and 1
    """
    Y_hat = np.sign(np.matmul(X.transpose(), w) + b)
    correct = (Y_hat == Y)
    return float(sum(correct)) / len(correct)

def GDA_linear_model_accuracy(X, Y, mu_p, mu_n, Sigma, p):
    """
    Input:
        X: a D-by-N matrix (numpy array) of the input data
        Y: a N-by-1 matrix (numpy array) of the label (labels are either +1 or -1)
        mu_p: the mean vector for class + 1: a D-by-1 matrix (numpy array)
        mu_n: the mean vector for class - 1: a D-by-1 matrix (numpy array)
        Sigma: the shared covariance matrix for both class: a D-by-D matrix (numpy array)
        p: the prior probability of class + 1
    Output:
        accuracy: a scalar between 0 and 1
    Useful tool:
        for Sigma to be invertible, you may add a very small diagonal matrix "10^-8 * identity matrix" to it
    """
    Sigma += (10 ** -8) * np.identity(X.shape[0])
    ### Your job Q5.2 starts here ###

    ### Your job Q5.2 ends here ###
    return accuracy

def GDA_nonlinear_model_accuracy(X, Y, mu_p, mu_n, Sigma_p, Sigma_n, p):
    """
    Input:
        X: a D-by-N matrix (numpy array) of the input data
        Y: a N-by-1 matrix (numpy array) of the label (labels are either +1 or -1)
        mu_p: the mean vector for class + 1: a D-by-1 matrix (numpy array)
        mu_n: the mean vector for class - 1: a D-by-1 matrix (numpy array)
        Sigma_p: the shared covariance matrix for class + 1: a D-by-D matrix (numpy array)
        Sigma_n: the shared covariance matrix for class - 1: a D-by-D matrix (numpy array)
        p: the prior probability of class + 1
    Output:
        accuracy: a scalar between 0 and 1
    Useful tool:
        for Sigma_p or Sigma_n to be invertible, you may add a diagonal matrix "10^-8 * identity matrix" to it
    """
    Sigma_p += (10 ** -8) * np.identity(X.shape[0])
    Sigma_n += (10 ** -8) * np.identity(X.shape[0])
    ### Your job Q7.2 starts here ###

    ### Your job Q7.2 ends here ###
    return accuracy


##### Algorithms #####
def pocket_train(X, Y, w, b, step_size=0.01, max_iterations=100):
    """
    Input:
        X: a D-by-N matrix (numpy array) of the input data
        Y: a N-by-1 matrix (numpy array) of the label (labels are either +1 or -1)
        w: the initialized weight vector of a linear model, which is a D-by-1 matrix (numpy array)
        b: the initialized bias of a linear model (a scalar)
        step_size: step size (or learning rate) used in the perception algorithm update
        max_iterations: the maximum number of iterations to update parameters
    Output:
        w: the weight vector. Please represent it as a D-by-1 matrix (numpy array)
        b: the bias (a scalar)
    Useful tool:
        1. np.matmul: for matrix-matrix multiplication
        2. the builtin "reshape" and "transpose()" functions of a numpy array
        3. np.sign: for sign
    """

    np.random.seed(1)
    D = X.shape[0]  # feature dimension
    N = X.shape[1]  # number of data instances
    tilde_X = np.concatenate((X, np.ones((1, N))), 0)  # add 1 to the end of each data instance
    tilde_w = np.zeros((D + 1, 1))
    best_tilde_w = np.zeros((D + 1, 1))  # for recording the best tilde_w so far
    best_training_accuracy = 0 # for recording the best training accuracy so far
    tilde_w[:D, :] = w # initialization
    tilde_w[D, 0] = b # initialization

    for t in range(max_iterations):
        permutation = np.random.permutation(N) # permute your data
        tilde_X = tilde_X[:, permutation]
        X = X[:, permutation]
        Y = Y[permutation, :]
        ### Your job Q2 starts here ###
        for n in range(N):

        ### Your job Q2 ends here ###

    w = best_tilde_w[:D, :]
    b = best_tilde_w[D, 0]

    return w, b

def GDA_linear_train(X, Y):
    """
    Input:
        X: a D-by-N matrix (numpy array) of the input data
        Y: a N-by-1 matrix (numpy array) of the label (labels are either +1 or -1)
    Output:
        mu_p: the mean vector for class + 1. Please represent it as a D-by-1 matrix (numpy array)
        mu_n: the mean vector for class - 1. Please represent it as a D-by-1 matrix (numpy array)
        Sigma: the shared covariance matrix for both class. Please represent it as a D-by-D matrix (numpy array)
        p: the prior probability of class + 1
    Useful tool:
        1. np.matmul: for matrix-matrix multiplication
        2. the builtin "reshape" and "transpose()" functions of a numpy array
    """

    D = X.shape[0] # feature dimension
    N = X.shape[1] # number of data instances

    ### Your job Q5.1 starts here ###

    ### Your job Q5.1 ends here ###

    return mu_p, mu_n, Sigma, p

def GDA_nonlinear_train(X, Y):
    """
    Input:
        X: a D-by-N matrix (numpy array) of the input data
        Y: a N-by-1 matrix (numpy array) of the label (labels are either +1 or -1)
    Output:
        mu_p: the mean vector for class + 1. Please represent it as a D-by-1 matrix (numpy array)
        mu_n: the mean vector for class - 1. Please represent it as a D-by-1 matrix (numpy array)
        Sigma_p: the shared covariance matrix for class + 1. Please represent it as a D-by-D matrix (numpy array)
        Sigma_n: the shared covariance matrix for class - 1. Please represent it as a D-by-D matrix (numpy array)
        p: the prior probability of class + 1
    Useful tool:
        1. np.matmul: for matrix-matrix multiplication
        2. the builtin "reshape" and "transpose()" functions of a numpy array
    """

    D = X.shape[0] # feature dimension
    N = X.shape[1] # number of data instances

    ### Your job Q7.1 starts here ###

    ### Your job Q7.1 ends here ###

    return mu_p, mu_n, Sigma_p, Sigma_n, p


## Main function
def main(args):

    ### Loading data
    # X_: the D-by-N data matrix (numpy array): every column is a data instance
    # Y_: the N-by-1 label vector
    X_train, Y_train, X_val, Y_val, X_test, Y_test = data_loader(args)
    print("size of training data instances: ", X_train.shape)
    print("size of validation data instances: ", X_val.shape)
    print("size of test data instances: ", X_test.shape)
    print("size of training data labels: ", Y_train.shape)
    print("size of validation data labels: ", Y_val.shape)
    print("size of test data labels: ", Y_test.shape)

    """
    display_data(X_train, Y_train)
    display_data(X_val, Y_val)
    display_data(X_test, Y_test)
    """

    ### Initialize the parameters
    w, b = inti_parameter(X_train)
    reg_coeff = float(args.reg_coeff)
    step_size = float(args.step_size)
    max_iterations = int(args.max_iterations)

    # ----------------Pocket-----------------------------------
    if args.algorithm == "pocket":
        print("Running pocket")
        w, b = pocket_train(X_train, Y_train, w, b, step_size=step_size, max_iterations=max_iterations)
        training_accuracy = linear_model_accuracy(X_train, Y_train, w, b)
        validation_accuracy = linear_model_accuracy(X_val, Y_val, w, b)
        test_accuracy = linear_model_accuracy(X_test, Y_test, w, b)
    # ----------------GDA_linear-----------------------
    elif args.algorithm == "GDA_linear":
        print("Running GDA_linear")
        mu_p, mu_n, Sigma, p = GDA_linear_train(X_train, Y_train)
        training_accuracy = GDA_linear_model_accuracy(X_train, Y_train, mu_p, mu_n, Sigma, p)
        validation_accuracy = GDA_linear_model_accuracy(X_val, Y_val, mu_p, mu_n, Sigma, p)
        test_accuracy = GDA_linear_model_accuracy(X_test, Y_test, mu_p, mu_n, Sigma, p)
    # ----------------GDA_nonlinear-----------------------
    elif args.algorithm == "GDA_nonlinear":
        print("Running GDA_nonlinear")
        mu_p, mu_n, Sigma_p, Simga_n, p = GDA_nonlinear_train(X_train, Y_train)
        training_accuracy = GDA_nonlinear_model_accuracy(X_train, Y_train, mu_p, mu_n, Sigma_p, Simga_n, p)
        validation_accuracy = GDA_nonlinear_model_accuracy(X_val, Y_val, mu_p, mu_n, Sigma_p, Simga_n, p)
        test_accuracy = GDA_nonlinear_model_accuracy(X_test, Y_test, mu_p, mu_n, Sigma_p, Simga_n, p)

    print("Accuracy: training set: ", training_accuracy)
    print("Accuracy: validation set: ", validation_accuracy)
    print("Accuracy: test set: ", test_accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running classifiers")
    parser.add_argument('--path', default="data", type=str)
    parser.add_argument('--algorithm', default="logistic", type=str)
    parser.add_argument('--feature', default="linear", type=str)
    parser.add_argument('--data', default="linear", type=str)
    parser.add_argument('--reg_coeff', default=0.1, type=float)
    parser.add_argument('--step_size', default=0.1, type=float)
    parser.add_argument('--max_iterations', default=500, type=int)
    args = parser.parse_args()
    main(args)

    # Fill in the other students you collaborate with:
    # e.g., Wei-Lun Chao, chao.209
    #
    #
    #