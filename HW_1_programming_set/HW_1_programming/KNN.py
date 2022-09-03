import numpy as np
import math
import argparse
import os
import os.path as osp


## Data loader and data generation functions
def data_loader(args):
    """
    Output:
        X_train: the data matrix (numpy array) of size D-by-N_train
        Y_train: the label matrix (numpy array) of size N_train-by-1
        X_test: the data matrix (numpy array) of size D-by-N_test
        Y_test: the label matrix (numpy array) of size N_test-by-1
    """
    print("Using MNIST")
    X_train, X_test, Y_train, Y_test = data_MNIST(args)

    return  X_train, X_test, Y_train, Y_test


def data_MNIST(args):
    X = np.loadtxt(osp.join(args.path, "mnist_test.csv"), delimiter=",")
    X = X.astype('float64')
    Y = X[:, 0]
    X_1st = X[Y == 1, 1:].transpose()
    X_2nd = X[Y == 8, 1:].transpose()
    N_1st = X_1st.shape[1]
    N_2nd = X_2nd.shape[1]
    Y_1st = -1 * np.ones(N_1st).reshape(-1,1)
    Y_2nd = 1 * np.ones(N_2nd).reshape(-1,1)
    X_1st_test = X_1st[:, int(0.7 * N_1st):]
    X_1st_train = X_1st[:, :int(0.7 * N_1st)]
    X_2nd_test = X_2nd[:, int(0.7 * N_2nd):]
    X_2nd_train = X_2nd[:, :int(0.7 * N_2nd)]
    Y_1st_test = Y_1st[int(0.7 * N_1st):, :]
    Y_1st_train = Y_1st[:int(0.7 * N_1st), :]
    Y_2nd_test = Y_2nd[int(0.7 * N_2nd):, :]
    Y_2nd_train = Y_2nd[:int(0.7 * N_2nd), :]

    return np.concatenate((X_1st_train, X_2nd_train), 1), np.concatenate((X_1st_test, X_2nd_test), 1),\
           np.concatenate((Y_1st_train, Y_2nd_train), 0), np.concatenate((Y_1st_test, Y_2nd_test), 0)


def distance(x_test, x_train, dis_metric):
    """
    Input:
        x_test: a D-by-1 test data instance
        x_train: a D-by-1 training data instance
        dis_metric: a string = "L1", "L2", or "cosine"
    Output:
        dist: a distance value
    """
    x_test = np.copy(x_test)
    x_train = np.copy(x_train)

    ### Your job 1 starts here ###
    if dis_metric == "L1":

    elif dis_metric == "L2":

    elif dis_metric == "cosine":

    ### Your job 1 ends here ###
    return dist


def KNN(x_test, X, Y, K, dis_metric):
    """
    Input:
        x_test: a D-by-1 test data instance
        X: a D-by-N matrix (numpy array) of the data
        Y: a N-by-1 matrix (numpy array) of the label
        K: the number of neighbors
        dis_metric: a string = "L1", "L2", or "cosine"
    Output:
        y_predict: the predict label
    Useful tool:
        1. np.matmul: for matrix-matrix multiplication
        2. the builtin "reshape" and "transpose()" functions of a numpy array
    """

    N = X.shape[1]
    label_set = np.unique(Y)

    ### Your job 2 starts here ###

    ### Your job 2 ends here ###
    return y_predict


def main(args):
    np.random.permutation(10)
    X_train, X_test, Y_train, Y_test = data_loader(args)
    print("dimension, number of training data instances: ", X_train.shape)
    print("dimension, number of test data instances: ", X_test.shape)
    N = X_train.shape[1]

    ### Your job 3 starts here ###
    # Please implement leave one out cross validation to select the best:
    # best_dis_metric from dis_metric_set = ["L1", "L2", "cosine"]
    # best_K from K_set = [1, 3, 5, 7, 9, 11, 13, 15, 17]

    dis_metric_set = ["L1", "L2", "cosine"]
    K_set = [1, 3, 5, 7, 9, 11, 13, 15, 17]
    val_acc = np.zeros((3, 9))

    for i in range(3):
        print(i)
        for j in range(9):
            print(j)
            for r in range(N):


    ### Your job 3 ends here ###

    print("Best dis_metric: ", best_dis_metric)
    print("Best K: ", best_K)
    print("Best Leave-one-out accuracy: ", best_val_accuracy)
    print("Test accuracy: ", test_accuracy)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running KNN")
    parser.add_argument('--path', default="data", type=str)
    args = parser.parse_args()
    main(args)

    # Fill in the other students you collaborate with:
    # e.g., Wei-Lun Chao, chao.209
    #
    #
    #
