import argparse
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


## Data loader and data generation functions
def data_loader(args):
    """
    Output:
        X: the data matrix (numpy array) of size 1-by-N
        Y: the label matrix (numpy array) of size N-by-1
    """
    if args.data == "linear":
        print("Using linear")
        X, Y = data_linear()
    elif args.data == "quadratic":
        print("Using quadratic")
        X, Y = data_quadratic()
    elif args.data == "unknown":
        print("Using unknown")
        X, Y = data_unknown()
    elif args.data == "unknown_noise":
        print("Using unknown_noise")
        X, Y = data_unknown_noise()
    else:
        print("Using simple")
        X, Y = data_simple()
    return X, Y


def data_linear():
    """
    length_phi = 1
    length_Y = 2.2
    bias = 0.5
    sigma = 0.7  # noise strength
    N = 50  # number of samples
    phi = length_phi * (np.random.rand(N)-0.7)
    xi = np.random.rand(N)
    X = phi
    Y = length_Y * phi + sigma * xi + bias
    X = X.reshape(1, -1)
    Y = Y.reshape(-1, 1)
    np.savez('Linear.npz', X = X, Y = Y)
    """
    data = np.load(osp.join(args.path, 'Linear.npz'))
    X = data['X']
    Y = data['Y']
    return X, Y


def data_quadratic():
    """
    length_phi = 1
    length_Y = 2.2
    length_YY = 3.7
    bias = 0.5
    sigma = 0.3  # noise strength
    N = 50  # number of samples
    phi = length_phi * (np.random.rand(N) - 0.7)
    xi = np.random.rand(N)
    X = phi
    Y = length_Y * phi + length_YY * (phi **2) + sigma * xi + bias
    X = X.reshape(1, -1)
    Y = Y.reshape(-1, 1)
    np.savez('Quadratic.npz', X = X, Y = Y)
    """
    data = np.load(osp.join(args.path, 'Quadratic.npz'))
    X = data['X']
    Y = data['Y']
    return X, Y


def data_unknown():
    """
    length_phi = 1
    length_Y = 2.2
    length_YY = 3.7
    length_YYY = -4.9
    length_YYYY = -5.8
    length_YYYYY = 4.2
    bias = 0.5
    sigma = 0.01  # noise strength
    N = 50  # number of samples
    phi = length_phi * (np.random.rand(N) - 0.7)
    xi = np.random.rand(N)
    X = phi
    Y = length_Y * phi + length_YY * (phi **2) + length_YYY * (phi **3) + \
        length_YYYY * (phi **4) + length_YYYYY * (phi **5) + sigma * xi + bias
    YY = length_Y * phi + length_YY * (phi ** 2) + length_YYY * (phi ** 3) + \
        length_YYYY * (phi ** 4) + length_YYYYY * (phi ** 5) + (sigma+0.99) * xi + bias
    X = X.reshape(1, -1)
    Y = Y.reshape(-1, 1)
    YY = YY.reshape(-1, 1)
    np.savez('Unknown.npz', X = X, Y = Y)
    np.savez('Unknown_noise.npz', X=X, Y = YY)
    """
    data = np.load(osp.join(args.path, 'Unknown.npz'))
    X = data['X']
    Y = data['Y']
    return X, Y


def data_unknown_noise():
    data = np.load(osp.join(args.path, 'Unknown_noise.npz'))
    X = data['X']
    Y = data['Y']
    return X, Y


def data_simple():
    N = 20
    X = np.linspace(0.0, 10.0, num=N).reshape(1, N)
    Y = np.linspace(1.0, 3.0, num=N).reshape(N, 1)
    return X, Y


## Displaying the results
def display_LR(args, poly, w, b, X_train, Y_train, X_test, Y_test):

    poly = int(poly)
    X = np.concatenate((X_train, X_test), 1)
    Y = np.concatenate((Y_train, Y_test), 0)
    N = X.shape[1]
    phi = np.ones(N)
    phi[:X_train.shape[1]] = 0
    x_min = np.min(X)
    x_max = np.max(X)
    x = np.linspace(x_min-0.05, x_max+0.05, num=1000)
    XX = polynomial_transform(x.reshape(1,-1), poly)
    YY = np.matmul(w.transpose(), XX) + b
    plt.scatter(X.reshape(-1), Y.reshape(-1), c=phi, cmap=plt.cm.Spectral)
    plt.plot(x.reshape(-1), YY.reshape(-1), color='black', linewidth=3)
    if args.save:
        plt.savefig(args.data + '_' + str(poly) + '.png', format='png')
        np.savez('Results_' + args.data + '_' + str(poly) + '.npz', w=w, b=b)
    plt.show()
    plt.close()


## auto_grader
def auto_grade(w, b):
    print("In auto grader!")
    if w.ndim != 2:
        print("Wrong dimensionality of w")
    else:
        if w.shape[0] != 2 or w.shape[1] != 1:
            print("Wrong shape of w")
        else:
            if sum((w - [[2.00000000e-01], [2.77555756e-17]]) ** 2) < 10 ** -6:
                print("Correct w")
            else:
                print("Incorrect w")

    if (b - 1) ** 2 < 10 ** -6:
        print("Correct b")
    else:
        print("Incorrect b")


def linear_regression(X, Y):
    """
    Input:
        X: a D-by-N matrix (numpy array) of the input data
        Y: a N-by-1 matrix (numpy array) of the label
    Output:
        w: the linear weight vector. Please represent it as a D-by-1 matrix (numpy array).
        b: the bias (a scalar)
    Useful tool:
        1. np.matmul: for matrix-matrix multiplication
        2. the builtin "reshape" and "transpose()" functions of a numpy array
        3. np.linalg.inv: for matrix inversion
    """

    X = np.copy(X)
    D = X.shape[0] # feature dimension
    N = X.shape[1] # number of data instances

    ### Your job 1 starts here ###
    # Note that, if you want to apply the closed-form solution, you may want to append 1 to every column of X

    ### Your job 1 ends here ###

    return w, b


def polynomial_transform(X, degree_polynomial):
    """
    Input:
        X: a 1-by-N matrix (numpy array) of the input data
        degree_polynomial
    Output:
        X_out: a degree_polynomial-by-N matrix (numpy array) of the data
    Useful tool:
        1. ** degree_polynomial: get the values to the power of degree_polynomial
    """

    X = np.copy(X)
    N = X.shape[1]  # number of data instances
    X_out = np.zeros((degree_polynomial, N))
    for d in range(degree_polynomial):
        X_out[d, :] = X.reshape(-1) ** (d + 1)
    return X_out


## Main function
def main(args):

    if args.auto_grade:
        args.data = "simple"
        args.polynomial = int(2)
        args.validation = False
        args.display = False
        args.save = False

    ## Loading data
    X_original, Y_original = data_loader(args)
    # X_original: the 1-by-N data matrix (numpy array)
    # Y: the N-by-1 label vector

    ## Setup (separate to train, validation, and test)
    N = X_original.shape[1]  # number of data instances of X
    X_original_test = X_original[:, int(0.7 * N):]
    X_original_val = X_original[:, :int(0.2 * N)]
    X_original_train = X_original[:, int(0.2 * N):int(0.7 * N)]
    Y_test = Y_original[int(0.7 * N):, :]
    Y_val = Y_original[:int(0.2 * N), :]
    Y_train = Y_original[int(0.2 * N):int(0.7 * N), :]
    best_poly = 0
    cur_best_val_error = 1000000000000000

    ## Validation + Testing
    if args.validation:
        for poly in range(1, 12):
            X_train = polynomial_transform(np.copy(X_original_train),
                                           int(poly))  # the poly-by-N_train data matrix (numpy array)
            X_val = polynomial_transform(np.copy(X_original_val),
                                           int(poly))  # the poly-by-N_val data matrix (numpy array)

            # Running LR (in validation step)
            w, b = linear_regression(X_train, Y_train)
            print("w: ", w)
            print("b: ", b)

            # Evaluation (in validation step)
            train_error = np.mean((np.matmul(w.transpose(), X_train) + b - Y_train.transpose()) ** 2)
            val_error = np.mean((np.matmul(w.transpose(), X_val) + b - Y_val.transpose()) ** 2)
            print("Val Polynomial: ", int(poly))
            print("Training mean square error: ", train_error)
            print("Validation mean square error: ", val_error)

            ### Your job 2 starts here ###
            # Here you have to be able to decide which polynomial degree (i.e., poly) within
            # the for loop (i.e., "for poly in range(1, 12)") that the leads to the smallest val_error
            # You may use the variable "best_poly" and "cur_best_val_error" defined above.
            # Please record the best polynomial degree in the variable best_poly.

            ### Your job 2 ends here ###

        # Running LR (in final testing step)
        X_train = polynomial_transform(np.copy(X_original_train),
                                       int(best_poly))  # the best_poly-by-N_train data matrix (numpy array)
        X_test = polynomial_transform(np.copy(X_original_test),
                                     int(best_poly))  # the best_poly-by-N_test data matrix (numpy array)

        w, b = linear_regression(X_train, Y_train)
        print("w: ", w)
        print("b: ", b)

        # Evaluation (in final testing step)
        train_error = np.mean((np.matmul(w.transpose(), X_train) + b - Y_train.transpose()) ** 2)
        test_error = np.mean((np.matmul(w.transpose(), X_test) + b - Y_test.transpose()) ** 2)
        print("Best Polynomial: ", int(best_poly))
        print("Training mean square error: ", train_error)
        print("Test mean square error: ", test_error)

        if args.display:
            display_LR(args, int(best_poly), w, b, X_original_train, Y_train, X_original_test, Y_test)

    else:
        X_train = polynomial_transform(np.copy(X_original_train),
                                 args.polynomial) # the args.polynomial-by-N_train data matrix (numpy array)
        X_test = polynomial_transform(np.copy(X_original_test),
                                      args.polynomial)  # the args.polynomial-by-N_test data matrix (numpy array)

        # Running LR (direct)
        w, b = linear_regression(X_train, Y_train)
        print("w: ", w)
        print("b: ", b)

        # Evaluation (direct)
        train_error = np.mean((np.matmul(w.transpose(), X_train) + b - Y_train.transpose()) ** 2)
        test_error = np.mean((np.matmul(w.transpose(), X_test) + b - Y_test.transpose()) ** 2)
        print("Polynomial: ", args.polynomial)
        print("Training mean square error: ", train_error)
        print("Test mean square error: ", test_error)

        if args.display:
            display_LR(args, args.polynomial, w, b, X_original_train, Y_train, X_original_test, Y_test)

    if args.auto_grade:
        auto_grade(w, b)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running linear regression (LR)")
    parser.add_argument('--path', default="data", type=str)
    parser.add_argument('--data', default="linear", type=str)
    parser.add_argument('--polynomial', default=1, type=int)
    parser.add_argument('--validation', action='store_true', default=False)
    parser.add_argument('--display', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--auto_grade', action='store_true', default=False)
    args = parser.parse_args()
    main(args)

    # Fill in the other students you collaborate with:
    # e.g., Wei-Lun Chao, chao.209
    #
    #
    #