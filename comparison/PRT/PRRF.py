import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
import argparse


# Main PR Random Forest
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Welcome to TDT builder')
    parser.add_argument('-x', type=str, help='dataset path')
    parser.add_argument('-s', type=str, help='splitting method', default='topk3')
    parser.add_argument('-m', type=str, help='criterion method', default='mseprob')
    parser.add_argument('-t', type=int, help='number of trees', default=100)
    parser.add_argument('-l', type=float, help='min leaf percentage', default=0.1)
    parser.add_argument('-sg', type=str, help='sigma values separated by , for each fold', default=None)
    parser.add_argument('-cvs', type=int, help='cross validation min range', default=0)
    parser.add_argument('-cve', type=int, help='cross validation max range', default=10)
    parser.add_argument('-ts', type=float, help='test size in percentage', default=0.2)
    args = parser.parse_args()

    # Read the dataset
    X = pd.read_csv(args.x, header=None, index_col=None)
    X = X.values
    Y = X[:, -1]
    X = X[:, 0:X.shape[1] - 1]

    rmses = []
    sigmas = None
    if args.sg is not None:
        sigmas = [float(v) for v in args.sg.split(',')]
    else:
        sigmas = np.ones(args.cve - args.cvs)

    for ind, k in enumerate(range(args.cvs, args.cve)):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=args.ts, random_state=k)

        rf_predictions = []
        for j in range(args.t):
            print('CV=', k, ', T=', j, sep='')

            # Split the dataset with Bootstrap with replacement
            np.random.seed(args.t * k + j)
            idx = np.random.randint(len(X_train), size=len(X_train))
            X_subtree = X_train[idx, :]
            y_subtree = y_train[idx]
            sigma_Xp = np.std(X_subtree, axis=0)

            sigma_Xp = sigma_Xp * sigmas[ind]
            temp_min_smp_leaf = round(len(X_subtree) * args.l)

            temp_method = args.m
            if args.sg:
                if sigmas[ind] == 0:
                    temp_method = 'mse'
            if temp_method in ['mse', 'msepred']:
                regressor = tree.DecisionTreeRegressor(criterion=temp_method, random_state=0,
                                                       min_samples_leaf=temp_min_smp_leaf, tol=sigma_Xp)
            else:
                regressor = tree.DecisionTreeRegressor(criterion=temp_method, random_state=0, splitter=args.s,
                                                       min_samples_leaf=temp_min_smp_leaf, tol=sigma_Xp)
            regressor.fit(X_subtree, y_subtree)

            if temp_method == 'mse':
                prediction = regressor.predict(X_test)
            else:
                F = [f for f in regressor.tree_.feature if f != -2]
                for s_current_node in range(len(F)):
                    for kk in range(s_current_node + 1, len(F)):
                        if F[s_current_node] == F[kk]:
                            F[kk] = -1
                F = np.array(F)
                prediction = regressor.predict3(X_test, F=F)

            rf_predictions.append(prediction)

        prediction = np.mean(rf_predictions, axis=0)
        error = abs(y_test - prediction)
        RMSE_test = np.sqrt(np.mean(error ** 2))
        print(k, RMSE_test)
        rmses.append(RMSE_test)

    # Print the Avg RMSE and the Std
    print()
    print('Avg RMSE', np.mean(rmses))
    print('Std RMSE', np.std(rmses))
