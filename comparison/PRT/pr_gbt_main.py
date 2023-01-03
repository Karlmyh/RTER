import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.model_selection import train_test_split
import argparse


# Main PR GBT
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Welcome to TDT builder')
    parser.add_argument('-x', type=str, help='dataset path')
    parser.add_argument('-s', type=str, help='splitting method', default='topv3')
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
        print('CV=', k)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=args.ts, random_state=k)
        min_samples_leaf = round(len(X_train) * args.l)

        # Presort needs to be False (for now)
        regressor = ensemble.GradientBoostingRegressor(random_state=0, n_estimators=args.t, presort=False,
                                                       min_samples_leaf=min_samples_leaf, criterion=args.m)
        if args.m == 'mse':
            regressor.fit(X_train, y_train)
        else:
            if sigmas is not None:
                regressor.fit_un(X_train, y_train, sigma_mult=sigmas[ind])
            else:
                regressor.fit_un(X_train, y_train)

        prediction = regressor.predict(X_test)

        error = abs(y_test - prediction)
        RMSE_test = np.sqrt(np.mean(error ** 2))
        print(k, RMSE_test)
        rmses.append(RMSE_test)

    # Print the Avg RMSE and the Std
    print()
    print('Avg RMSE', np.mean(rmses))
    print('Std RMSE', np.std(rmses))
