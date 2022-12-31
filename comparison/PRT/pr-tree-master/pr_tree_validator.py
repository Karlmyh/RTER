# This file can be used to re-generate the results of the probabilstic trees.
# It can also be used to run experiments on new datasets.

import sys
import csv
import pandas as pd
from sklearn import tree
import numpy as np
from sklearn.model_selection import train_test_split
import argparse


# Validator probabilistic trees
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Welcome to TDT builder')
    parser.add_argument('-x', type=str, help='x path')
    parser.add_argument('-s', type=str, help='splitting method', default='topk3')
    parser.add_argument('-m', type=str, help='criterion method', default='mseprob')
    parser.add_argument('-l', type=float, help='min leaf percentage', default=0.1)    
    parser.add_argument('-cvs', type=int, help='cross validation min/max range', default=0)
    parser.add_argument('-cve', type=int, help='cross validation max range', default=10)
    parser.add_argument('-ts', type=float, help='test size in percentage', default=0.2)
    args = parser.parse_args()

    X = pd.read_csv(args.x, header=None, index_col=None)
    X = X.values
    Y = X[:, -1]
    X = X[:, 0:X.shape[1] - 1]

    rmses = []
    
	# Sigma_u validation step, 1e-20 = Std Decision Trees
    sigma_values = [1e-20, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
	
	# 10 Cross-validation by default, can be changed by the user in the input.
    for k in range(args.cvs, args.cve):
		
		# Split the dataset into training (80%) and testing (20%) sets
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=args.ts, random_state=k)
		
		# Calculate the standard deviation (noise)
        sigma_Xp = np.std(X_train, axis=0)
        min_error = sys.float_info.max
        errors_arr = []
        regressor_arr = []
		
		# Split the training (80%) into a validation training set (~65% of the original size) 
		# and validation testing set (~15% of the original size)
		# Seed is fixed to avoid different values
        X_tr_valid, X_ts_valid, y_tr_valid, y_ts_valid = train_test_split(X_train, y_train,
                                                                          test_size=args.ts, random_state=0)
        
		# The stopping criteria requires all leaves to have at least 10% of the training size
        temp_min_smp_leaf = round(len(X_tr_valid) * args.l)

        # Validation loop
        for sigma_val in sigma_values:
			
		# Calculate the new standard deviation based on the noise modifier
            sigma_arr = sigma_Xp * sigma_val
			
			# Run the model
            regressor = tree.DecisionTreeRegressor(criterion=args.m, random_state=0, splitter=args.s,
                                                       min_samples_leaf=temp_min_smp_leaf, tol=sigma_arr)
            regressor.fit(X_tr_valid, y_tr_valid)
			
			# Test the model on the validation test set
            F = [f for f in regressor.tree_.feature if f != -2]
            for s_current_node in range(len(F)):
                for k_ind in range(s_current_node + 1, len(F)):
                    if F[s_current_node] == F[k_ind]:
                        F[k_ind] = -1
            F = np.array(F)
            prediction = regressor.predict3(X_ts_valid, F=F)
			
			# Calculate the RMSE
            error = abs(y_ts_valid - prediction)
            RMSE_test = np.sqrt(np.mean(error ** 2))
			
			# print the current CV, current sigma_modifier, and current RMSE
            print(k, sigma_val, "{:.3f}".format(RMSE_test))
            errors_arr.append(RMSE_test)
            regressor_arr.append(regressor)

        # Testing
		# Pick the best value of sigma_u (standard deviation) based on the validation
        ranking_sigma = np.argsort(errors_arr)
        best_sigma = sigma_values[ranking_sigma[0]]
        sigma_arr = sigma_Xp * best_sigma
			
		# Recalculate the stopping criteria
        temp_min_smp_leaf = round(len(X_train) * args.l)
		
		# Run the model on the training set
        regressor = tree.DecisionTreeRegressor(criterion=args.m, random_state=0, splitter=args.s,
                                                   min_samples_leaf=temp_min_smp_leaf, tol=sigma_arr)
        regressor.fit(X_train, y_train)
		
		# Run the model on the test set
        F = [f for f in regressor.tree_.feature if f != -2]
        for s_current_node in range(len(F)):
            for k_ind in range(s_current_node + 1, len(F)):
                if F[s_current_node] == F[k_ind]:
                    F[k_ind] = -1
        F = np.array(F)
        prediction = regressor.predict3(X_test, F=F)
        error = abs(y_test - prediction)
        RMSE_test = np.sqrt(np.mean(error ** 2))
		
		# Print the best RMSE
        print('Best', k, best_sigma, "{:.3f}".format(RMSE_test))
        print()
        rmses.append(RMSE_test)

	# Finally, print the Avg RMSE and the Std
    print()
    print('Avg:', "{:.3f}".format(np.mean(rmses)))
    print('Std:', "{:.3f}".format(np.std(rmses)))
