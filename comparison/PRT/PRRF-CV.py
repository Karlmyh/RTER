# This file can be used to re-generate the results of the probabilstic trees.
# It can also be used to run experiments on new datasets.

import sys
import csv
import pandas as pd
from sklearn import tree, ensemble
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import argparse
from time import time
import os







data_file_dir = "../../data/real_data_cleaned/"

data_file_name_seq = ["bodyfat_scale.csv"]


log_file_dir = "../../results/realdata_forest/"


for data_file_name in data_file_name_seq:
    # load dataset
    data_name = os.path.splitext(data_file_name)[0]
    data_file_path = os.path.join(data_file_dir, data_file_name)

    X = pd.read_csv(data_file_path, header=None, index_col=None)
    X = X.values
    Y = X[:, 0]
    X = X[:, 1:]


    # Sigma_u validation step, 1e-20 = Std Decision Trees
    sigma_values = [0.05,0.1,0.2, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5,10]
    min_leaf_percentage_values = [0.02,0.05,0.1,0.2]


    
    for k in range(5):

       
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.7, random_state=k+10)

        
        sigma_Xp = np.std(X_train, axis=0)
        
        parameters = {"splitter":["topk3","topk5"], "criterion":["mseprob"],
                     "min_samples_leaf":[round(len(X_train) * min_leaf_percentage) for min_leaf_percentage in min_leaf_percentage_values],
                     "tol":[sigma_Xp * sigma_val for sigma_val in sigma_values],
                      "n_estimators":[100,200,300],
                     "max_features":[0.5,1]
                     }
        
        
        cv_model_PRRF = GridSearchCV(estimator=ensemble.RandomForestRegressor(),param_grid=parameters, cv=5, n_jobs=-1)
        cv_model_PRRF.fit(X_train, y_train)
        
        time_start = time()
        model_PRRF = cv_model_PRRF.best_estimator_
        
        best_n_estimators = model_PRRF.best_params_["n_estimator"]
        
        prediction_RF = np,zeros(y_test.shape[0])
        
        for single_tree in model_PRRF.estimators_:
        
            F = [f for f in single_tree.feature if f != -2]
            for s_current_node in range(len(F)):
                for k_ind in range(s_current_node + 1, len(F)):
                    if F[s_current_node] == F[k_ind]:
                        F[k_ind] = -1
            F = np.array(F)
            prediction_RF += single_tree.predict3(X_test, F = F)
            
        error = abs(y_test - prediction_RF/best_n_estimators)
        MSE_test = np.mean(error ** 2)
        time_end = time()
        
   
        
        log_file_name = "{}.csv".format("PRRF")
        log_file_path = os.path.join(log_file_dir, log_file_name)
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{}\n".format(data_name,
                                          MSE_test, time_end-time_start,
                                          k)
            f.writelines(logs)


