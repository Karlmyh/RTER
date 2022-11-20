import os
import numpy as np 
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from RTER import RegressionTree
from comparison.ensemble import RegressionTreeBoosting, RegressionTreeEnsemble
from comparison.EKNN import EKNN
from sklearn.metrics import mean_squared_error as MSE

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from time import time


data_file_dir = "./data/real_data_cleaned/"

data_file_name_seq = ["abalone.csv", "bodyfat_scale.csv"]

log_file_dir = "./results/realdata/"


for data_file_name in data_file_name_seq:
    # load dataset
    data_name = os.path.splitext(data_file_name)[0]
    data_file_path = os.path.join(data_file_dir, data_file_name)
    data = pd.read_csv(data_file_path)
    data = np.array(data)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    
    X = data[:,:-1]
    y = data[:,-1]
    
    
   
    
    
    repeat_times = 5
        
    for i in range(repeat_times):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
        
        
        time_start=time()
        parameters={"truncate_ratio_low":[0], "truncate_ratio_up":[0.4,0.6,0.8 ],
           "min_samples_split":[10,30], "max_depth":[1,2,4,6],
           "order":[0,1,3,6],"splitter":["varreduction"],
            "estimator":["pointwise_extrapolation_estimator"],
           "r_range_low":[0],"r_range_up":[1],
           "step":[1,2,4,8],"lamda":[0.001,0.01,0.1,1,5]}
        cv_model_RTER=GridSearchCV(estimator=RegressionTree(),param_grid=parameters, cv=5, n_jobs=-1)
        cv_model_RTER.fit(X_train, y_train) ##############
        RTER_model = cv_model_RTER.best_estimator_
        mse_score=-RTER_model.score(X_test, y_test)
        time_end=time()
     
        log_file_name = "{}.csv".format("RTER")
        log_file_path = os.path.join(log_file_dir, log_file_name)
        
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{}\n".format(data_name,
                                          mse_score, time_end-time_start,
                                          i)
            f.writelines(logs)
        
        # boosting
        time_start=time()
        parameters={"rho":[0.01,0.05,0.1], "boost_num":[50,100,200], "min_samples_split":[10], "max_depth":[2,5,8],"splitter":["varreduction"]}
        cv_model_boosting=GridSearchCV(estimator=RegressionTreeBoosting(),param_grid=parameters, cv=5, n_jobs=-1)
        cv_model_boosting.fit(X_train, y_train)
        boosting_model = cv_model_boosting.best_estimator_
        mse_score= - boosting_model.score(X_test, y_test)
        log_file_name = "{}.csv".format("boosting")
        log_file_path = os.path.join(log_file_dir, log_file_name)
        time_end=time()
        
        log_file_name = "{}.csv".format("boosting")
        log_file_path = os.path.join(log_file_dir, log_file_name)
        
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{}\n".format(data_name,
                                          mse_score, time_end-time_start,
                                          i)
            f.writelines(logs)
         
        # ensemble
        time_start=time()
        parameters={ "ensemble_num":[50,100,200], "min_samples_split":[10], "max_depth":[2,5,8],"splitter":["varreduction"]}
        cv_model_ensemble=GridSearchCV(estimator=RegressionTreeEnsemble(),param_grid=parameters, cv=5, n_jobs=-1)
        cv_model_ensemble.fit(X_train, y_train)
        ensemble_model = cv_model_ensemble.best_estimator_
        mse_score= - ensemble_model.score(X_test, y_test)
        time_end=time()
        
        log_file_name = "{}.csv".format("ensemble")
        log_file_path = os.path.join(log_file_dir, log_file_name)
        
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{}\n".format(data_name,
                                          mse_score, time_end-time_start,
                                          i)
            f.writelines(logs)
         
        # GBRT
        time_start=time()
        parameters= {"n_estimators":[500,1000,2000], "learning_rate":[0.01,0.05]}
        cv_model_GBRT=GridSearchCV(estimator=GradientBoostingRegressor(),param_grid=parameters, cv=5, n_jobs=-1)
        cv_model_GBRT.fit(X_train, y_train)
        model_GBRT = cv_model_GBRT.best_estimator_
        model_GBRT.fit(X_train, y_train.ravel())
        y_hat=model_GBRT.predict(X_test)
        mse_score = MSE(y_hat, y_test)
        time_end=time()
        
        log_file_name = "{}.csv".format("GBRT")
        log_file_path = os.path.join(log_file_dir, log_file_name)
        
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{}\n".format(data_name,
                                          mse_score, time_end-time_start,
                                          i)
            f.writelines(logs)
            
            
        # RF
        time_start=time()
        parameters = {"n_estimators":[10,100,200]}
        cv_model_RFR = GridSearchCV(estimator=RandomForestRegressor(),param_grid=parameters, cv=5, n_jobs=-1) 
        cv_model_RFR.fit(X_train, y_train)
        model_RFR = cv_model_RFR.best_estimator_
        model_RFR.fit(X_train, y_train)
        y_hat=model_RFR.predict(X_test)
        mse_score = MSE(y_hat, y_test)
        time_end=time()
        
        log_file_name = "{}.csv".format("RFR")
        log_file_path = os.path.join(log_file_dir, log_file_name)
        
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{}\n".format(data_name,
                                          mse_score, time_end-time_start,
                                          i)
            f.writelines(logs)
            
            
        
        # EKNN
        time_start=time()
        parameters = {"V":[4,8,12,16], "C":[1,3,5,7,9,11],"alpha":[0.01,0.05,0.1]}
        cv_model_EKNN = GridSearchCV(estimator=EKNN(),param_grid=parameters, cv=5, n_jobs=-1) 
        cv_model_EKNN.fit(X_train, y_train)
        model_EKNN = cv_model_EKNN.best_estimator_
        model_EKNN.fit(X_train, y_train)
        y_hat=model_EKNN.predict(X_test)
        mse_score = MSE(y_hat, y_test)
        time_end=time()
        
        log_file_name = "{}.csv".format("EKNN")
        log_file_path = os.path.join(log_file_dir, log_file_name)
        
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{}\n".format(data_name,
                                          mse_score, time_end-time_start,
                                          i)
            f.writelines(logs)
        
     
    
        