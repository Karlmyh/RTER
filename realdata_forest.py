import os
import numpy as np 
import pandas as pd
from time import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as MSE

from RTER import RegressionTree
from comparison.ensemble import RegressionTreeBoosting, RegressionTreeEnsemble
from comparison.EKNN import EKNN


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor






data_file_dir = "./data/real_data_cleaned/"

data_file_name_seq = ["space_ga_scale.csv","triazines_scale.csv",  "bodyfat_scale.csv","housing_scale.csv","mpg_scale.csv"]

#data_seq = glob.glob("{}/*.csv".format(log_file_dir))
#data_file_name_seq = [os.path.split(data)[1] for data in data_seq]

log_file_dir = "./results/realdata_forest/"


for data_file_name in data_file_name_seq:
    # load dataset
    data_name = os.path.splitext(data_file_name)[0]
    data_file_path = os.path.join(data_file_dir, data_file_name)
    data = pd.read_csv(data_file_path)
    data = np.array(data)
    
    X = data[:,1:]
    y = data[:,0]
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    

    repeat_times = 5
        
    for i in range(repeat_times):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i+10)
        
        
        time_start=time()
        parameters={"truncate_ratio_low":[0], "truncate_ratio_up":[1],
           "min_samples_split":[2,5,10,30], "max_depth":[1,2,4,5],
           "order":[0,1,3,6],"splitter":["varreduction"],
            "estimator":["pointwise_extrapolation_estimator"],
           "r_range_low":[0,0.1],"r_range_up":[0.4,0.6,0.8,1],
           "step":[1],"lamda":[0.001,0.01,0.1,1,5],"V":[3,7,11,15,20,25]}
        cv_model_RTER=GridSearchCV(estimator=RegressionTree(),param_grid=parameters, cv=3, n_jobs=-1)
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
            
            
        # DT
        time_start=time()
        parameters = {"max_depth":[2,5,8]}
        cv_model_DT = GridSearchCV(estimator=DecisionTreeRegressor(),param_grid=parameters, cv=5, n_jobs=-1) 
        cv_model_DT.fit(X_train, y_train)
        model_DT = cv_model_DT.best_estimator_
        model_DT.fit(X_train, y_train)
        y_hat=model_DT.predict(X_test)
        mse_score = MSE(y_hat, y_test)
        time_end=time()
    
        log_file_name = "{}.csv".format("DT")
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
        
     
    
        