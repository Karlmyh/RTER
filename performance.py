from distribution import TestDistribution

from RTER import RegressionTree

import numpy as np

from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from time import time
import os



distribution_index_vec=[1,2,3,4]
repeat_time=2


log_file_dir = "./results/accuracy/"


for distribution_iter,distribution_index in enumerate(distribution_index_vec):

    for iterate in range(repeat_time):


        
        np.random.seed(iterate)
        # generate distribution


        sample_generator=TestDistribution(distribution_index).returnDistribution()
        n_test, n_train = 2000,1000
        X_train, Y_train = sample_generator.generate(n_train)
        X_test, Y_test = sample_generator.generate_true(n_test)

        '''
        # RTER
        time_start=time()
        #parameters={"C":[i for i in np.logspace(-1.5,1.5,15)]}
        #cv_model_AWNN=GridSearchCV(estimator=AWNN(),param_grid=parameters,n_jobs=-1,cv=10)
        #cv_model_AWNN.fit(X_train)
        RTER_model=RegressionTree(estimator="pointwise_extrapolation_estimator",
                         splitter="midpoint",
                         min_samples_split=30,
                         max_depth=3,
                         random_state=iterate,
                         truncate_ratio_low=0.3,
                         truncate_ratio_up=0.7)
        RTER_model.fit(X_train,Y_train)
        Y_hat=RTER_model.predict(X_test)
        
        time_end=time()
        
        log_file_name = "{}.csv".format("RTER")
        log_file_path = os.path.join(log_file_dir, log_file_name)
        
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{},{},{}\n".format(distribution_index,
                                          MSE(Y_hat,Y_test), time_end-time_start,
                                          iterate,n_train,n_test)
            f.writelines(logs)

            '''
        
        
        # RTER with cv
        time_start=time()
        parameters={"truncate_ratio_low":[0.1,0.2,0.3 ], "truncate_ratio_up":[0.7,0.8 ]}
        cv_model_RTER=GridSearchCV(estimator=RegressionTree(min_samples_split=30, max_depth=3,parallel_jobs=0),param_grid=parameters, cv=3, n_jobs=18)
        cv_model_RTER.fit(X_train, Y_train)
        
        RTER_model = cv_model_RTER.best_estimator_
        mse_score=RTER_model.score(X_test, Y_test)
        
        time_end=time()
        
        log_file_name = "{}.csv".format("RTER_cv")
        log_file_path = os.path.join(log_file_dir, log_file_name)
        
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{},{},{}\n".format(distribution_index,
                                          mse_score, time_end-time_start,
                                          iterate,n_train,n_test)
            f.writelines(logs)
            
            
            
            
        
        # GBRT
        time_start=time()
        
        model_GBRT = GradientBoostingRegressor(n_estimators = 3000)
        model_GBRT.fit(X_train, Y_train.ravel())
        
        y_hat=model_GBRT.predict(X_test)
        mse_score = MSE(y_hat, Y_test)
        
        time_end=time()
        
        log_file_name = "{}.csv".format("GBRT")
        log_file_path = os.path.join(log_file_dir, log_file_name)
        
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{},{},{}\n".format(distribution_index,
                                          mse_score, time_end-time_start,
                                          iterate,n_train,n_test)
            f.writelines(logs)
            
            
        # RF
        time_start=time()
        
        model_RFR = RandomForestRegressor(n_estimators = 200)
        model_RFR.fit(X_train, Y_train.ravel())
        
        y_hat=model_RFR.predict(X_test)
        mse_score = MSE(y_hat, Y_test)
        
        time_end=time()
        
        log_file_name = "{}.csv".format("RFR")
        log_file_path = os.path.join(log_file_dir, log_file_name)
        
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{},{},{}\n".format(distribution_index,
                                          mse_score, time_end-time_start,
                                          iterate,n_train,n_test)
            f.writelines(logs)
        