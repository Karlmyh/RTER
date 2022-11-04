from distribution import TestDistribution

from RTER import RegressionTree

import numpy as np


from sklearn.metrics import mean_squared_error as MSE

from time import time
import os



distribution_index_vec=[1,2,3,4]
repeat_time=10


log_file_dir = "./results/simulation/"


for distribution_iter,distribution_index in enumerate(distribution_index_vec):

    for iterate in range(repeat_time):


        
        np.random.seed(iterate)
        # generate distribution


        sample_generator=TestDistribution(distribution_index).returnDistribution()
        n_test, n_train = 5000,5000
        X_train, Y_train = sample_generator.generate(n_train)
        X_test, Y_test = sample_generator.generate_true(n_test)

        #### score mae mse    method time C(parameter) iter ntrain ntest

        # RTER
        time_start=time()
        #parameters={"C":[i for i in np.logspace(-1.5,1.5,15)]}
        #cv_model_AWNN=GridSearchCV(estimator=AWNN(),param_grid=parameters,n_jobs=-1,cv=10)
        #cv_model_AWNN.fit(X_train)
        RTER_model=RegressionTree(estimator="pointwise_extrapolation_estimator",
                         splitter="midpoint",
                         min_samples_split=30,
                         max_depth=3,
                         order=1,
                         random_state=1,
                         truncate_ratio_low=0.3,
                         truncate_ratio_up=0.7,
                         parallel_jobs=8,
                         numba_acc=1)
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
