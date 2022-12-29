from distribution import TestDistribution

from RTER import RegressionTree

import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from time import time
import os



distribution_index_vec=[1,2]
repeat_time=5


log_file_dir = "./results/efficiency/"

###### pre computation to trigger compilation #############################
###### do not count into results ##########################################
sample_generator=TestDistribution(1).returnDistribution()
X_train, Y_train = sample_generator.generate(50)
X_test, Y_test = sample_generator.generate_true(50)
RTER_model=RegressionTree(estimator="pointwise_extrapolation_estimator",
                 splitter="midpoint",
                 min_samples_split=30,
                 max_depth=3,
                 random_state=iterate,
                 truncate_ratio_low=0.3,
                 truncate_ratio_up=0.7)
RTER_model.fit(X_train,Y_train)
Y_hat=RTER_model.predict(X_test)
##########################################################################
##########################################################################

for distribution_iter,distribution_index in enumerate(distribution_index_vec):
    for n_train in [1000, 2000, 5000, 10000, 20000, 30000, 40000, 60000]:
        for n_test in [1000,2000,5000]:

            for iterate in range(repeat_time):



                np.random.seed(iterate)
                # generate distribution


                sample_generator=TestDistribution(distribution_index).returnDistribution()
                X_train, Y_train = sample_generator.generate(n_train)
                X_test, Y_test = sample_generator.generate_true(n_test)
                
                
              


                # RTER
                time_start=time()

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
                    logs= "{},{},{},{},{}\n".format(distribution_index, time_end-time_start,
                                                  iterate,n_train,n_test)
                    f.writelines(logs)



                # RTER with parallel
                time_start=time()

                RTER_model=RegressionTree(estimator="pointwise_extrapolation_estimator",
                                 splitter="midpoint",
                                 min_samples_split=30,
                                 max_depth=3,
                                 random_state=iterate,
                                 truncate_ratio_low=0.3,
                                 truncate_ratio_up=0.7,
                                 parallel_jobs = "auto")
                RTER_model.fit(X_train,Y_train)
                Y_hat=RTER_model.predict(X_test)

                time_end=time()

                log_file_name = "{}.csv".format("RTER_parallel")
                log_file_path = os.path.join(log_file_dir, log_file_name)

                with open(log_file_path, "a") as f:
                    logs= "{},{},{},{},{}\n".format(distribution_index, time_end-time_start,
                                                  iterate,n_train,n_test)
                    f.writelines(logs)

       
            
            