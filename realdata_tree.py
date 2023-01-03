import os
import numpy as np 
import pandas as pd
from time import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as MSE

from RTER import RegressionTree



from sklearn.tree import DecisionTreeRegressor





data_file_dir = "./data/real_data_cleaned/"

data_file_name_seq = ['housing_scale.csv', 'mpg_scale.csv','space_ga_scale.csv','mg_scale.csv','cpusmall_scale.csv',
                      'triazines_scale.csv','pyrim_scale.csv',
                      'abalone.csv','bodyfat_scale.csv']

#data_seq = glob.glob("{}/*.csv".format(log_file_dir))
#data_file_name_seq = [os.path.split(data)[1] for data in data_seq]

log_file_dir = "./results/realdata_tree/"


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
        
        
        # RTER
        parameters={"min_samples_split":[2,5,10], "max_depth":[1,2,3,4,5,6,7,8],
       "order":[0,1],"splitter":["maxedge"],
        "estimator":["pointwise_extrapolation_estimator"],
       "r_range_low":[0],"r_range_up":[0.6,1],
       "lamda":[0.0001,0.001,0.01,0.1],"V":[2,"auto"]}
        
        cv_model_RTER=GridSearchCV(estimator=RegressionTree(),param_grid=parameters, cv=3, n_jobs=40)
        cv_model_RTER.fit(X_train, y_train)
        
        
        time_start=time()
        RTER_model = cv_model_RTER.best_estimator_
        mse_score= -RTER_model.score(X_test, y_test)
        time_end=time()
     
        log_file_name = "{}.csv".format("RTER_pure")
        log_file_path = os.path.join(log_file_dir, log_file_name)
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{}\n".format(data_name,
                                          mse_score, time_end-time_start,
                                          i)
            f.writelines(logs)
            
        
        '''
        # RTER with var reduction
        parameters={"min_samples_split":[2,5,10], "max_depth":[1,2,3,4,5,6,7,8],
       "order":[0,1],"splitter":["varreduction"],
        "estimator":["pointwise_extrapolation_estimator"],
       "r_range_low":[0],"r_range_up":[0.6,1],
       "lamda":[0.0001,0.001,0.01,0.1],"V":[2,"auto"]}
        
        cv_model_RTER=GridSearchCV(estimator=RegressionTree(),param_grid=parameters, cv=3, n_jobs=40)
        cv_model_RTER.fit(X_train, y_train)
        
        
        time_start=time()
        RTER_model = cv_model_RTER.best_estimator_
        mse_score= -RTER_model.score(X_test, y_test)
        time_end=time()
     
        log_file_name = "{}.csv".format("RTER")
        log_file_path = os.path.join(log_file_dir, log_file_name)
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{}\n".format(data_name,
                                          mse_score, time_end-time_start,
                                          i)
            f.writelines(logs)
            
        
        # DT
        
        parameters = {"max_depth":[2,4,6]}
        
        cv_model_DT = GridSearchCV(estimator=DecisionTreeRegressor(),param_grid=parameters, cv=3, n_jobs=-1) 
        cv_model_DT.fit(X_train, y_train)
        
        time_start=time()
        model_DT = cv_model_DT.best_estimator_
        prediction = model_DT.predict(X_test) 
        mse_score = np.mean((prediction - y_test)**2)
        time_end=time()
    
        log_file_name = "{}.csv".format("DT")
        log_file_path = os.path.join(log_file_dir, log_file_name)
        with open(log_file_path, "a") as f:
            logs= "{},{},{},{}\n".format(data_name,
                                          mse_score, time_end-time_start,
                                          i)
            f.writelines(logs)
            
            
            '''

            
     
    
        