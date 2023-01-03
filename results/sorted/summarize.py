import os
import numpy as np 
import pandas as pd
import glob

# real data tree summarize
log_file_dir = "../realdata_tree"
method_seq = glob.glob("{}/*.csv".format(log_file_dir))

method_seq = [os.path.split(method)[1].split('.')[0] for method in method_seq]

print(method_seq)

summarize_log=pd.DataFrame([])

for method in method_seq:
    
    log = pd.read_csv("{}/{}.csv".format(log_file_dir,method), header=None)
    
    
    log.columns = "dataset,mse,time,iteration".split(',')
    log["method"]=method
    summarize_log=summarize_log.append(log)
    
    
summary = pd.pivot_table(summarize_log, index=["method"],columns=["dataset"], values=[ "mse","time"], aggfunc=[np.mean, np.std, len])

summary.to_excel("./realdata_tree_summary.xlsx")


# real data forest summarize
log_file_dir = "../realdata_forest"
method_seq = glob.glob("{}/*.csv".format(log_file_dir))

method_seq = [os.path.split(method)[1].split('.')[0] for method in method_seq]

print(method_seq)

summarize_log=pd.DataFrame([])

for method in method_seq:
    
    log = pd.read_csv("{}/{}.csv".format(log_file_dir,method), header=None)
    
    
    log.columns = "dataset,mse,time,iteration".split(',')
    log["method"]=method
    summarize_log=summarize_log.append(log)
    
    
summary = pd.pivot_table(summarize_log, index=["method"],columns=["dataset"], values=[ "mse","time"], aggfunc=[np.mean, np.std, len])

summary.to_excel("./realdata_forest_summary.xlsx")






# performance summarize
log_file_dir = "../accuracy"
method_seq = glob.glob("{}/*.csv".format(log_file_dir))

method_seq = [os.path.split(method)[1].split('.')[0] for method in method_seq]

print(method_seq)

summarize_log=pd.DataFrame([])

for method in method_seq:
    
    log = pd.read_csv("{}/{}.csv".format(log_file_dir,method), header=None)
    
    
    log.columns = "distribution,mse,time,seed,n_train,n_test".split(',')
    log["method"]=method
    summarize_log=summarize_log.append(log)
    
    
summary = pd.pivot_table(summarize_log, index=["method"],columns=["distribution"], values=[ "mse","time"], aggfunc=[np.mean, np.std, len])

summary.to_excel("./accuracy_summary.xlsx")




