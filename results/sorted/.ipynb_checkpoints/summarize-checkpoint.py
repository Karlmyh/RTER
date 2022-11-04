import os
import numpy as np 
import pandas as pd
import glob

# simulation summarize
log_file_dir = "./simulation_result"
method_seq = glob.glob("{}/*.csv".format(log_file_dir))

method_seq = [os.path.split(method)[1].split('.')[0] for method in method_seq]

print(method_seq)

summarize_log=pd.DataFrame([])

for method in method_seq:
    
    log = pd.read_csv("{}/{}.csv".format(log_file_dir,method), header=None)
    
    
    log.columns = "distribution,dim,anll,L2,mae,mse,time,best_params,seed,n_train,n_test".split(',')
    log["method"]=method
    summarize_log=summarize_log.append(log)
    
    
summary = pd.pivot_table(summarize_log, index=["dim", "method"],columns=["distribution"], values=[ "L2","mae"], aggfunc=[np.mean, np.std, len])

summary.to_excel("./sorted_result/simulation_summary.xlsx")


# simulation l1 summarize
log_file_dir = "./simulation_result_l1"
method_seq = glob.glob("{}/*.csv".format(log_file_dir))

method_seq = [os.path.split(method)[1].split('.')[0] for method in method_seq]

print(method_seq)

summarize_log=pd.DataFrame([])

for method in method_seq:
    
    log = pd.read_csv("{}/{}.csv".format(log_file_dir,method), header=None)
    
    
    log.columns = "distribution,dim,anll,L2,mae,mse,time,best_params,seed,n_train,n_test".split(',')
    log["method"]=method
    summarize_log=summarize_log.append(log)
    
    
summary = pd.pivot_table(summarize_log, index=["dim", "method"],columns=["distribution"], values=[ "L2","mae"], aggfunc=[np.mean, np.std, len])

summary.to_excel("./sorted_result/simulation_l1_summary.xlsx")





# real data summarize
log_file_dir = "./realdata_result"
method_seq = glob.glob("{}/*.csv".format(log_file_dir))

method_seq = [os.path.split(method)[1].split('.')[0] for method in method_seq]

print(method_seq)

summarize_log=pd.DataFrame([])

for method in method_seq:

    log = pd.read_csv("{}/{}.csv".format(log_file_dir,method), header=None)
    
    
    log.columns = "dataset,dim,ANLL,L2,time,params,iter".split(',')
    log["method"]=method.split('_')[1]
    summarize_log = summarize_log.append(log)
    
summary = pd.pivot_table(summarize_log, index=["dataset", "dim"], columns=["method"],values=["L2"], aggfunc=[np.mean, np.std, len])
    
summary.to_excel("./sorted_result/realdata_summary.xlsx")

'''
 
summary = pd.pivot_table(summarize_log, index=["dataset", "dim"], columns=["method"],values=["time"], aggfunc=[np.mean, np.std, len])
    
summary.to_excel("./sorted_result/real_data_time.xlsx")
 



# anomaly summarize

log_file_dir = "./anomaly_result"
log_file_name = "anomaly"
log = pd.read_csv("{}/{}.csv".format(log_file_dir,log_file_name), header=None)
log.columns = "filename,IF,LOF,OCSVM,KNN,AWNN,rankIF,rankLOF,rankOCSVM,rankKNN,rankAWNN".split(',')
summary=log[{"filename","IF","LOF","OCSVM","KNN","AWNN"}].reindex(columns=["filename",
                                                                           "IF","LOF","OCSVM","KNN","AWNN"])

summary=summary.append({"filename":"ranksum","IF":log["rankIF"].sum(),
                "LOF":log["rankLOF"].sum(),"OCSVM":log["rankOCSVM"].sum(),
                "KNN":log["rankKNN"].sum(),"AWNN":log["rankAWNN"].sum()},ignore_index=True)



summary.to_excel("./sorted_result/anomaly_summary.xlsx")

'''

