import numpy as np
from RTER import RegressionTree


class Ensemble(object):
    def __init__(self, 
                 estimator_fun, 
                 estimator_kargs, 
                 ensemble_num, 
                 rho, 
                 ):
        self.estimator_fun = estimator_fun
        self.estimator_kargs = estimator_kargs
        self.ensemble_num = ensemble_num
        self.regs = []
        
        
    def fit(self, X, y):
       
        for i in range(self.ensemble_num):
            self.regs.append(self.estimator_fun(**self.estimator_kargs))
            self.regs[i].fit(X, y)
            
        
    def predict(self, X):
        y_hat = np.zeros(X.shape[0])
        for i in range(self.ensemble_num):
            y_hat +=  self.regs[i].predict(X)
        y_hat/= self.ensemble_num
        return y_hat
    
    
        

class RegressionTreeEnsemble(Ensemble):
    def __init__(self,  ensemble_num=20, splitter="maxedge", estimator="naive_estimator",
                 min_samples_split=2, max_depth=None, log_Xrange=True, random_state=None, order=1,
                 polynomial_output=0, truncate_ratio_low=0 , truncate_ratio_up=1,numba_acc=1, 
                 parallel_jobs=0, r_range_low=0,r_range_up=1):
        
        estimator_fun = RegressionTree
        estimator_kargs = {"splitter":splitter, "estimator":estimator, "min_samples_split":min_samples_split, 
                           "max_depth":max_depth,"log_Xrange":log_Xrange ,"random_state":random_state,
                           "order":order,"polynomial_output":polynomial_output, "truncate_ratio_low":truncate_ratio_low,
                           "truncate_ratio_up":truncate_ratio_up, "numba_acc":numba_acc,
                           "parallel_jobs":parallel_jobs, "r_range_low":r_range_low,"r_range_up":r_range_up} 
        
        super(Ensemble, self).__init__(estimator_fun=estimator_fun,  estimator_kargs =estimator_kargs, ensemble_num = ensemble_num )