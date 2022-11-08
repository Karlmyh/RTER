import numpy as np
from RTER import RegressionTree


class Boosting(object):
    def __init__(self, 
                 estimator_fun, 
                 estimator_kargs, 
                 boost_num, 
                 rho, 
                 ):
        self.estimator_fun = estimator_fun
        self.estimator_kargs = estimator_kargs
        self.boost_num = boost_num
        self.rho = rho
        self.regs = []
        
        
    def fit(self, X, y):
        length = X.shape[0]
        f_hat = np.zeros(length)
        for i in range(self.boost_num):
            self.regs.append(self.estimator_fun(**self.estimator_kargs))
            self.regs[i].fit(X, y-f_hat)
            f_hat += self.rho * self.regs[i].predict(X)
        
    def predict(self, X):
        y_hat = np.zeros(X.shape[0])
        for i in range(self.boost_num):
            y_hat += self.rho * self.regs[i].predict(X)
        return y_hat
    
    
        

class RegressionTreeBoosting(Boosting):
    def __init__(self, rho=0.1, boost_num=20, splitter="maxedge", estimator="naive_estimator",
                 min_samples_split=2, max_depth=None, log_Xrange=True, random_state=None, order=1,
                 polynomial_output=0, truncate_ratio_low=0 , truncate_ratio_up=1,numba_acc=1, 
                 parallel_jobs=0, r_range_low=0,r_range_up=1):
        
        estimator_fun = RegressionTree
        estimator_kargs = {"splitter":splitter, "estimator":estimator, "min_samples_split":min_samples_split, 
                           "max_depth":max_depth,"log_Xrange":log_Xrange ,"random_state":random_state,
                           "order":order,"polynomial_output":polynomial_output, "truncate_ratio_low":truncate_ratio_low,
                           "truncate_ratio_up":truncate_ratio_up, "numba_acc":numba_acc,
                           "parallel_jobs":parallel_jobs, "r_range_low":r_range_low,"r_range_up":r_range_up} 
        
        super(Boosting, self).__init__(estimator_fun=estimator_fun,  estimator_kargs =estimator_kargs, boost_num = boost_num, rho =rho )