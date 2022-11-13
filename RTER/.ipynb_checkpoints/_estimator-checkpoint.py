import numpy as np
from sklearn.linear_model import LinearRegression

from ._utils import extrapolation_jit, extrapolation_nonjit

class NaiveEstimator(object):
    def __init__(self, 
                 X_range, 
                 num_samples, 
                 dt_X, 
                 dt_Y, 
                 order=None,
                 truncate_ratio_low=0,
                 truncate_ratio_up=1,
                 step =1,
                 r_range_up=1,
                 r_range_low=0,
                lamda=0.01):
        self.dt_Y=dt_Y
        self.dtype = np.float64
        self.n_node_samples=dt_X.shape[0]
        self.X_range = X_range

        
    def fit(self):
        if self.n_node_samples != 0:
            self.y_hat = self.dt_Y.mean()
        else:
            self.y_hat= 0
        
    def predict(self, test_X,numba_acc=0):
        y_predict = np.full(test_X.shape[0],self.y_hat, dtype=self.dtype)
        return y_predict
    

class PointwiseExtrapolationEstimator(object):
    def __init__(self, 
                 X_range, 
                 num_samples, 
                 dt_X,
                 dt_Y,
                 order,
                 truncate_ratio_low,
                 truncate_ratio_up,
                 step=1,
                 r_range_up=1,
                 r_range_low=0,
                lamda=0.01,
                ):
        self.X_range = X_range

      
        

        self.dim=X_range.shape[1]
        self.dt_X=dt_X
        self.dt_Y=dt_Y
        self.order=order
        self.lamda = lamda
        self.n_node_samples=dt_X.shape[0]

        
        self.dtype = np.float64

        
   
        self.truncate_ratio_low=truncate_ratio_low
        self.truncate_ratio_up=truncate_ratio_up
        self.r_range_up=r_range_up
        self.r_range_low = r_range_low
        self.step =step
        
        
    
    def fit(self):
        
     
        if self.n_node_samples==0:
            self.y_hat = 0
        else:
            self.y_hat=None
        
    
        
    def predict(self, test_X,numba_acc=0):
        
        if len(test_X)==0:
            return np.array([])
        
        if self.y_hat is None:
            pre_vec=[]
            for X in test_X:
                if numba_acc:
                    pre_vec.append(extrapolation_jit(self.dt_X,self.dt_Y, 
                                                      X, self.X_range, self.order,
                                                      self.truncate_ratio_low,self.truncate_ratio_up,self.r_range_low,self.r_range_up,self.step,self.lamda))
                else:
                    pre_vec.append(extrapolation_nonjit(self.dt_X,self.dt_Y, 
                                                      X, self.X_range, self.order,
                                                      self.truncate_ratio_low,self.truncate_ratio_up,self.r_range_low,self.r_range_up,self.step,self.lamda))
            
            y_predict=np.array(pre_vec)
        else:
            y_predict = np.full(test_X.shape[0],self.y_hat, dtype=self.dtype)
        
        return y_predict