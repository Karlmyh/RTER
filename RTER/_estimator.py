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
                 polynomial_output=0,
                 truncate_ratio_low=0,
                 truncate_ratio_up=1,
                 r_range_up=1,
                 r_range_low=0):
        self.dt_Y=dt_Y
        self.dtype = np.float64
        self.n_node_samples=dt_X.shape[0]
        self.X_range = X_range

        
    def fit(self):
        if self.n_node_samples != 0:
            self.y_hat = self.dt_Y.mean()
        else:
            self.y_hat= self.dt_Y.mean()
        
    def predict(self, test_X,numba_acc=0):
        y_predict = np.full(test_X.shape[0],self.y_hat, dtype=self.dtype)
        return y_predict
    
    
class ExtrapolationEstimator(object):
    def __init__(self, 
                 X_range, 
                 num_samples, 
                 dt_X,
                 dt_Y,
                 order,
                 polynomial_output,
                 truncate_ratio_low,
                 truncate_ratio_up,
                 r_range_up=1,
                 r_range_low=0):
        self.X_range = X_range

        self.X_central = X_range.mean(axis=0)
        self.X_edge_ratio = X_range[1]-X_range[0]

        self.dim=X_range.shape[1]
        self.dt_X=dt_X
        self.dt_Y=dt_Y
        self.order=order
        
        self.n_node_samples=dt_X.shape[0]

        
        self.dtype = np.float64
        self.polynomial_output=polynomial_output
        
   
        self.truncate_ratio_low=truncate_ratio_low
        self.truncate_ratio_up=truncate_ratio_up
        self.r_range_up=r_range_up
        self.r_range_low = r_range_low
    
        
    def similar_ratio(self,instance,X_central,X_edge_ratio):
        return self.unit_square_similar_ratio(self.linear_transform(instance,X_central,X_edge_ratio))
    
    @staticmethod  
    def unit_square_similar_ratio(instance):
        return np.abs(instance-np.zeros(instance.shape[0])).max()
    @staticmethod  
    def linear_transform(instance,X_central,X_edge_ratio):
        return (instance-X_central)/X_edge_ratio*2
    
    def similar_ratio_vec(self, X):
        return [self.similar_ratio(X[i],self.X_central,self.X_edge_ratio) for i in range(X.shape[0])]
    

    def extrapolation(self):
        

        ratio_vec=self.similar_ratio_vec(self.dt_X)
        
        idx_sorted_by_ratio=np.argsort(ratio_vec)      
        self.sorted_ratio = np.array(ratio_vec)[idx_sorted_by_ratio]
        self.sorted_y = self.dt_Y[idx_sorted_by_ratio]
        self.sorted_prediction= np.array([ self.sorted_y[:(i+1)].mean()  for i in range(self.sorted_y.shape[0])])
        
        #print(self.dt_X)
        
        ratio_mat=[[r**(2*i+2) for i in range(self.order)] for r in self.sorted_ratio][int(self.sorted_ratio.shape[0]*self.truncate_ratio_low):int(self.sorted_ratio.shape[0]*self.truncate_ratio_up)]
        pre_vec=[ self.sorted_y[:(i+1)].mean()  for i in range(self.sorted_y.shape[0])][int(self.sorted_ratio.shape[0]*self.truncate_ratio_low):int(self.sorted_ratio.shape[0]*self.truncate_ratio_up)]
        
        ratio_mat = np.array(ratio_mat)
        pre_vec = np.array(pre_vec)
        
        ratio_range_idx_up = (ratio_mat[:,0]**0.5)< self.r_range_up
        ratio_range_idx_low  = (ratio_mat[:,0]**0.5)> self.r_range_low
        ratio_range_idx = ratio_range_idx_up*ratio_range_idx_low
        ratio_mat=ratio_mat[ratio_range_idx]
        pre_vec=pre_vec[ratio_range_idx]
        
        linear_model=LinearRegression()
        linear_model.fit(np.array(ratio_mat),np.array(pre_vec).reshape(-1,1))
        
        self.coef=linear_model.coef_.reshape(-1,1)
        self.intercept=linear_model.intercept_.item()
        

        
        
    
    def fit(self):
        
     
        if self.n_node_samples==0:
            self.y_hat = 0
        else:
            if self.order==0:
                self.y_hat = self.dt_Y.mean()
            else:
                self.extrapolation()
                if self.polynomial_output:
                    self.y_hat=None
                    self.naive_est=self.intercept
                else:
                    self.y_hat=self.intercept
        
    
        
    def predict(self, test_X,numba_acc=0):
        
        if len(test_X)==0:
            return np.array([])
        
        if self.y_hat is None:
            
            ratio_vec=self.similar_ratio_vec(test_X)
            
            
            ratio_mat=np.array([[r**(2*i+2) for i in range(self.order)] for r in ratio_vec])
    
            y_hat=(ratio_mat @ self.coef +self.intercept).ravel()
            
            y_predict=[]
            
            for i in range(y_hat.shape[0]):
                inner_index =  self.sorted_ratio<ratio_vec[i]
                num_inner= inner_index.sum()
                y_predict.append(y_hat[i]* (num_inner+1)-self.sorted_y[self.sorted_ratio<ratio_vec[i]].sum() )
                #print(num_inner)
                #print((y_hat[i]* (num_inner+1),self.sorted_y[self.sorted_ratio<ratio_vec[i]].sum()))
            y_predict=np.array(y_predict)
            
           
            
            truncate_index= (ratio_vec<self.sorted_ratio[self.truncate_low])
            y_predict[truncate_index]=self.naive_est
            
         
        else:
            y_predict = np.full(test_X.shape[0],self.y_hat, dtype=self.dtype)
        
        return y_predict
    

class PointwiseExtrapolationEstimator(object):
    def __init__(self, 
                 X_range, 
                 num_samples, 
                 dt_X,
                 dt_Y,
                 order,
                 polynomial_output,
                 truncate_ratio_low,
                 truncate_ratio_up,
                 r_range_up=1,
                 r_range_low=0):
        self.X_range = X_range

      
        

        self.dim=X_range.shape[1]
        self.dt_X=dt_X
        self.dt_Y=dt_Y
        self.order=order
        
        self.n_node_samples=dt_X.shape[0]

        
        self.dtype = np.float64
        self.polynomial_output=polynomial_output
        
   
        self.truncate_ratio_low=truncate_ratio_low
        self.truncate_ratio_up=truncate_ratio_up
        self.r_range_up=r_range_up
        self.r_range_low = r_range_low
  
        
        
    
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
                                                      self.truncate_ratio_low,self.truncate_ratio_up,self.r_range_low,self.r_range_up))
                else:
                    pre_vec.append(extrapolation_nonjit(self.dt_X,self.dt_Y, 
                                                      X, self.X_range, self.order,
                                                      self.truncate_ratio_low,self.truncate_ratio_up,self.r_range_low,self.r_range_up))
            
            y_predict=np.array(pre_vec)
        else:
            y_predict = np.full(test_X.shape[0],self.y_hat, dtype=self.dtype)
        
        return y_predict