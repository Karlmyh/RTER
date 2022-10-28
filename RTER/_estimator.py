import numpy as np
from sklearn.linear_model import LinearRegression



class NaiveEstimator(object):
    def __init__(self, X_range, num_samples, dt_X, dt_Y, order=None):
        self.dt_Y=dt_Y
        self.dtype = np.float64
        self.n_node_samples=dt_X.shape[0]
        
    def fit(self):
        if self.n_node_samples != 0:
            self.y_hat = self.dt_Y.mean()
        else:
            self.y_hat=0
        
    def predict(self, test_X):
        y_predict = np.full(test_X.shape[0],self.y_hat, dtype=self.dtype)
        return y_predict
    
    
class ExtrapolationEstimator(object):
    def __init__(self, X_range, num_samples, dt_X,dt_Y,order,polynomial_output=1):
        self.X_range = X_range
        self.X_central = X_range.mean(axis=0)
        self.X_edge_ratio = X_range[1]-X_range[0]
        self.volume=np.prod(self.X_edge_ratio)
        self.dim=X_range.shape[1]
        self.dt_X=dt_X
        self.dt_Y=dt_Y
        self.order=order
        
        self.n_node_samples=dt_X.shape[0]

        
        self.dtype = np.float64
        self.polynomial_output=polynomial_output
        
        
    
    @staticmethod    
    def similar_ratio(instance,X_central,X_edge_ratio):
        return (np.abs(instance-X_central)*2/X_edge_ratio).max()
    
    def similar_ratio_vec(self, X):
        return [self.similar_ratio(X[i],self.X_central,self.X_edge_ratio) for i in range(X.shape[0])]
    

    def extrapolation(self):
        

        ratio_vec=self.similar_ratio_vec(self.dt_X)
        
        idx_sorted_by_ratio=np.argsort(ratio_vec)      
        sorted_ratio = np.array(ratio_vec)[idx_sorted_by_ratio]
        sorted_y = self.dt_Y[idx_sorted_by_ratio]
        
       
        ratio_mat=[[r**(2*i+2) for i in range(self.order)] for r in sorted_ratio]
        pre_vec=[ sorted_y[:(i+1)].mean()  for i in range(sorted_y.shape[0])]
        
        linear_model=LinearRegression()
        linear_model.fit(np.array(ratio_mat),np.array(pre_vec).reshape(-1,1))
        
        self.coef=linear_model.coef_.reshape(-1,1)
        self.intercept=linear_model.intercept_.item()

        
        
    
    def fit(self):
        
        if self.n_node_samples==0:
            self.y_hat=0    
        elif self.order==0:
            self.y_hat = self.dt_Y.mean()
        else:
            self.extrapolation()
            if self.polynomial_output:
                self.y_hat=None
            else:
                self.y_hat=self.intercept
        
    
        
    def predict(self, test_X):
        
        if self.y_hat is None:
            ratio_vec=self.similar_ratio_vec(test_X)
            ratio_mat=np.array([[r**(2*i+2) for i in range(self.order)] for r in ratio_vec])
            y_predict=(ratio_mat @ self.coef +self.intercept).ravel()
        else:
            y_predict = np.full(test_X.shape[0],self.y_hat, dtype=self.dtype)
        
        return y_predict

