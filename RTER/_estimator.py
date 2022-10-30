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
            self.y_hat= self.dt_Y.mean()
        
    def predict(self, test_X):
        y_predict = np.full(test_X.shape[0],self.y_hat, dtype=self.dtype)
        return y_predict
    
    
class ExtrapolationEstimator(object):
    def __init__(self, X_range, num_samples, dt_X,dt_Y,order,polynomial_output=1):
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
        
        self.truncate_low=30
        self.truncate_up=50
        self.truncate_ratio_low=0.55
        self.truncate_ratio_up=0.2
        
    
    @staticmethod    
    def similar_ratio(instance,X_central,X_edge_ratio):
        return (np.abs(instance-X_central)*2/X_edge_ratio).max()
    
    def similar_ratio_vec(self, X):
        return [self.similar_ratio(X[i],self.X_central,self.X_edge_ratio) for i in range(X.shape[0])]
    

    def extrapolation(self):
        

        ratio_vec=self.similar_ratio_vec(self.dt_X)
        
        idx_sorted_by_ratio=np.argsort(ratio_vec)      
        self.sorted_ratio = np.array(ratio_vec)[idx_sorted_by_ratio]
        self.sorted_y = self.dt_Y[idx_sorted_by_ratio]
        
        print(self.dt_X)
   
        ratio_mat=[[r**(2*i+2) for i in range(self.order)] for r in self.sorted_ratio][-int(self.sorted_ratio.shape[0]*self.truncate_ratio_low):-int(self.sorted_ratio.shape[0]*self.truncate_ratio_up)]
        pre_vec=[ self.sorted_y[:(i+1)].mean()  for i in range(self.sorted_y.shape[0])][-int(self.sorted_ratio.shape[0]*self.truncate_ratio_low):-int(self.sorted_ratio.shape[0]*self.truncate_ratio_up)]
        
        #print(np.array(ratio_mat).shape)
        #print(np.array(pre_vec).shape)
        
        print([r**2  for r in self.sorted_ratio])
        print([ self.sorted_y[:(i+1)].mean()  for i in range(self.sorted_y.shape[0])])
        
        linear_model=LinearRegression()
        linear_model.fit(np.array(ratio_mat),np.array(pre_vec).reshape(-1,1))
        
        self.coef=linear_model.coef_.reshape(-1,1)
        self.intercept=linear_model.intercept_.item()
        
        print(self.intercept)

        
        
    
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
        
    
        
    def predict(self, test_X):
        
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

