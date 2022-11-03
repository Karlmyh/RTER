import numpy as np
from sklearn.linear_model import LinearRegression
from numba import njit


class NaiveEstimator(object):
    def __init__(self, 
                 X_range, 
                 num_samples, 
                 dt_X, 
                 dt_Y, 
                 order=None,
                 polynomial_output=0,
                 truncate_ratio_low=0.55,
                 truncate_ratio_up=0.2):
        self.dt_Y=dt_Y
        self.dtype = np.float64
        self.n_node_samples=dt_X.shape[0]
        
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
                 truncate_ratio_up):
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
    
def extrapolation_nonjit(dt_X,dt_Y, X_extra, X_range, order, truncate_ratio_low,truncate_ratio_up):

    ratio_vec=np.array([])
    for idx_X, X in enumerate(dt_X):
        
        centralized=X-X_extra
        
        for d in range(X_extra.shape[0]):
            positive_len=X_range[1,d]-X_extra[d]
            negative_len=X_extra[d]-X_range[0,d]
            
            if centralized[d]>=0:
                centralized[d]/=positive_len
            else:
                centralized[d]/=negative_len
        
        ratio_X= np.abs(centralized).max()
        ratio_vec=np.append(ratio_vec,ratio_X)

    

    idx_sorted_by_ratio=np.argsort(ratio_vec)      
    sorted_ratio = ratio_vec[idx_sorted_by_ratio]
    sorted_y = dt_Y[idx_sorted_by_ratio]

    ratio_mat=np.array([[r**(2*i) for i in range(order+1)] for r in sorted_ratio][int(sorted_ratio.shape[0]*truncate_ratio_low):int(sorted_ratio.shape[0]*truncate_ratio_up)])
    pre_vec=np.array([ sorted_y[:(i+1)].mean()  for i in range(sorted_y.shape[0])][int(sorted_ratio.shape[0]*truncate_ratio_low):int(sorted_ratio.shape[0]*truncate_ratio_up)]).reshape(-1,1)
    
    

    

    return (np.linalg.inv(ratio_mat.T @ ratio_mat) @ ratio_mat.T @ pre_vec)[0].item()

@njit
def extrapolation_jit(dt_X,dt_Y, X_extra, X_range, order, truncate_ratio_low,truncate_ratio_up):

    ratio_vec=np.array([])
    for idx_X, X in enumerate(dt_X):
        
        centralized=X-X_extra
        
        for d in range(X_extra.shape[0]):
            positive_len=X_range[1,d]-X_extra[d]
            negative_len=X_extra[d]-X_range[0,d]
            
            if centralized[d]>=0:
                centralized[d]/=positive_len
            else:
                centralized[d]/=negative_len
        
        ratio_X= np.abs(centralized).max()
        ratio_vec=np.append(ratio_vec,ratio_X)
        
    np.argsort(ratio_vec) 

    idx_sorted_by_ratio = np.argsort(ratio_vec)      
    sorted_ratio = ratio_vec[idx_sorted_by_ratio]
    sorted_y = dt_Y[idx_sorted_by_ratio]
    
    
    
    ratio_mat=np.zeros((sorted_ratio.shape[0], order+1))
    
    n_test= sorted_ratio.shape[0]
    
 
    i=0
    while(i<n_test):
        r= sorted_ratio[i]
        i+=1
        for j in range(order +1):
            ratio_mat[i,j]= r**(2*j) 
            
    ratio_mat_used=ratio_mat[int(sorted_ratio.shape[0]*truncate_ratio_low):int(sorted_ratio.shape[0]*truncate_ratio_up)]
    
   
    pre_vec=np.zeros((sorted_y.shape[0],1))
    for k in range(sorted_y.shape[0]):
        pre_vec[k,0]= np.mean(sorted_y[:(k+1)])
    
    pre_vec_used=pre_vec[int(sorted_ratio.shape[0]*truncate_ratio_low):int(sorted_ratio.shape[0]*truncate_ratio_up)]
    

    return (np.linalg.inv(ratio_mat_used.T @ ratio_mat_used) @ ratio_mat_used.T @ pre_vec_used )[0,0]
    
    
    

class PointwiseExtrapolationEstimator(object):
    def __init__(self, 
                 X_range, 
                 num_samples, 
                 dt_X,
                 dt_Y,
                 order,
                 polynomial_output,
                 truncate_ratio_low,
                 truncate_ratio_up):
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
        
  
        
        
    
    def fit(self):
        
     
        if self.n_node_samples==0:
            self.y_hat = 0
        else:
            if self.order==0:
                self.y_hat = self.dt_Y.mean()
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
                                                      self.truncate_ratio_low,self.truncate_ratio_up))
                else:
                    pre_vec.append(extrapolation_nonjit(self.dt_X,self.dt_Y, 
                                                      X, self.X_range, self.order,
                                                      self.truncate_ratio_low,self.truncate_ratio_up))
            
            y_predict=np.array(pre_vec)
        else:
            y_predict = np.full(test_X.shape[0],self.y_hat, dtype=self.dtype)
        
        return y_predict