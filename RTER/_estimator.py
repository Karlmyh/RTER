import numpy as np
from sklearn.linear_model import LinearRegression



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
        
    def predict(self, test_X):
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
        
        ratio_mat=[[r**(2*i+2) for i in range(self.order)] for r in self.sorted_ratio][-int(self.sorted_ratio.shape[0]*self.truncate_ratio_low):-int(self.sorted_ratio.shape[0]*self.truncate_ratio_up)]
        pre_vec=[ self.sorted_y[:(i+1)].mean()  for i in range(self.sorted_y.shape[0])][-int(self.sorted_ratio.shape[0]*self.truncate_ratio_low):-int(self.sorted_ratio.shape[0]*self.truncate_ratio_up)]
        
      
        
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
    
    
    
class PointwiseNaiveExtrapolationEstimator(object):
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
        
    
    def similar_ratio(self,instance,X_extra,X_range):
  

        return self.unit_square_similar_ratio(self.piecewise_linear_transform(instance,X_extra,X_range))
    
    @staticmethod  
    def unit_square_similar_ratio(instance):
        return np.abs(instance-np.zeros(instance.shape[0])).max()
    @staticmethod  
    def piecewise_linear_transform(instance, X_extra, X_range):
        
        centralized=instance-X_extra
        
        for d in range(instance.shape[0]):
            
            positive_len=X_range[1,d]-X_extra[d]
            negative_len=X_extra[d]-X_range[0,d]
            
            if centralized[d]>=0:
                centralized[d]/=positive_len
            else:
                centralized[d]/=negative_len
        
        return centralized
    

    def similar_ratio_vec(self, X, X_extra):
        return [self.similar_ratio(X[i],X_extra,self.X_range) for i in range(X.shape[0])]
    

    def extrapolation(self):
        
        ratio_mat_all=[]
        pre_all=[]
    
        
        for idx_X, X in enumerate(self.dt_X):
            
            ratio_vec=self.similar_ratio_vec(np.delete(self.dt_X,idx_X,axis=0 ), X)
            
            idx_sorted_by_ratio=np.argsort(ratio_vec)      
            sorted_ratio = np.array(ratio_vec)[idx_sorted_by_ratio]
            sorted_y = self.dt_Y[idx_sorted_by_ratio]
            
        
        #print(self.dt_X)
        
            ratio_mat=[[r**(2*i+2) for i in range(self.order)] for r in sorted_ratio][-int(sorted_ratio.shape[0]*self.truncate_ratio_low):-int(sorted_ratio.shape[0]*self.truncate_ratio_up)]
            pre_vec=[ sorted_y[:(i+1)].mean()  for i in range(sorted_y.shape[0])][-int(sorted_ratio.shape[0]*self.truncate_ratio_low):-int(sorted_ratio.shape[0]*self.truncate_ratio_up)]
        
            ratio_mat_all+=ratio_mat
            pre_all+=pre_vec
      
        self.sorted_ratio=np.array(ratio_mat_all)[:,0] ## saved r^2
        self.sorted_prediction = np.array(pre_all)
          
        linear_model=LinearRegression()
        linear_model.fit(np.array(ratio_mat_all),np.array(pre_all).reshape(-1,1))
        
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
        
    
        
    def predict(self, test_X):
        
        if len(test_X)==0:
            return np.array([])
        
        if self.y_hat is None:
            pass
            
         
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
        
    
    def similar_ratio(self,instance,X_extra,X_range):
  

        return self.unit_square_similar_ratio(self.piecewise_linear_transform(instance,X_extra,X_range))
    
    @staticmethod  
    def unit_square_similar_ratio(instance):
        return np.abs(instance-np.zeros(instance.shape[0])).max()
    @staticmethod  
    def piecewise_linear_transform(instance, X_extra, X_range):
        
        centralized=instance-X_extra
        
        for d in range(instance.shape[0]):
            
            positive_len=X_range[1,d]-X_extra[d]
            negative_len=X_extra[d]-X_range[0,d]
            
            if centralized[d]>=0:
                centralized[d]/=positive_len
            else:
                centralized[d]/=negative_len
        
        return centralized
    

    def similar_ratio_vec(self, X, X_extra):
        return [self.similar_ratio(X[i],X_extra,self.X_range) for i in range(X.shape[0])]
    

    def extrapolation(self,X_extra):
        
        
            
        ratio_vec=self.similar_ratio_vec(self.dt_X, X_extra)
        
        idx_sorted_by_ratio=np.argsort(ratio_vec)      
        sorted_ratio = np.array(ratio_vec)[idx_sorted_by_ratio]
        sorted_y = self.dt_Y[idx_sorted_by_ratio]
            
        
        #print(self.dt_X)
        
        ratio_mat=[[r**(2*i+2) for i in range(self.order)] for r in sorted_ratio][-int(sorted_ratio.shape[0]*self.truncate_ratio_low):-int(sorted_ratio.shape[0]*self.truncate_ratio_up)]
        pre_vec=[ sorted_y[:(i+1)].mean()  for i in range(sorted_y.shape[0])][-int(sorted_ratio.shape[0]*self.truncate_ratio_low):-int(sorted_ratio.shape[0]*self.truncate_ratio_up)]
    
        
          
        linear_model=LinearRegression()
        linear_model.fit(np.array(ratio_mat),np.array(pre_vec).reshape(-1,1))
        
        
        self.intercept=linear_model.intercept_.item()
        
        return self.intercept

        
        
    
    def fit(self):
        
     
        if self.n_node_samples==0:
            self.y_hat = 0
        else:
            if self.order==0:
                self.y_hat = self.dt_Y.mean()
            else:
                self.y_hat=None
        
    
        
    def predict(self, test_X):
        
        if len(test_X)==0:
            return np.array([])
        
        if self.y_hat is None:
            pre_vec=[]
            for X in test_X:
                pre_vec.append(self.extrapolation(X))
            
            y_predict=np.array(pre_vec)
        else:
            y_predict = np.full(test_X.shape[0],self.y_hat, dtype=self.dtype)
        
        return y_predict