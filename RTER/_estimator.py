import numpy as np
from sklearn.linear_model import LinearRegression

class ConstantEstimator(object):
    def __init__(self):
        pass
    def assign(self, const):
        self.const = const
    def predict(self, X):
        n_features = X.shape[0]
        y_hat = np.ones(shape=n_features, dtype=self.dtype) * self.const
        return y_hat

class DensityConstantEstimator(ConstantEstimator):
    def __init__(self):
        self.dtype = np.float64

class DensityEstimator(object):
    def __init__(self, X_range, num_samples, dt_X,order=None):
        self.X_range = X_range
        self.volume = np.prod(X_range[1]-X_range[0])
        self.num_samples = num_samples
        self.n_node_samples = dt_X.shape[0]
       
        
    def fit(self):
        if self.n_node_samples != 0:
            pdf = self.n_node_samples / (self.num_samples * self.volume)
        else:
            pdf = 0
        self.den = DensityConstantEstimator()
        self.den.assign(pdf)
    def predict(self, test_X):
        pdf_predict = self.den.predict(test_X)
        return pdf_predict
    
    
class ExtrapolationEstimator(object):
    def __init__(self, X_range, num_samples, dt_X,order):
        self.X_range = X_range
        self.X_central = X_range.mean(axis=0)
        self.X_edge_ratio = X_range[1]-X_range[0]
        self.volume=np.prod(self.X_edge_ratio)
        self.dim=X_range.shape[1]
        self.dt_X=dt_X
        self.order=order
        
        self.n_node_samples=dt_X.shape[0]
        self.num_samples = num_samples
        
        self.truncate=10
        
        
    
    @staticmethod    
    def similar_ratio(instance,X_central,X_edge_ratio):
        return (np.abs(instance-X_central)*2/X_edge_ratio).max()
    

    def extrapolation_vector(self,truncate):
        

        ratio_vec=[self.similar_ratio(self.dt_X[i],self.X_central,self.X_edge_ratio) for i in range(self.n_node_samples)]

        ratio_vec.sort()
        
        #radius=np.prod(self.X_edge_ratio)**(1/self.dim)
        # radius=np.linalg(self.X_edge_ratio)/4
        radius=1
        
        ratio_mat=[[(r*radius)**(2*i+2) for i in range(self.order)] for r in ratio_vec]
        pre_vec=[(i+1)/(self.volume*ratio_vec[i]**self.dim*self.num_samples) for i in range(len(ratio_vec))]
        
        #print(pre_vec)
        
        linear_model=LinearRegression()
       
        linear_model.fit(np.array(ratio_mat)[truncate:],np.array(pre_vec[truncate:]).reshape(-1,1))
      
        if linear_model.intercept_.item()<0:
            print(linear_model.coef_)
            print(np.array(ratio_mat)[truncate:])
            print(np.array(pre_vec[truncate:]))
            raise ValueError
            
     
        
        return linear_model.intercept_.item()
        
    
    def fit(self):
        
        if self.n_node_samples==0:
            pdf=0
        elif self.n_node_samples<=self.truncate:
            pdf=self.n_node_samples / (self.num_samples * self.volume)
        else:
            pdf=self.extrapolation_vector(self.truncate)
        
        self.den = DensityConstantEstimator()
        self.den.assign(pdf)
        
    def predict(self, test_X):
        pdf_predict = self.den.predict(test_X)
        return pdf_predict

