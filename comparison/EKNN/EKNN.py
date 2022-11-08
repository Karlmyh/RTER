import numpy as np
from sklearn.neighbors import KDTree
from sklearn.linear_model import LinearRegression

class zeronn(object):
    def __init__(self, V = 5 ,C=10 ):
        
        self.V = 5 
        self.C = 10
        
    def fit(self, X, y):
        self.tree = KDTree(X)
        self.y = y
        
        self.dim = X.shape[1]
        self.n_train = X.shape[0]
        
        self.k_list = [(v+1)*int(self.n_train**(4/(4+self.dim))) for v in range(self.V) ]
    
        return self
        
    def predict(self, X):
        
        return np.array([self.predict_single(x) for x in X])
        
        
    def predict_single(self,xi):
        
        r_matrix = []
        y_hat_vec = []

        for k in self.k_list:  
    
            dist, indices = self.tree.query(xi.reshape(1,-1), k=k)
            r = dist[-1]
            r_matrix.append([r**(2*j+2) for j in range(self.C)])
            y_hat_vec.append(self.y[indices].mean())
        r_matrix = np.array(r_matrix)
        y_hat_vec = np.array(y_hat_vec)
        reg = LinearRegression().fit(r_matrix, y_hat_vec)
        return reg.intercept_