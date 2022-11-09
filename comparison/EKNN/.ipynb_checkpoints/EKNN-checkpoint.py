import numpy as np
from sklearn.neighbors import KDTree
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error as MSE

class EKNN(object):
    def __init__(self, V = 5 ,C=10 ,alpha=0.01):
        
        self.V = V
        self.C = C
        self.alpha= alpha 
        
    def fit(self, X, y):
        self.tree = KDTree(X)
        self.y = y
        
        self.dim = X.shape[1]
        self.n_train = X.shape[0]
        
        k_list = [(v+1)*int(self.n_train**(4/(4+self.dim))) for v in range(self.V) ]
        self.k_list = list(set([i for i in k_list if i < self.n_train]))
        return self
        
    def predict(self, X):
        
        return np.array([self.predict_single(x) for x in X])
        
        
    def predict_single(self,xi):
        
        r_matrix = []
        y_hat_vec = []

        for k in self.k_list:  
    
            dist, indices = self.tree.query(xi.reshape(1,-1), k=k)
            r = dist[0][-1]
            r_matrix.append([r**(2*j+2) for j in range(self.C)])
            y_hat_vec.append(self.y[indices].mean())
        r_matrix = np.array(r_matrix)
        y_hat_vec = np.array(y_hat_vec)

        reg = Ridge(alpha=self.alpha).fit(r_matrix, y_hat_vec)
        return reg.intercept_.item()
    
    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in [ "V","C","alpha"]:
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out
    
    
    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)


        for key, value in params.items():
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))
            setattr(self, key, value)
            valid_params[key] = value

        return self
    
    
    def score(self, X, y):
        
        return -MSE(self.predict(X),y)