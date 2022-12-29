import numpy as np
from RTER import RegressionTree
from sklearn.metrics import mean_squared_error as MSE

class Ensemble(object):
    def __init__(self, 
                 estimator, 
                 estimator_kargs, 
                 n_estimators,
                 max_samples
                 ):
        self.estimator = estimator
        self.estimator_kargs = estimator_kargs
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.trees = []
        
        
    def fit(self, X, y):
       
        for i in range(self.n_estimators):
            self.estimator_kargs["random_state"] = i
            
            bootstrap_idx = np.random.choice(X.shape[0], int(X.shape[0] * self.max_samples))
            
            self.trees.append(self.estimator(**self.estimator_kargs))
            self.trees[i].fit(X[bootstrap_idx] , y[bootstrap_idx])
            
        
    def predict(self, X):
        y_hat = np.zeros(X.shape[0])
        for i in range(self.n_estimators):
            y_hat +=  self.trees[i].predict(X)
        y_hat/= self.n_estimators
        return y_hat
    
    
        

class RegressionTreeEnsemble(Ensemble):
    def __init__(self,  n_estimators = 20, max_features = 1.0, max_samples = 1.0,
                 splitter="maxedge", estimator = "naive_estimator", 
                 min_samples_split=2, max_depth=None, order=1, log_Xrange=True, 
                 random_state=None,truncate_ratio_low=0 , truncate_ratio_up=1,
                 index_by_r=1, parallel_jobs=0, r_range_low=0,
                 r_range_up=1,step = 1, V = 0,lamda=0.01, 
                 ):
        

        
        estimator = RegressionTree
        estimator_kargs = {"max_features":max_features, "splitter":splitter, 
                           "estimator":estimator, "min_samples_split":min_samples_split, 
                           "max_depth":max_depth,"log_Xrange":log_Xrange,
                           "order":order, "truncate_ratio_low":truncate_ratio_low,
                           "truncate_ratio_up":truncate_ratio_up, "index_by_r":index_by_r,
                           "parallel_jobs":parallel_jobs, "r_range_low":r_range_low,
                           "r_range_up":r_range_up, "step":step, "V":V,
                           "lamda":lamda} 
        
        super(RegressionTreeEnsemble, self).__init__(estimator=estimator,  
                                                     estimator_kargs = estimator_kargs, 
                                                     n_estimators = n_estimators,
                                                     max_samples = max_samples)
        
        
        
        
        
        
        
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
        for key in [ "n_estimators" ,"min_samples_split", "max_features"
                    "max_depth","order","truncate_ratio_low", "max_samples"
                    "truncate_ratio_up","splitter","r_range_low","r_range_up",
                    "step","lamda","estimator","V","max_features"]:
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