import numpy as np
from RTER import RegressionTree
from sklearn.metrics import mean_squared_error as MSE
from multiprocessing import Pool

def single_parallel(input_tuple):
    tree, X, y, random_state, max_samples = input_tuple
    np.random.seed(random_state)
    bootstrap_idx = np.random.choice(X.shape[0], int(np.ceil(X.shape[0] * max_samples)))
    return tree.fit(X[bootstrap_idx],y[bootstrap_idx])


class RegressionTreeEnsemble(object):
    def __init__(self,  n_estimators = 20, max_features = 1.0, max_samples = 1.0,
                 splitter="maxedge", estimator = "naive_estimator", ensemble_parallel = 0,
                 min_samples_split=2, max_depth=None, order=1, log_Xrange=True, 
                 random_state=None,truncate_ratio_low=0 , truncate_ratio_up=1,
                 index_by_r=1, parallel_jobs=0, r_range_low=0,
                 r_range_up=1,step = 1, V = 0,lamda=0.01 
                 ):
        
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.splitter = splitter
        self.estimator = estimator
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.order=order
        self.step=step
        self.log_Xrange = log_Xrange
        self.random_state = random_state
        self.truncate_ratio_low=truncate_ratio_low
        
        self.truncate_ratio_up=truncate_ratio_up
        self.index_by_r=index_by_r
        
        self.parallel_jobs = parallel_jobs
        self.r_range_up =r_range_up
        self.r_range_low =r_range_low
        self.lamda=lamda
        self.V = V
        self.ensemble_parallel = ensemble_parallel
        
        self.trees = []

        
    def fit(self, X, y):
        
        if self.ensemble_parallel !=0:
        
            for i in range(self.n_estimators):

                self.trees.append(RegressionTree(splitter=self.splitter, 
                                                 estimator=self.estimator, 
                                                 min_samples_split=self.min_samples_split,
                                                 order=self.order, 
                                                 max_depth=self.max_depth, 
                                                 log_Xrange=self.log_Xrange, 
                                                 random_state=i,
                                                 truncate_ratio_low=self.truncate_ratio_low,
                                                 truncate_ratio_up=self.truncate_ratio_up,
                                                 index_by_r=self.index_by_r,
                                                 parallel_jobs=self.parallel_jobs,
                                                 r_range_low=self.r_range_low,
                                                 r_range_up=self.r_range_up,
                                                 step=self.step,
                                                 V=self.V,
                                                 lamda=self.lamda,
                                                 max_features=self.max_features))

            with Pool(min(50,self.n_estimators)) as p:
                self.trees = p.map(single_parallel, [(self.trees[i],X,y,i,self.max_samples) for i in range(self.n_estimators)])
                
                
        else:
            for i in range(self.n_estimators):
                np.random.seed(i)
            
                bootstrap_idx = np.random.choice(X.shape[0], int(np.ceil(X.shape[0] * self.max_samples)))



                self.trees.append(RegressionTree(splitter=self.splitter, 
                                                 estimator=self.estimator, 
                                                 min_samples_split=self.min_samples_split,
                                                 order=self.order, 
                                                 max_depth=self.max_depth, 
                                                 log_Xrange=self.log_Xrange, 
                                                 random_state=i,
                                                 truncate_ratio_low=self.truncate_ratio_low,
                                                 truncate_ratio_up=self.truncate_ratio_up,
                                                 index_by_r=self.index_by_r,
                                                 parallel_jobs=self.parallel_jobs,
                                                 r_range_low=self.r_range_low,
                                                 r_range_up=self.r_range_up,
                                                 step=self.step,
                                                 V=self.V,
                                                 lamda=self.lamda,
                                                 max_features=self.max_features))

                self.trees[i].fit(X[bootstrap_idx] , y[bootstrap_idx])
        
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
        for key in [ "n_estimators" ,"min_samples_split", "max_features",
                    "max_depth","order","truncate_ratio_low", "max_samples",
                    "truncate_ratio_up","splitter","r_range_low","r_range_up",
                    "step","lamda","estimator","V"]:
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
    
    def predict(self, X):
        y_hat = np.zeros(X.shape[0])
        for i in range(self.n_estimators):
            y_hat +=  self.trees[i].predict(X)
        y_hat/= self.n_estimators
        return y_hat
    
    
    def score(self, X, y):
        
        return -MSE(self.predict(X),y)