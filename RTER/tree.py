import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import MinMaxScaler

from ._tree import TreeStruct, RecursiveTreeBuilder
from ._splitter import PurelyRandomSplitter,MidPointRandomSplitter, MaxEdgeRandomSplitter
from ._estimator import NaiveEstimator,ExtrapolationEstimator,PointwiseExtrapolationEstimator
from ._utils import extrapolation_jit_return_info


SPLITTERS = {"purely": PurelyRandomSplitter,"midpoint":MidPointRandomSplitter, "maxedge":MaxEdgeRandomSplitter}
ESTIMATORS = {"naive_estimator": NaiveEstimator,"extrapolation_estimator":ExtrapolationEstimator,"pointwise_extrapolation_estimator":PointwiseExtrapolationEstimator}

class BaseRecursiveTree(object):
    def __init__(self, 
                 splitter="midpoint", 
                 estimator=None, 
                 min_samples_split=2, 
                 max_depth=None, 
                 order=None, 
                 log_Xrange=None, 
                 random_state=None,
                 polynomial_output=None,
                 truncate_ratio_low=None,
                 truncate_ratio_up=None,
                 numba_acc=None,
                 parallel_jobs=None):
        self.splitter = splitter
        self.estimator = estimator
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.order=order
    
        self.log_Xrange = log_Xrange
        self.random_state = random_state
        self.polynomial_output=polynomial_output
        self.truncate_ratio_low=truncate_ratio_low
        
        self.truncate_ratio_up=truncate_ratio_up
        self.numba_acc=numba_acc
        
        self.parallel_jobs = parallel_jobs
             
    def fit(self, X, Y,X_range=None):
        self.n_samples, self.n_features = X.shape
        # checking parameters and preparation
        max_depth = (1 if self.max_depth is None
                     else self.max_depth)
        
        order= (10 if self.order is None
                else self.order)
        if self.min_samples_split < 1:
            raise ValueError("min_samples_split should be larger than 1, got {}.".format(self.min_samples_split))
        # begin
        splitter = SPLITTERS[self.splitter](self.random_state)
        Estimator = ESTIMATORS[self.estimator]
        self.tree_ = TreeStruct(self.n_samples, self.n_features, self.log_Xrange)
        builder = RecursiveTreeBuilder(splitter, 
                                       Estimator, 
                                       self.min_samples_split, 
                                       max_depth, 
                                       order,
                                       self.polynomial_output,
                                       self.truncate_ratio_low,
                                       self.truncate_ratio_up)
        builder.build(self.tree_, X, Y,X_range)
    def apply(self, X):
        return self.tree_.apply(X)
    def predict(self, X):
        if self.parallel_jobs != 0:
            #print("we are using parallel computing!")
            return self.tree_.predict_parallel(X, self.numba_acc,parallel_jobs=self.parallel_jobs)
        else:
            return self.tree_.predict(X, self.numba_acc)


class RegressionTree(BaseRecursiveTree):
    def __init__(self, splitter="maxedge", estimator="pointwise_extrapolation_estimator", min_samples_split=2, max_depth=None, order=1, log_Xrange=True, random_state=None,polynomial_output=0, truncate_ratio_low=0.2 , truncate_ratio_up=0.55,numba_acc=1,parallel_jobs=0):
        super(RegressionTree, self).__init__(splitter=splitter, estimator=estimator, min_samples_split=min_samples_split,order=order, max_depth=max_depth, log_Xrange=log_Xrange, random_state=random_state,polynomial_output=polynomial_output,truncate_ratio_low=truncate_ratio_low,truncate_ratio_up=truncate_ratio_up,numba_acc=numba_acc,parallel_jobs=parallel_jobs)
    def fit(self, X,Y, X_range=None):
        if X_range is None:
            X_range = np.zeros(shape=(2, X.shape[1]))
            X_range[0] = X.min(axis=0)-0.01*(X.max(axis=0)-X.min(axis=0))
            X_range[1] = X.max(axis=0)+0.01*(X.max(axis=0)-X.min(axis=0))
        
        self.X_range_original = X_range
        self.X_range = np.array([np.zeros(X.shape[1]),np.ones(X.shape[1])])
        
        scaled_X = (X -self.X_range[0])/(self.X_range[1]-self.X_range[0])
          
        print(scaled_X)
        
        super(RegressionTree, self).fit(scaled_X,Y,self.X_range)
        
    def predict(self, X):
        scaled_X = (X -self.X_range_original[0])/(self.X_range_original[1]-self.X_range_original[0])
        y_hat = super(RegressionTree, self).predict(scaled_X)
        
        print(y_hat)
        print(scaled_X)
        # check boundary
        check_lowerbound = (scaled_X - 1 >= 0).all(axis=1)
        check_upperbound = (scaled_X - 0 <= 0).all(axis=1)
        is_inboundary = check_lowerbound * check_upperbound
        # assign 0 to points outside the boundary
        y_hat[np.logical_not(is_inboundary)] = 0
        
        return y_hat
    
    
    
    
    
    def get_node_information(self,node_idx,pt_idx):
        if self.estimator == "extrapolation_estimator":
            querying_object=list(self.tree_.leafnode_fun.values())[node_idx]
            return_vec=(self.tree_.node_range[node_idx],
                        querying_object.dt_X,
                        querying_object.dt_Y,
                        querying_object.sorted_ratio,
                        querying_object.sorted_prediction,
                        querying_object.intercept)
        else:
            querying_object=list(self.tree_.leafnode_fun.values())[node_idx]
            X_extra=querying_object.dt_X[pt_idx]
            sorted_ratio, sorted_prediction, intercept=extrapolation_jit_return_info(querying_object.dt_X,
                                                                                     querying_object.dt_Y,
                                                                                     X_extra, querying_object.X_range,
                                                                                     self.order,self.truncate_ratio_low,
                                                                                     self.truncate_ratio_up)
            return_vec=(querying_object.X_range,
                        querying_object.dt_X,
                        querying_object.dt_Y,
                        sorted_ratio,
                        sorted_prediction,
                        intercept)
        return return_vec
    
    
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
        for key in ['min_samples_split',"max_depth","order","truncate_ratio_low","truncate_ratio_up"]:
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

