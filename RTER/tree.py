import numpy as np

from ._tree import TreeStruct, RecursiveTreeBuilder
from ._splitter import PurelyRandomSplitter,MidPointRandomSplitter
from ._estimator import NaiveEstimator,ExtrapolationEstimator,PointwiseExtrapolationEstimator


SPLITTERS = {"purely": PurelyRandomSplitter,"midpoint":MidPointRandomSplitter}
ESTIMATORS = {"naive_estimator": NaiveEstimator,"extrapolation_estimator":ExtrapolationEstimator,"pointwise_extrapolation_estimator":PointwiseExtrapolationEstimator}

class BaseRecursiveTree(object):
    def __init__(self, 
                 splitter="purely", 
                 estimator=None, 
                 min_samples_split=2, 
                 max_depth=None, 
                 order=None, 
                 log_Xrange=None, 
                 random_state=None,
                 polynomial_output=0,
                 truncate_ratio_low=0.55,
                 truncate_ratio_up=0.2,
                 numba_acc=0):
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
    def predict(self, X,numba_acc):
        return self.tree_.predict(X, numba_acc)


class RegressionTree(BaseRecursiveTree):
    def __init__(self, splitter="purely", estimator="naive_estimator", min_samples_split=2, max_depth=None, order=None, log_Xrange=True, random_state=None,polynomial_output=None, truncate_ratio_low=None , truncate_ratio_up=None,numba_acc=0):
        super(RegressionTree, self).__init__(splitter=splitter, estimator=estimator, min_samples_split=min_samples_split,order=order, max_depth=max_depth, log_Xrange=log_Xrange, random_state=random_state,polynomial_output=polynomial_output,truncate_ratio_low=truncate_ratio_low,truncate_ratio_up=truncate_ratio_up)
    def fit(self, X,Y, X_range=None):
        if X_range is None:
            X_range = np.zeros(shape=(2, X.shape[1]))
            X_range[0] = X.min(axis=0)-0.01*(X.max(axis=0)-X.min(axis=0))
            X_range[1] = X.max(axis=0)+0.01*(X.max(axis=0)-X.min(axis=0))
        super(RegressionTree, self).fit(X,Y, X_range)
        self.X_range = X_range
    def predict(self, X, numba_acc=0):
        y_hat = super(RegressionTree, self).predict(X,numba_acc)
        # check boundary
        check_lowerbound = (X - self.X_range[0] >= 0).all(axis=1)
        check_upperbound = (X - self.X_range[1] <= 0).all(axis=1)
        is_inboundary = check_lowerbound * check_upperbound
        # assign 0 to points outside the boundary
        y_hat[np.logical_not(is_inboundary)] = 0
        return y_hat
    
    
    def get_node_information(self,node_idx):
        querying_object=list(self.tree_.leafnode_fun.values())[node_idx]
        return_vec=(self.tree_.node_range[node_idx],
                    querying_object.dt_X,
                    querying_object.dt_Y,
                    querying_object.sorted_ratio,
                    querying_object.sorted_prediction,
                    querying_object.intercept)
        return return_vec
    
    
    ## node x range dt_X,dt_Y, sorted_ratio sorted_y sorted_prediction intercept
        