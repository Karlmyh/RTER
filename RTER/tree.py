import numpy as np

from ._tree import TreeStruct, RecursiveTreeBuilder
from ._splitter import PurelyRandomSplitter
from ._estimator import NaiveEstimator,ExtrapolationEstimator


SPLITTERS = {"purely": PurelyRandomSplitter}
ESTIMATORS = {"naive_estimator": NaiveEstimator,"extrapolation_estimator":ExtrapolationEstimator}

class BaseRecursiveTree(object):
    def __init__(self, splitter="purely", estimator=None, min_samples_split=2, max_depth=None, order=None, log_Xrange=None, random_state=None):
        self.splitter = splitter
        self.estimator = estimator
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.order=order
    
        self.log_Xrange = log_Xrange
        self.random_state = random_state
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
        builder = RecursiveTreeBuilder(splitter, Estimator, self.min_samples_split, max_depth, order)
        builder.build(self.tree_, X, Y,X_range)
    def apply(self, X):
        return self.tree_.apply(X)
    def predict(self, X):
        return self.tree_.predict(X)


class RegressionTree(BaseRecursiveTree):
    def __init__(self, splitter="purely", estimator="naive_estimator", min_samples_split=2, max_depth=None, order=None, log_Xrange=True, random_state=None):
        super(RegressionTree, self).__init__(splitter=splitter, estimator=estimator, min_samples_split=min_samples_split,order=order, max_depth=max_depth, log_Xrange=log_Xrange, random_state=random_state)
    def fit(self, X,Y, X_range=None):
        if X_range is None:
            X_range = np.zeros(shape=(2, X.shape[1]))
            X_range[0] = X.min(axis=0)
            X_range[1] = X.max(axis=0)
        super(RegressionTree, self).fit(X,Y, X_range)
        self.X_range = X_range
    def predict(self, X):
        y_hat = super(RegressionTree, self).predict(X)
        # check boundary
        check_lowerbound = (X - self.X_range[0] >= 0).all(axis=1)
        check_upperbound = (X - self.X_range[1] <= 0).all(axis=1)
        is_inboundary = check_lowerbound * check_upperbound
        # assign 0 to points outside the boundary
        y_hat[np.logical_not(is_inboundary)] = 0
        return y_hat