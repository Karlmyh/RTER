import numpy as np

from ._tree import TreeStruct, RecursiveTreeBuilder
from ._splitter import PurelyRandomSplitter
from ._estimator import DensityEstimator,ExtrapolationEstimator


SPLITTERS = {"purely": PurelyRandomSplitter}
ESTIMATORS = {"density_estimator": DensityEstimator,"extrapolation_estimator":ExtrapolationEstimator}

class BaseRecursiveTree(object):
    def __init__(self, splitter="purely", estimator=None, min_samples_split=2, max_depth=None, order=None, log_Xrange=None, random_state=None):
        self.splitter = splitter
        self.estimator = estimator
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.order=order
    
        self.log_Xrange = log_Xrange
        self.random_state = random_state
    def fit(self, X, X_range=None):
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
        builder.build(self.tree_, X, X_range)
    def apply(self, X):
        return self.tree_.apply(X)
    def predict(self, X):
        return self.tree_.predict(X)


class DensityTree(BaseRecursiveTree):
    def __init__(self, splitter="purely", estimator="density_estimator", min_samples_split=2, max_depth=None, order=None, log_Xrange=True, random_state=None):
        super(DensityTree, self).__init__(splitter=splitter, estimator=estimator, min_samples_split=min_samples_split,order=order, max_depth=max_depth, log_Xrange=log_Xrange, random_state=random_state)
    def fit(self, X, X_range=None):
        if X_range is None:
            X_range = np.zeros(shape=(2, X.shape[1]))
            X_range[0] = X.min(axis=0)
            X_range[1] = X.max(axis=0)
        super(DensityTree, self).fit(X, X_range)
        self.X_range = X_range
    def predict(self, X):
        pdf = super(DensityTree, self).predict(X)
        # check boundary
        check_lowerbound = (X - self.X_range[0] >= 0).all(axis=1)
        check_upperbound = (X - self.X_range[1] <= 0).all(axis=1)
        is_inboundary = check_lowerbound * check_upperbound
        # assign 0 to points outside the boundary
        pdf[np.logical_not(is_inboundary)] = 0
        return pdf