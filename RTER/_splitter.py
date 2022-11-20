import numpy as np
from ._utils import compute_variace_dim

class PurelyRandomSplitter(object):
    def __init__(self, random_state=None):
        self.random_state = random_state
        np.random.seed(self.random_state)
    def __call__(self, X, X_range,dt_Y=None):
        n_node_samples, dim = X.shape
        rd_dim = np.random.randint(0, dim)
        rddim_min = X_range[0, rd_dim]
        rddim_max = X_range[1, rd_dim]
        rd_split = np.random.uniform(rddim_min, rddim_max)
        return rd_dim, rd_split
    
class MidPointRandomSplitter(object):
    def __init__(self, random_state=None):
        self.random_state = random_state
        np.random.seed(self.random_state)
    def __call__(self, X, X_range,dt_Y=None):
        n_node_samples, dim = X.shape
        rd_dim = np.random.randint(0, dim)
        rddim_min = X_range[0, rd_dim]
        rddim_max = X_range[1, rd_dim]
        rd_split = (rddim_min+ rddim_max)/2
        return rd_dim, rd_split
    
    
class MaxEdgeRandomSplitter(object):
    def __init__(self, random_state=None):
        self.random_state = random_state
        np.random.seed(self.random_state)
    def __call__(self, X, X_range,dt_Y=None):
        n_node_samples, dim = X.shape
        edge_ratio= X_range[1]-X_range[0]
        
        rd_dim = np.random.choice(np.where(edge_ratio==edge_ratio.max())[0])
        #rd_dim = np.random.randint(0, dim)
        rddim_min = X_range[0, rd_dim]
        rddim_max = X_range[1, rd_dim]
        rd_split = (rddim_min+ rddim_max)/2
        return rd_dim, rd_split
    
    
class VarianceReductionSplitter(object):
    def __init__(self, random_state=None):
        self.random_state = random_state
        np.random.seed(self.random_state)
    def __call__(self, X, X_range, dt_Y ):
        n_node_samples, dim = X.shape
        
        
        max_mse = np.inf
        split_dim = None
        split_point = None
        
        for d in range(dim):
            
            check_mse, check_split_point = compute_variace_dim(X[:,d],dt_Y)
            
            if check_mse < max_mse:
              
                max_mse = check_mse
                split_dim = d
                split_point = check_split_point
                

        return split_dim, split_point