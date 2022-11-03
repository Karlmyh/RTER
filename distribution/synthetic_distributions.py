from .marginal_distributions import (LaplaceDistribution, 
                          BetaDistribution,
                          DeltaDistribution,
                          MultivariateNormalDistribution,
                          UniformDistribution,
                          MarginalDistribution,
                          ExponentialDistribution,
                          MixedDistribution,
                          UniformCircleDistribution,
                          CauchyDistribution,
                          CosineDistribution,
                          TDistribution
                          )

from .regression_function import RegressionFunction

from .noise_distributions import GaussianNoise

import numpy as np
import math




class TestDistribution(object):
    def __init__(self,index,dim):
        self.dim=dim
        self.index=index
        
    def testDistribution_1(self,dim):
        return MultivariateNormalDistribution(mean=np.zeros(dim),cov=np.diag(np.ones(dim)))
    
   
    
    def returnDistribution(self):
        switch = {'1': self.testDistribution_1,                
          
          }

        choice = str(self.index)  
        #print(switch.get(choice))                # 获取选择
        result=switch.get(choice)(self.dim)
        return result
    
