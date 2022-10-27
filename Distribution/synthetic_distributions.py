from .distributions import (LaplaceDistribution, 
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
import numpy as np
import math




class TestDistribution(object):
    def __init__(self,index,dim):
        self.dim=dim
        self.index=index
        
    def testDistribution_1(self,dim):
        return MultivariateNormalDistribution(mean=np.zeros(dim),cov=np.diag(np.ones(dim)))
    
    def testDistribution_2(self,dim):
        density1 = MultivariateNormalDistribution(mean=np.zeros(dim)+1.5,cov=np.diag(np.ones(dim)*0.05)) 
        density2 = MultivariateNormalDistribution(mean=np.zeros(dim)-1.5,cov=np.diag(np.ones(dim)*0.3)) 
        
    
        density_seq = [density1, density2]
        prob_seq = [0.4,0.6]
        densitymix = MixedDistribution(density_seq, prob_seq)
        return densitymix
    
    def testDistribution_3(self,dim):
        return LaplaceDistribution(scale=np.ones(dim)*0.1,loc=np.zeros(dim)) 
    
    def testDistribution_4(self,dim):
        return BetaDistribution(a=np.ones(dim)*2,b=np.ones(dim)*2) 
    
    
    def testDistribution_5(self,dim):
        density1 = DeltaDistribution(point=np.ones(dim)*0) 
        density2 = MultivariateNormalDistribution(mean=np.zeros(dim),cov=np.diag(np.ones(dim)))  
    
        density_seq = [density1, density2]
        prob_seq = [2/3,1/3]
        densitymix = MixedDistribution(density_seq, prob_seq)
        return densitymix
    
    def testDistribution_6(self,dim):
        density1 = DeltaDistribution(point=np.ones(dim)*0) 
        density2 = UniformDistribution(low=np.ones(dim)*(-1),upper=np.ones(dim)) 
    
        density_seq = [density1, density2]
        prob_seq = [2/3,1/3]
        densitymix = MixedDistribution(density_seq, prob_seq)
        return densitymix
    
    
    def testDistribution_7(self,dim):
        density1 = LaplaceDistribution(scale=np.ones(1)*0.5,loc=np.zeros(1)) 
        density2 = UniformDistribution(low=np.ones(1)*2,upper=np.ones(1)*4) 
        #density2 = MultivariateNormalDistribution(mean=np.ones(1)*3,cov=0.5*np.diag(np.ones(1)))
        
        density_seq = [density1, density2]
        prob_seq = [1/2,1/2]
        densitymix = MixedDistribution(density_seq, prob_seq)
        marginal_density_vector=[]
        for i in range(dim):
            marginal_density_vector=marginal_density_vector+[densitymix]
        densitymarginal = MarginalDistribution(marginal_density_vector)
        return densitymarginal
    
    def testDistribution_8(self,dim):
        density1 = BetaDistribution(a=np.ones(1)*2,b=np.ones(1)*5)
        density2 = UniformDistribution(low=np.ones(1)*0.6,upper=np.ones(1)*1) 
        #density2 = MultivariateNormalDistribution(mean=np.ones(1)*0.8,cov=0.15*np.diag(np.ones(1)))
        
        density_seq = [density1, density2]
        prob_seq = [0.7,0.3]
        densitymix = MixedDistribution(density_seq, prob_seq)
        marginal_density_vector=[]
        for i in range(dim):
            marginal_density_vector=marginal_density_vector+[densitymix]
        densitymarginal = MarginalDistribution(marginal_density_vector)
        return densitymarginal
    
    def testDistribution_9(self,dim):
        density1 = ExponentialDistribution(lamda=0.5) 
        density2 = UniformDistribution(low=0,upper=5) 
        #density2 = MultivariateNormalDistribution(mean=0,cov=0.5)
        
        density_seq=[]
        for i in range(dim-1):
            density_seq = density_seq+[density1]
        density_seq=density_seq+[density2]
        
    
        densitymarginal = MarginalDistribution(density_seq)
        return densitymarginal
    
    def testDistribution_10(self,dim):
        assert dim==2
        density1=UniformCircleDistribution(radius=1)
        density2= UniformDistribution(low=np.zeros(dim)*(-2),upper=np.ones(dim)*2) 
        density_seq=[density1, density2]
        prob_seq = [1/2,1/2]
        densitymix = MixedDistribution(density_seq, prob_seq)
        return densitymix
    
    def testDistribution_11(self,dim):
        return CosineDistribution(loc=np.zeros(dim),scale=np.ones(dim)) 
    
    def testDistribution_12(self,dim):
        return CauchyDistribution(loc=np.zeros(dim),scale=np.ones(dim)*0.1)
    def testDistribution_13(self,dim):
        return TDistribution(loc=np.zeros(dim),scale=np.ones(dim)*0.1,df=2/3)
    def testDistribution_14(self,dim):
        return TDistribution(loc=np.zeros(dim),scale=np.ones(dim)*0.01,df=0.2)
    
    def testDistribution_15(self,dim):
        return UniformDistribution(low=np.zeros(dim),upper=np.ones(dim)) 
    def testDistribution_16(self,dim):
        assert dim==2
        density1 = UniformDistribution(low=np.ones(dim)*(-1.8),upper=np.ones(dim)*(-0.2)) 
        density2 = UniformDistribution(low=np.ones(dim)*(0.2),upper=np.ones(dim)*(1.8)) 
        density3=CosineDistribution(loc=np.array([-1,1]),scale=np.ones(dim)*(1/1.2/math.pi))
        density4=CosineDistribution(loc=np.array([1,-1]),scale=np.ones(dim)*(1/1.2/math.pi))
    
        density_seq = [density1, density2,density3,density4]
        prob_seq = [1/4,1/4,1/4,1/4]
        densitymix = MixedDistribution(density_seq, prob_seq)
        return densitymix
    
    def testDistribution_17(self,dim):
        density1 = MultivariateNormalDistribution(mean=np.zeros(dim)+1.5,cov=np.diag(np.ones(dim)*0.4)) 
        density2 = MultivariateNormalDistribution(mean=np.zeros(dim)-1.5,cov=np.diag(np.ones(dim)*0.7)) 
        
    
        density_seq = [density1, density2]
        prob_seq = [0.4,0.6]
        densitymix = MixedDistribution(density_seq, prob_seq)
        return densitymix
    
    def testDistribution_18(self,dim):
        return CauchyDistribution(loc=np.zeros(dim),scale=np.ones(dim))
    def testDistribution_19(self,dim):
        return TDistribution(loc=np.zeros(dim),scale=np.ones(dim)*0.1,df=2/3)
    
    
    def returnDistribution(self):
        switch = {'1': self.testDistribution_1,                
          '2': self.testDistribution_2,
          '3': self.testDistribution_3,
          '4': self.testDistribution_4,
          '5': self.testDistribution_5,
          '6': self.testDistribution_6,
          '7': self.testDistribution_7,
          '8': self.testDistribution_8,
          '9': self.testDistribution_9,
          '10':self.testDistribution_10,
          '11':self.testDistribution_11,
          '12':self.testDistribution_12,
          '13':self.testDistribution_13,
          '14':self.testDistribution_14,
          '15':self.testDistribution_15,
          '16':self.testDistribution_16,
          '17':self.testDistribution_17,
          '18':self.testDistribution_18,
          '19':self.testDistribution_19,
          }

        choice = str(self.index)  
        #print(switch.get(choice))                # 获取选择
        result=switch.get(choice)(self.dim)
        return result
    
