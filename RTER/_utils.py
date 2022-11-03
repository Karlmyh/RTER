from numba import njit
import numpy as np

def extrapolation_nonjit(dt_X,dt_Y, X_extra, X_range, order, truncate_ratio_low,truncate_ratio_up):

    ratio_vec=np.array([])
    for idx_X, X in enumerate(dt_X):
        
        centralized=X-X_extra
        
        for d in range(X_extra.shape[0]):
            positive_len=X_range[1,d]-X_extra[d]
            negative_len=X_extra[d]-X_range[0,d]
            
            if centralized[d]>=0:
                centralized[d]/=positive_len
            else:
                centralized[d]/=negative_len
        
        ratio_X= np.abs(centralized).max()
        ratio_vec=np.append(ratio_vec,ratio_X)

    

    idx_sorted_by_ratio=np.argsort(ratio_vec)      
    sorted_ratio = ratio_vec[idx_sorted_by_ratio]
    sorted_y = dt_Y[idx_sorted_by_ratio]

    ratio_mat=np.array([[r**(2*i) for i in range(order+1)] for r in sorted_ratio][int(sorted_ratio.shape[0]*truncate_ratio_low):int(sorted_ratio.shape[0]*truncate_ratio_up)])
    pre_vec=np.array([ sorted_y[:(i+1)].mean()  for i in range(sorted_y.shape[0])][int(sorted_ratio.shape[0]*truncate_ratio_low):int(sorted_ratio.shape[0]*truncate_ratio_up)]).reshape(-1,1)
    
    

    

    return (np.linalg.inv(ratio_mat.T @ ratio_mat) @ ratio_mat.T @ pre_vec)[0].item()

@njit
def extrapolation_jit(dt_X,dt_Y, X_extra, X_range, order, truncate_ratio_low,truncate_ratio_up):
    
    
    n_pts= dt_X.shape[0]
    
    
    ratio_vec=np.zeros(n_pts)
    
    for idx_X, X in enumerate(dt_X):
        
        centralized=X-X_extra
        
        for d in range(X_extra.shape[0]):
            positive_len=X_range[1,d]-X_extra[d]
            negative_len=X_extra[d]-X_range[0,d]
            
            if centralized[d]>=0:
                centralized[d]/=positive_len
            else:
                centralized[d]/=negative_len
        
        ratio_X= np.abs(centralized).max()
        ratio_vec[idx_X]=ratio_X
        


    idx_sorted_by_ratio = np.argsort(ratio_vec)      
    sorted_ratio = ratio_vec[idx_sorted_by_ratio]
    sorted_y = dt_Y[idx_sorted_by_ratio]
    
    
    
    ratio_mat=np.zeros((sorted_ratio.shape[0], order+1))

 
    i=0
    while(i<n_pts):
        r= sorted_ratio[i]
        
        for j in range(order +1):
            ratio_mat[i,j]= r**(2*j) 
            
        i+=1
            
    ratio_mat_used=ratio_mat[int(sorted_ratio.shape[0]*truncate_ratio_low):int(sorted_ratio.shape[0]*truncate_ratio_up)]
    
   
    pre_vec=np.zeros((sorted_y.shape[0],1))
    for k in range(sorted_y.shape[0]):
        pre_vec[k,0]= np.mean(sorted_y[:(k+1)])
    
    pre_vec_used=pre_vec[int(sorted_ratio.shape[0]*truncate_ratio_low):int(sorted_ratio.shape[0]*truncate_ratio_up)]
    

    return (np.linalg.inv(ratio_mat_used.T @ ratio_mat_used) @ ratio_mat_used.T @ pre_vec_used )[0,0]
    
    
    