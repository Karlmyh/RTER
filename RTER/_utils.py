from numba import njit
import numpy as np

def assign_parallel_jobs(input_tuple):
    idx, node_object, X, numba_acc =input_tuple
    return idx, node_object.predict(X, numba_acc=numba_acc)



def extrapolation_nonjit(dt_X,dt_Y, X_extra, X_range, order, truncate_ratio_low,truncate_ratio_up,r_range_low,r_range_up,step,lamda):
    
    radius = X_range[1,0]- X_range[0,0]
    ratio_vec=np.array([])
    for idx_X, X in enumerate(dt_X):
        
        centralized=X-X_extra
        
        for d in range(X_extra.shape[0]):
            positive_len=X_range[1,d]-X_extra[d]
            negative_len=X_extra[d]-X_range[0,d]
            
            if centralized[d]>0:
                centralized[d]/=positive_len
          
            elif centralized[d]<0:
                centralized[d]/=negative_len
        
        ratio_X= np.abs(centralized).max() *radius
        ratio_vec=np.append(ratio_vec,ratio_X)

    

    idx_sorted_by_ratio=np.argsort(ratio_vec)      
    sorted_ratio = ratio_vec[idx_sorted_by_ratio]
    sorted_y = dt_Y[idx_sorted_by_ratio]

    ratio_mat=np.array([[r**i for i in range(order+1)] for r in sorted_ratio][int(sorted_ratio.shape[0]*truncate_ratio_low):int(sorted_ratio.shape[0]*truncate_ratio_up):step])
    pre_vec=np.array([ sorted_y[:(i+1)].mean()  for i in range(sorted_y.shape[0])][int(sorted_ratio.shape[0]*truncate_ratio_low):int(sorted_ratio.shape[0]*truncate_ratio_up):step]).reshape(-1,1)
    
    if order != 0:
        ratio_range_idx_up = ratio_mat[:,1]< r_range_up
        ratio_range_idx_low  = ratio_mat[:,1]> r_range_low
        ratio_range_idx = ratio_range_idx_up*ratio_range_idx_low
        ratio_mat=ratio_mat[ratio_range_idx]
        pre_vec=pre_vec[ratio_range_idx]
        
  
    
    id_matrix = np.eye(ratio_mat.shape[1])
    #id_matrix[0,0] = 0

    

    return (np.linalg.inv(ratio_mat.T @ ratio_mat+ id_matrix*lamda) @ ratio_mat.T @ pre_vec)[0].item()

@njit
def extrapolation_jit(dt_X,dt_Y, X_extra, X_range, order, truncate_ratio_low, truncate_ratio_up, r_range_low, r_range_up, step, lamda):
    


    radius = X_range[1,0]- X_range[0,0]
    n_pts= dt_X.shape[0]
    
    
    ratio_vec=np.zeros(n_pts)
    
    for idx_X, X in enumerate(dt_X):
        
        centralized=X-X_extra
        
        for d in range(X_extra.shape[0]):
            positive_len=X_range[1,d]-X_extra[d]
            negative_len=X_extra[d]-X_range[0,d]
            
            if centralized[d]>0:
                
                centralized[d]/=positive_len
            elif centralized[d]<0:
                
                centralized[d]/=negative_len
        
        ratio_X= np.abs(centralized).max()*radius
        ratio_vec[idx_X]=ratio_X
        


    idx_sorted_by_ratio = np.argsort(ratio_vec)      
    sorted_ratio = ratio_vec[idx_sorted_by_ratio]
    sorted_y = dt_Y[idx_sorted_by_ratio]
    
    
    
    ratio_mat=np.zeros((sorted_ratio.shape[0], order+1))

 
    i=0
    while(i<n_pts):
        r= sorted_ratio[i]
        
        for j in range(order +1):
            ratio_mat[i,j]= r**j 
            
        i+=1
            
    ratio_mat_used=ratio_mat[int(sorted_ratio.shape[0]*truncate_ratio_low):int(sorted_ratio.shape[0]*truncate_ratio_up):step]
    

    
   
    pre_vec=np.zeros((sorted_y.shape[0],1))
    for k in range(sorted_y.shape[0]):
        pre_vec[k,0]= np.mean(sorted_y[:(k+1)])
    
    
    pre_vec_used=pre_vec[int(sorted_ratio.shape[0]*truncate_ratio_low):int(sorted_ratio.shape[0]*truncate_ratio_up):step]
    
    if order != 0:
        
        ratio_range_idx_up = ratio_mat_used[:,1]< r_range_up
        ratio_range_idx_low  = ratio_mat_used[:,1]> r_range_low
        ratio_range_idx = ratio_range_idx_up*ratio_range_idx_low
        ratio_mat_final=ratio_mat_used[ratio_range_idx]
        pre_vec_final=pre_vec_used[ratio_range_idx]
        
    else:
        
        ratio_mat_final=ratio_mat_used
        pre_vec_final=pre_vec_used
    
    id_matrix = np.eye(ratio_mat_final.shape[1])
    #id_matrix[0,0] = 0
    
    ratio_mat_final_T = np.ascontiguousarray(ratio_mat_final.T)
    ratio_mat_final = np.ascontiguousarray(ratio_mat_final)
    RTR = np.ascontiguousarray(ratio_mat_final_T @ ratio_mat_final+ id_matrix*lamda)
    RTR_inv = np.ascontiguousarray(np.linalg.inv(RTR))
    pre_vec_final = np.ascontiguousarray(pre_vec_final)
    

    return (RTR_inv @ ratio_mat_final_T @ pre_vec_final )[0,0]
   
    
    
@njit
def extrapolation_jit_return_info(dt_X,dt_Y, X_extra, X_range, order, truncate_ratio_low,truncate_ratio_up,r_range_low,r_range_up,step,lamda):
    radius = X_range[1,0]- X_range[0,0]
    
    n_pts= dt_X.shape[0]
    
    
    ratio_vec=np.zeros(n_pts)
    
    for idx_X, X in enumerate(dt_X):
        
        centralized=X-X_extra
        
        #print("group")
        #print(X_extra)
        #print(X_range)
        #print(dt_X)
        
        
        for d in range(X_extra.shape[0]):
            positive_len=X_range[1,d]-X_extra[d]
            negative_len=X_extra[d]-X_range[0,d]
            
            if centralized[d]>0:
                #print((centralized[d],positive_len))
                centralized[d]/=positive_len
            elif centralized[d]<0:
                #print((centralized[d],negative_len))
                centralized[d]/=negative_len
        
        
            
        ratio_X= np.abs(centralized).max()*radius
        #if ratio_X>5:
            #print(centralized)
        
        ratio_vec[idx_X]=ratio_X
        
        

    
    idx_sorted_by_ratio = np.argsort(ratio_vec)      
    sorted_ratio = ratio_vec[idx_sorted_by_ratio]
    sorted_y = dt_Y[idx_sorted_by_ratio]
    
    
    
    ratio_mat=np.zeros((sorted_ratio.shape[0], order+1))

 
    i=0
    while(i<n_pts):
        r= sorted_ratio[i]
        
        for j in range(order +1):
            ratio_mat[i,j]= r**j 
            
        i+=1
            
    ratio_mat_used=ratio_mat[int(sorted_ratio.shape[0]*truncate_ratio_low):int(sorted_ratio.shape[0]*truncate_ratio_up):step]
    
   
    pre_vec=np.zeros((sorted_y.shape[0],1))
    for k in range(sorted_y.shape[0]):
        pre_vec[k,0]= np.mean(sorted_y[:(k+1)])
    
    pre_vec_used=pre_vec[int(sorted_ratio.shape[0]*truncate_ratio_low):int(sorted_ratio.shape[0]*truncate_ratio_up):step]
    
    if order != 0:
        
        ratio_range_idx_up = ratio_mat_used[:,1]< r_range_up
        ratio_range_idx_low  = ratio_mat_used[:,1]> r_range_low
        ratio_range_idx = ratio_range_idx_up*ratio_range_idx_low
        ratio_mat_final=ratio_mat_used[ratio_range_idx]
        pre_vec_final=pre_vec_used[ratio_range_idx]
        
    else:
        
        ratio_mat_final=ratio_mat_used
        pre_vec_final=pre_vec_used
    

    id_matrix = np.eye(ratio_mat_final.shape[1])
    #id_matrix[0,0] = 0

    return sorted_ratio, pre_vec,  (np.linalg.inv(ratio_mat_final.T @ ratio_mat_final+ id_matrix*lamda) @ ratio_mat_final.T @ pre_vec_final )[0,0], ratio_mat_final @ np.linalg.inv(ratio_mat_final.T @ ratio_mat_final) @ ratio_mat_final.T  
    
    
    
    
  
@njit    
def insample_ssq(y):
    return np.var(y)*len(y)
    
    
@njit  
def compute_variace_dim(dt_X_dim, dt_Y):
    
    dt_X_dim_unique = np.unique(dt_X_dim)
    
    if len(dt_X_dim_unique) == 1:
        return np.inf, 0
    
    sorted_X = np.sort(dt_X_dim_unique) 
    
    num_unique_samples = len(sorted_X)
    
    split_point = 0
    mse = np.inf
    
    for i in range(num_unique_samples- 1):
        
        check_split = (sorted_X[i]+sorted_X[i+1])/2
        
        if len(dt_Y[dt_X_dim<check_split])==0:
            print(dt_Y)
            print((dt_X_dim<check_split).sum())
            print("check dtY")
        check_mse = insample_ssq(dt_Y[dt_X_dim<check_split])+insample_ssq(dt_Y[dt_X_dim>=check_split])
    
        if check_mse < mse:
            split_point = check_split
            mse = check_mse
    
    return mse, split_point
