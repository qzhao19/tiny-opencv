import numpy as np

def _average_pooling(img_in, pool_size=8):
    """
    Parameters
    ----------
    img_path : TYPE
        DESCRIPTION.
    pool_size : TYPE, optional
        DESCRIPTION. The default is 8.

    Returns
    -------

    """
    img_out = img_in.copy()
    
    img_h, img_w, img_c = img_in.shape
    
    N_h, N_w = int(img_h / pool_size), int(img_w / pool_size)
    
    for x in range(N_h):
        for y in range(N_w):
            for c in range(img_c):
                img_out[pool_size * x : pool_size * (x + 1), \
                        pool_size * y : pool_size * (y + 1), c] = np.mean(img_out[pool_size * x : pool_size * (x + 1), \
                                                                                  pool_size * y : pool_size * (y + 1), c]).astype(np.int8)
                    
    return img_out

def _max_pooling(img_in, pool_size=8):
    """
        
    Parameters
    ----------
    img_path : TYPE
        DESCRIPTION.
    pool_size : TYPE, optional
        DESCRIPTION. The default is 8.

    Returns
    -------
    None.
    """
    
    img_out = img_in.copy()
    img_h, img_w, img_c = img_in.shape
    
    N_h, N_w = int(img_h / pool_size), int(img_w / pool_size)
    
    for x in range(N_h):
        for y in range(N_w):
            for c in range(img_c):
                img_out[pool_size * x : pool_size * (x + 1), 
                        pool_size * y : pool_size * (y + 1), c] = np.max(img_out[pool_size * x : pool_size * (x + 1), 
                                                                                  pool_size * y : pool_size * (y + 1), c]).astype(np.int8)       
    return img_out

