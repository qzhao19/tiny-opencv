import numpy as np

def bgr2rgb(img_in):
    """
    """
    img_out = img_in.copy()    

    b = img_out[:, :, 0].copy()
    g = img_out[:, :, 1].copy()
    r = img_out[:, :, 2].copy()

    # BGR => RGB
    img_out[:, :, 0] = r
    img_out[:, :, 1] = g
    img_out[:, :, 2] = b

    return img_out

def bgr2gray(img_in):
    img_out = img_in.astype(np.float)

    b = img_out[:, :, 0].copy()
    g = img_out[:, :, 1].copy()
    r = img_out[:, :, 2].copy()

    img_out = 0.2126*r + 0.7152*g + 0.0722*b
    img_out = img_out.astype(np.uint8)

    return img_out

def bgr2hsv(img_in):
    """Convert bgr image to hsv image
    """
    if not isinstance(img_in, np.float32):
        img_in = img_in.astype(np.float32)
    
    rgb = img_in.copy() / 255.
    
    hsv = np.zeros_like(rgb, dtype=np.float32)
    
    # get max et min value
    max_vals = np.max(rgb, axis=2).copy()
    min_vals = np.min(rgb, axis=2).copy()
    min_args = np.argmin(rgb, axis=2)
    
    # h
    hsv[:, :, 0][np.where(max_vals == min_vals)] = 0
    
    ## if min_vals = B
    indices = np.where(min_args == 0)
    hsv[:, :, 0][indices] = 60 * (rgb[:, :, 1][indices] - rgb[:, :, 2][indices]) / (max_vals[indices] - min_vals[indices]) + 60

    ## if min = G
    indices = np.where(min_args == 1)
    hsv[:, :, 0][indices] = 300 * (rgb[:, :, 2][indices] - rgb[:, :, 0][indices]) / (max_vals[indices] - min_vals[indices]) + 300
    
    ## if min = R
    indices = np.where(min_args == 2)
    hsv[:, :, 0][indices] = 180 * (rgb[:, :, 0][indices] - rgb[:, :, 1][indices]) / (max_vals[indices] - min_vals[indices]) + 180
    
    # S
    hsv[:, :, 1] = max_vals.copy() - min_vals.copy()
    
    # V
    hsv[:, :, 2] = max_vals.copy()
    
    return hsv

def hvs2bgr(img_in, hsv):
    """
    """
    img = img_in.copy() / 255.

	# get max and min
    max_v = np.max(img, axis=2).copy()
    min_v = np.min(img, axis=2).copy()
    img_out = np.zeros_like(img)

    H = hsv[..., 0]
    S = hsv[..., 1]
    V = hsv[..., 2]

    C = S
    H_ = H / 60.
    X = C * (1 - np.abs( H_ % 2 - 1))
    Z = np.zeros_like(H)
    vals = [[Z,X,C], [Z,C,X], [X,C,Z], [C,X,Z], [C,Z,X], [X,Z,C]]

	
    for i in range(6):
        ind = np.where((i <= H_) & (H_ < (i+1)))
        img_out[..., 0][ind] = (V - C)[ind] + vals[i][0][ind]
        img_out[..., 1][ind] = (V - C)[ind] + vals[i][1][ind]
        img_out[..., 2][ind] = (V - C)[ind] + vals[i][2][ind]

	
    img_out[np.where(max_v == min_v)] = 0
    img_out = np.clip(img_out, 0, 1)
    img_out = (img_out * 255).astype(np.uint8)

    return img_out

def discretization(img_in):
    """Image discretization of color
    """   
    img_out = img_in.copy()
    for i in range(4):
        idx = np.where(((64 * i - 1) <= img_out) & ((64 * (i + 1) - 1)))
        img_out[idx] = 32 * (2 * 1 + 1)
        
    return img_out

def binarization(img_in, threshold=155):
    img_out = img_in.copy()
    
    img_out[img_out < threshold] = 0
    img_out[img_out > threshold] = 255
    
    return img_out
