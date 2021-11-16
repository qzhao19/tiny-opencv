import numpy as np

def BGR2GRAY(img_in):
    img = img_in.astype(np.float)

    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    img_out = 0.2126*r + 0.7152*g + 0.0722*b
    img_out = img_out.astype(np.uint8)

    return img_out

def binarization(img_in, threshold=155):
    img = img_in.copy()
    
    img[img < threshold] = 0
    img[img > threshold] = 255
    
    return img
