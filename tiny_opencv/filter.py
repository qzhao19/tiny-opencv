
import numbers
import numpy as np
from ../core/_extmath import gaussian_fn


class Filter(object):
    def __init__(self, ksize=3, sigma=1.5):
        self._ksize = ksize
        self._sigma = sigma
       
    def _gaussian_kernel(self):
        """Return a 2d-array of shape [ksize, ksize]
        """
        ksize = self._ksize
        pad =  ksize // 2
        kernel = np.zeros((ksize, ksize))

        for y in range(-pad, -pad + ksize):
            for x in range(-pad, -pad + ksize):
                kernel[y + pad, x + pad] = gaussian_fn(x, y)
        
        kernel /= kernel.sum()
        self._kernel = kernel


    def _mean_filter(self, img_in):
        """Mean image filter
        Parameters
        ----------
            img : float32 ndarray of shape [height, width, channel]
                input image.
            ksize : int
                kernel size.
        Returns
        -------
            ndarray of shape [h, w, c].
        """
        
        # img_in = img_in.astype(np.float32)
        
        img_h, img_w, img_c = img_in.shape 
        
        pad = self._ksize // 2
        img_out = np.zeros((img_h + pad * 2, img_w + pad * 2, img_c), 
                        dtype=np.float32)
        
        img_out[pad : pad + img_h, pad : pad + img_w, :] = img_in.copy()
        
        tmp = img_out.copy()
        
        for h in range(img_h):
            for w in range(img_w):
                for c in range(img_c):
                    img_out[h + pad, w + pad, c] = np.mean(tmp[h : h + self._ksize, w : w + self._ksize, c])
        
        
        img_out = np.clip(img_out, 0, 255)
        img_out = img_out[pad : pad + img_h, pad : pad + img_w, :].astype(np.float32)
        
        return img_out

    def _median_filter(self, img_in):
        """
        Parameters
        ----------
        img : ndarray of shape [height, width, channel]
            input image.

        Returns
        -------
        None.

        """
        
        ksize = self._ksize
        img_h, img_w, img_c = img_in.shape
    
        # zero padding, get padding size
        pad = ksize // 2
        img_out = np.zeros((img_h + pad * 2, img_w + pad * 2, img_c), 
                           dtype=np.float32)
        
        img_out[pad : pad + img_h, pad : pad + img_w, :] = img_in.copy()
        
         # temp image 
        tmp = img_out.copy()
        for y in range(img_h):
            for x in range(img_w):
                for z in range(img_c):
                    img_out[y + pad, x + pad, z] = np.median(tmp[y : y + ksize, x : x + ksize, z])
    
        img_out = np.clip(img_out, 0, 255)
        img_out = img_out[pad : pad + img_h, pad : pad + img_w, :].astype(np.float32) 
        
        return img_out

    def _motion_filter(self, img_in):
        """
        Parameters
        ----------
        img_in : float32 ndarray of shape [height, width, channel]
            input image.

        Returns
        -------
            None.
        """
        
        pad = self.ksize // 2
        
        img_h, img_w, img_c = img_in.shape
        
        img_out = np.zeros((img_h + pad * 2, img_w + pad * 2, img_c), 
                           dtype=np.float32)
        img_out[pad : img_h + pad, pad: img_w + pad, :] = img_in.copy()
        tmp = img_out.copy()
        
        for h in range(img_h):
            for w in range(img_w):
                for c in range(img_c):
                    img_out[h + pad, w + pad, c] = \
                    np.sum(self.kernel * tmp[h : h + self.ksize, w : w + self.ksize, :])
        
        
        img_out = np.clip(img_out, 0, 255)
        img_out = img_out[pad : pad + img_h, 
                          pad : pad + img_w, :].astype(np.float32)
        
        return img_out
