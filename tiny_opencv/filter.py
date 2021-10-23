import cv2
import numbers
import numpy as np



class GaussianFilter(object):
    """
    """
    def __init__(self, ksize=3, sigma=1.5):
        self._ksize = ksize
        self._sigma = sigma

    def _gaussian_fn(self, x, y):
        """Return a 2d gaussian distribution with mu = 0 and sigma = sigma 

        Parameters:
        ----------
            x: int
                The height of input image 
            y: int
                The width of input image 
            
        """
        return (1 / (2 * np.pi * np.power(self._sigma, 2))) * \
            np.exp(-(np.power(x, 2) + np.power(y, 2)) / (2 * np.power(self._sigma, 2)))

    def _gaussian_kernel(self):
        """Return a 2d-array of shape [ksize, ksize]
        """
        ksize = self._ksize
        pad =  ksize // 2
        kernel = np.zeros((ksize, ksize))

        for y in range(-pad, -pad + ksize):
            for x in range(-pad, -pad + ksize):
                kernel[y + pad, x + pad] = self._gaussian_fn(x, y)
        
        kernel /= kernel.sum()
        self._kernel = kernel

    def _gaussian_filter(self, X):
        """Compute gaussian filter

        Parameters
        ----------
            X : np.ndarray of shape [height, width, channel]
                input image.
        Returns
        -------
            self class.
        """

        
        # -*- coding: utf-8 -*-


    def _mean_filter(self, img_in, ksize=3):
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

        pad = ksize // 2

        img_out = np.zeros((img_h + pad * 2, img_w + pad * 2, img_c), 
                           dtype=np.float32)

        img_out[pad : pad + img_h, pad : pad + img_w, :] = img_in.copy()

        tmp = img_out.copy()

        for h in range(img_h):
            for w in range(img_w):
                for c in range(img_c):
                    img_out[h + pad, w + pad, c] = np.mean(tmp[h : h + ksize, w : w + ksize, c])


        img_out = np.clip(img_out, 0, 255)
        img_out = img_out[pad : pad + img_h, pad : pad + img_w, :].astype(np.float32)

        return img_out



    def median_filter(self, img_path, ksize=3):
        """


        Parameters
        ----------
            img_path : strings
                input image path.

            ksize : TYPE, optional
                DESCRIPTION. The default is 3.

        Returns
        -------
            None.

        """
        if not isinstance(ksize, numbers.Number):
            raise ValueError('{} is not an Integer')

        else:
            if not isinstance(ksize, int):
                ksize = int(ksize)

        try:
            img_in = cv2.imread(img_path)
        except:
            raise ValueError('Cannot read image from {}, check input path!'.format(img_path))


        if len(img_in.shape) == 1:
            img_in = cv2.cvtColor(img_in, cv2.COLOR_GRAY2RGB)
        elif len(img_in.shape) == 3:
            img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError('The dimensions of image must be one or three '
                             'but got %s' %str(len(img_in.shape)))


        if not isinstance(img_in, np.ndarray):
            img_in = np.asarray(img_in)


        if img_in.dtype.type != np.float32:
            img_in = img_in.astype(np.float32)


        result = _mean_filter(img_in, ksize)

        return result
    

    def _kernel(self):
        """generate motion kernel
        

        Returns
        -------
        None.

        """
        
        kernel = np.diag([1] * self.ksize)
        kernel /= self.ksize
        
        self.kernel = kernel

    def _filter(self, img_in):
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
                    np.sum(self.kernel * tmp[h : h + self.ksize, w : w + sel.ksize, :])
        
        
        img_out = np.clip(img_out, 0, 255)
        img_out = img_out[pad : pad + img_h, 
                          pad : pad + img_w, :].astype(np.float32)
        
        self.img_out = img_out
    
    def fit(self, img_in):
        """
        

        Parameters
        ----------
        img_in : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self._kernel()
        
        self._filter(img_in)
        
        img_out = self.img_out
        
        return img_out
        
