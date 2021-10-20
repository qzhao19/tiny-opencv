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
