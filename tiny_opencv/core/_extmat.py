import numpy as np

def _gaussian_fn(x, y, sigma, mu):
    return (1 / (2 * np.pi * np.power(sigma, 2))) * \
        np.exp(-(np.power(x, 2) + np.power(y, 2)) / (2 * np.power(sigma, 2)))


