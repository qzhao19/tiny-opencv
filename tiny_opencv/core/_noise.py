import numbers

import cv2
import numpy as np

def _face_blur(image, blocks=5):
    """[summary]

    Parameters
    ----------
    image : [type]
        [description]
    blocks : int, optional
        [description], by default 5

    Returns
    -------
    [type]
        [description]
    """
    (img_h, img_w) = image.shape[:2]
    x_steps = np.linspace(0, img_w, blocks + 1, dtype="int")
    y_steps = np.linspace(0, img_h, blocks + 1, dtype="int")
    for i in range(1, len(y_steps)):
        for j in range(1, len(x_steps)):
            start_x = x_steps[j - 1]
            start_y = y_steps[i - 1]
            end_x = x_steps[j]
            end_y = y_steps[i]
            roi = image[start_y:end_y, start_x:end_x]
            (B, G, R) = [int(x) for x in np.mean(roi)[:3]]
            cv2.rectangle(image, (start_x, start_y), (end_x, end_y),(B, G, R), -1)
    return image

def _gaussian(img_in, snr_db = 3):
    """generate gaussain noise and add these into source image

    Parameters
    ----------
    img_in : [ndarray of shape 3D]
        input image
    snr_db : int, optional
        [rsb: SNR(db) signal-to-noise ration be expressed 
        in decibels, that compares the level of a desired signal 
        to the level of background noise. The default is 3.], by default 3

    Returns
    -------
    [ndarray of shape 3D]
        [output image with gaussian noise]

    """
    img_h, img_w, img_c = img_in.shape
    img_out = np.zeros((img_h, img_w, img_c), dtype=np.float32)

    # calculate source image variance ==> SNR=var(image)/var(noise)
    # SNR(db)=10*log(SNR) ==> SNR=10**(SNR(db)/10)
    img_var = np.var(img_in)
    noise_var = img_var / np.power(10, (snr_db / 10))

    sigma = np.sqrt(noise_var)
    gaussian_noise = np.random.normal(0, sigma, (img_h, img_w))

    img_out[:, :, 0] = img_in[:, :, 0] + gaussian_noise
    img_out[:, :, 1] = img_in[:, :, 1] + gaussian_noise
    img_out[:, :, 2] = img_in[:, :, 2] + gaussian_noise
    img_out = np.clip(img_out, 0, 255)
    img_out = img_out.astype(np.uint8) 
    
    return img_out


def _salt_pepper(img_in, prob):
    """salt&pepper image noise

    Parameters:
        img_in (ndarray of 3D): [input image]
        prob (float): [The percentage of noise, it should be between 0 and 1.]

    Returns:
        [ndarray of 3D]: [output image with salt pepper noise]
    """
    img_h, img_w, img_c = img_in.shape
    img_out = np.zeros((img_h, img_w, img_c), dtype=np.float32)

    amount = 0.004
    n_salts = np.ceil(img_in.size * amount * prob)
    coords = [np.random.randint(0, i - 1, int(n_salts))
              for i in img_in.shape]
    
    img_out[coords] = 1
    n_peppers = np.ceil(img_in.size * amount * (1 - prob))
    
    coords = [np.random.randint(0, i - 1, int(n_peppers))
              for i in img_in.shape]
    
    img_out[coords] = 0
    img_out = img_out.astype(np.uint8) 
    
    return img_out

def _speckly(img_in):
    """[image speckly noise]

    Parameters
    ----------
    img_in : [ndarray of shape height, width, channel]
        [input image.]

    Returns
    -------
    [type]
        [ndarray of shape]
    """
    img_h, img_w, img_c = img_in.shape
    # a sdantard gaussian distribution
    gaussian_noise = np.random.randn(img_h, img_w, img_c)
    
    img_out = img_in.copy() + img_in.copy() * gaussian_noise
    
    return img_out
