# -*- coding: utf-8 -*-
import numbers
import numpy as np


def face_blur(image, blocks=5):
	(h, w) = image.shape[:2]
	xSteps = np.linspace(0, w, blocks + 1, dtype="int")
	ySteps = np.linspace(0, h, blocks + 1, dtype="int")
	for i in range(1, len(ySteps)):
		for j in range(1, len(xSteps)):
			startX = xSteps[j - 1]
			startY = ySteps[i - 1]
			endX = xSteps[j]
			endY = ySteps[i]
			roi = image[startY:endY, startX:endX]
			(B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
			cv2.rectangle(image, (startX, startY), (endX, endY),(B, G, R), -1)
	return image


def _gaussian_noise(img_in, snr_db=3):
    """Function gaussNoise allows to generate gaussain 
    noise and add these into source image
    
    Parameters
    ----------
        img_in : ndarray of shape 
            input image.
        snr_db : TYPE, optional
            rsb: SNR(db) signal-to-noise ration be expressed 
            in decibels, that compares the level of a desired signal 
            to the level of background noise. The default is 3.

    Returns
    -------
        image noised that.
    """
    
    img_in = img_in.copy()
    
    img_h, img_w, img_c = img_in.shape
    
    img_out = np.zeros((img_h, img_w, img_c), dtype=np.float32)
    
    # calculate source image variance 
    # SNR=var(image)/var(noise)
    # SNR(db)=10*log(SNR) ==> SNR=10**(SNR(db)/10)
    
    img_var = np.var(img_in)
    
    noise_var = img_var / np.power(10, (snr_db / 10))
    # noise_var=img_var / (10 ** (snr_db / 10))
    
    sigma = np.sqrt(noise_var)

    gaussian_noise = np.random.normal(0, sigma, (img_h, img_w))
    
    
    img_out[:, :, 0] = img_in[:, :, 0] + gaussian_noise
    img_out[:, :, 1] = img_in[:, :, 1] + gaussian_noise
    img_out[:, :, 2] = img_in[:, :, 2] + gaussian_noise


    img_out = np.clip(img_out, 0, 255)
    
    img_out = img_out.astype(np.uint8) 
    
    return img_out



def _salt_pepper_noise(img_in, prob):
    """salt&pepper image noise

    Parameters
    ----------
        img_in : ndarray of shape [height, width, channel]
            input image.
        prob : float
            The percentage of noise, it should be between 0 and 1.

    Returns
    -------
        noised image.

    """
    
    img_in = img_in.copy()
    
    img_h, img_w, img_c = img_in.shape
    
    img_out = img_in.copy()
    
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
    """speckly noise
    

    Parameters
    ----------
        img_in : ndarray of shape [height, width, channel]
                input image.

    Returns
    -------
    

    """
    img_h, img_w, img_c = img_in.shape
    
    # a sdantard gaussian distribution
    gaussian_noise = np.random.randn(img_h, img_w, img_c)
    
    img_out = img_in.copy() + img_in.copy() * gaussian_noise
    
    return img_out


def image_noise(img_path, mode='gaussian', snr=3, prob=0.5):
    """
    

    Parameters
    ----------
    img_path : TYPE
        DESCRIPTION.
    mode : TYPE, optional
        DESCRIPTION. The default is 'gaussian'.
    snr : TYPE, optional
        DESCRIPTION. The default is 3.
    prob : TYPE, optional
        DESCRIPTION. The default is 0.5.

    Returns
    -------
    None.

    """
    
    
    if not isinstance(snr, numbers.Number):
        raise ValueError('{} is not an Integer')
        
    else:
        if not isinstance(snr, int):
            snr = int(snr)
    
    










