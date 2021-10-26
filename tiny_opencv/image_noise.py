import numbers

import cv2
import numpy as np

def _face_blur(image, blocks=5):
    """[summary]

    Args:
        image ([type]): [description]
        blocks (int, optional): [description]. Defaults to 5.

    Returns:
        [type]: [description]
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

def _gaussian_noise(img_in, snr_db = 3):
    """generate gaussain noise and add these into source image

    Args:
        img_in (ndarray of shape): input image
        snr_db (int, optional): 
            rsb: SNR(db) signal-to-noise ration be expressed 
            in decibels, that compares the level of a desired signal 
            to the level of background noise. The default is 3.

    Returns:
        [ndarray of shape]: [output image with gaussian noise]
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
