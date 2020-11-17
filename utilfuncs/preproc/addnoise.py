import numpy as np
import cv2

def add_gaussian_noise(image,mu=0):
    image_2d = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h,w = image_2d.shape
    sig2s = np.logspace(0,2,5) # evenly distributed in log-space
    
    # adding noise
    noisy_image = image_2d
    for sig2 in sig2s:
        sig = sig2**0.5
        gauss = np.random.normal(mu,sig,(h,w))

        noisy_image = noisy_image + gauss
    
        # clipping between 0 and 255
        nois_image = np.clip(noisy_image, 0, 255)

    return noisy_image


def add_saltnpepper(image,s=0.5,p=0.5,amt=0.005):
    assert s+p == 1.0
    assert amt < 1
    
    image_2d = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h,w = image_2d.shape
    noisy_img = image_2d.copy()
    for _ in range(15):
        
        # add salt
        salt = np.ceil(amt * image_2d.size * s)
        coords = [np.random.randint(0,i-1,int(salt)) for i in noisy_img.shape] # shape (2,int(salt))
        noisy_img[coords] = 255

        # add pepper
        pepper = np.ceil(amt * image_2d.size * p)
        coords = [np.random.randint(0,i-1,int(pepper)) for i in noisy_img.shape] # shape (2,int(pepper))
        noisy_img[coords] = 0

    return noisy_img