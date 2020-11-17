import os, cv2
import numpy as np
from PIL import Image
from utilfuncs.preproc.addnoise import add_gaussian_noise, add_saltnpepper

def open_image_arr(path):
    return np.asarray(Image.open(path), dtype='int64')

def make_gray_dir(gray_path):
    if not os.path.isdir(gray_path):
        os.mkdir(gray_path)

def save_gray(gray_path, data_path):
    if not os.path.isdir(gray_path):
        image_names = os.listdir(data_path)

        for im in image_names:
            img = cv2.imread(os.path.join(data_path, im))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            im_name, _ = im.split('.')
            cv2.imwrite(os.path.join(gray_path, im_name+".jpg"), gray)

        print('Successfully saved all images')


def make_noise_dirs(noisy_dir):
    gauss_path = snp_path = None
    if not os.path.isdir(noisy_dir):
        os.mkdir(noisy_dir)
    if not os.path.isdir(os.path.join(noisy_dir, "gauss")):
        gauss_path = os.mkdir(os.path.join(noisy_dir, "gauss"))
    if not os.path.isdir(os.path.join(noisy_dir, "saltnpepp")):
        snp_path = os.mkdir(os.path.join(noisy_dir, "saltnpepp"))
    
    return gauss_path, snp_path

def add_noise_save(noisy_dir, gray_path, gauss_path, snp_path):
    if not os.path.isdir(noisy_dir):    
        gray_images = os.listdir(gray_path)
        
        for im in gray_images:
            img = cv2.imread(os.path.join(gray_path, im))
            img = add_gaussian_noise(img)
            cv2.imwrite(os.path.join(gauss_path, im), img)

        print('Successfully added Gaussian noise to all grayscale images')
        print('Saved to: ./noisy/gauss')

        for im in gray_images:
            img = cv2.imread(os.path.join(gray_path, im))
            img = add_saltnpepper(img)
            cv2.imwrite(os.path.join(snp_path, im), img)

        print('Successfully added Salt and Pepper noise to all grayscale images')
        print('Saved to: ./noisy/saltnpepp')