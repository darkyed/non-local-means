import numpy as np

def get_mse(noisy, orig):
    # mean suqare error
    return np.mean(np.square(orig - noisy))

def get_PSNR(noisy, orig, maxval=100):
    # measured in decibles(dB)
    mse = get_mse(noisy,orig)
    return -10 * np.log10(mse/(maxval**2))