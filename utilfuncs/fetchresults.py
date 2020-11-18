from itertools import product
from utilfuncs.metric import *

filter, noise_type = ["gauss", "nlm"], ["gauss", "snp"]
pairs = product(filter, noise_type)

def get_timings(start:list, finish:list, image_test:str, save_path):
    pairs = product(filter, noise_type)
    diff = [j-i for (i,j) in zip(start, finish)]

    with open(save_path, 'w') as f:
        f.write("Tested on %s\n" % image_test + "-"*60 + "\n")
        print('-'*60 + "\n")
        # <filter, noise>
        for i,e in enumerate(pairs):
            print("<%s, %s>: Time in seconds: %.3f" % (*e, diff[i]))
            f.write("<%s, %s>: Time in seconds: %.3f\n" % (*e, diff[i]))
        print('-'*60)


# Metrics
def save_results(image_test:str, base_image:np.ndarray, results:list,\
                noisy_image_gauss:np.ndarray, noisy_image_snp:np.ndarray, save_path):
    pairs = product(filter, noise_type)
    with open(save_path, 'w') as f:
        f.write("Tested on %s\n\n" % image_test + "-"*60 + "\n")
        f.write("Noisy Images:\n")
        f.write("Gaussian Noise: MSE: %f, PSNR: %f\n" % (get_mse(noisy_image_gauss, base_image),\
                get_PSNR(noisy_image_gauss, base_image, base_image.max())))
        f.write("Salt and Pepper Noise: MSE: %f, PSNR: %f\n" % (get_mse(noisy_image_snp, base_image),\
                get_PSNR(noisy_image_snp, base_image, base_image.max())))
        f.write("-"*70 + "\n\n")
        f.write("Denoised Images:\n")
        for i,e in enumerate(pairs):
            f.write("<%s, %s>: MSE: %f, PSNR: %f\n" % (*e, *results[i]))