import os, cv2, time, warnings
from scipy.ndimage import gaussian_filter
warnings.filterwarnings('ignore')

# Import utility functions
from utilfuncs.fetchresults import *
from utilfuncs.preproc.preprocess import *
from utilfuncs.nlmeans import non_local_means
from utilfuncs.metric import get_mse, get_PSNR


image_test = "Image3.jpg"

curr_path = os.getcwd()
data_path = os.path.join(curr_path, "Data")
gray_path = os.path.join(curr_path, "gray")
noisy_path = os.path.join(curr_path, "noisy")

# Creating directories to save preprocessed images
make_gray_dir(gray_path)
make_noise_dirs(noisy_path)
gauss_path, snp_path = os.path.join(noisy_path, "gauss"), os.path.join(noisy_path, "saltnpepp")

# saving the images
save_gray(gray_path, data_path)
add_noise_save(noisy_path, gray_path, gauss_path, snp_path)
base_image = open_image_arr(os.path.join(gray_path, image_test))
noisy_image_gauss = open_image_arr(os.path.join(gauss_path, image_test))
noisy_image_snp = open_image_arr(os.path.join(snp_path, image_test))

# <filter>_deno_<noise_type>
# Using  Gaussian Filter
print("Denoising gaussian noise using gaussian filter")
s1 = time.perf_counter()
gauss_deno_gauss = gaussian_filter(noisy_image_gauss, sigma=1)
f1 = time.perf_counter()
print("Done")
# ------------------------------------------------------------------------
print("Denoising salt and pepper noise using gaussian filter")
s2 = time.perf_counter()
gauss_deno_snp = gaussian_filter(noisy_image_snp, sigma=1)
f2 = time.perf_counter()
print("Done")

# ------------------------------------------------------------------------
# Using Non-Local Means
print("Denoising gaussian noise using Non-Local Means")
s3 = time.perf_counter()
nlm_deno_gauss = non_local_means(noisy_image_gauss)
f3 = time.perf_counter()
print("Done")
# ------------------------------------------------------------------------
print("Denoising salt and pepper noise using Non-Local Means")
s4 = time.perf_counter()
nlm_deno_snp = non_local_means(noisy_image_snp)
f4 = time.perf_counter()
print("Done")

results = [(get_mse(deno, base_image), get_PSNR(deno, base_image, 255))\
            for deno in [gauss_deno_gauss, gauss_deno_snp, nlm_deno_gauss, nlm_deno_snp]]

get_timings([s1,s2,s3,s4], [f1,f2,f3,f4], image_test, os.path.join(curr_path, "time.txt"))

# Metrics
if not os.path.isdir("./results"):
    os.mkdir("./results")

result_path = os.path.join(curr_path, "results")
save_results(image_test, base_image, results, noisy_image_gauss, \
            noisy_image_snp, os.path.join(result_path, "results.txt"))

cv2.imwrite("./results/deno_gs_gs.jpg", gauss_deno_gauss)
cv2.imwrite("./results/deno_gs_snp.jpg", gauss_deno_snp)
cv2.imwrite("./results/deno_nlm_gs.jpg", nlm_deno_gauss)
cv2.imwrite("./results/deno_nlm_snp.jpg", nlm_deno_snp)

print("Save the denoised images to ./results.txt")