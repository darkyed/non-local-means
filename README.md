# non-local-means 

Implementation of Non-Local Means algorithm, and its comparison with Gaussian Filter, on:
- Gaussian Noise
- Salt and Pepper Noise

This implementation heavily relies on numba's `jit` to speed up the computation

## Program Structure
To execute the code on a particular image, locate the variable `image_test` in `main.py` and change it to your image's name(default: `"Image3.jpg"`). Make sure the image is present in the `Data` directory.

- Non-Local Means program: `utilfuncs/nlmeans.py`

The directory `utilfuncs/` also contains utility functions for preprocessing the images, adding noise to images, computing metrics etc.

## Results
After `main.py` finishes executing, three directories will be spawned in the current directory
- **gray**: grayscale version of original images
- **noisy**: grayscale images with noise added to them
  - **gauss**: Gaussian Noise
  - **saltnpepp**: Salt and Pepper Nouse
- **results**: Images after denoising
  - **images of format**: `deno_\<algorithm>_\<noise_type>.jpg`
  - **result.txt**: MSE and PSNR values

And a file named `time.txt`:
- Contains the details about the time taken by each of the algorithms on different types of noises
  - Gaussian Filter on Gaussian Noise
  - Gaussian Filter on Salt and Pepper Noise
  - Non-Local Means on Gaussian Noise
  - Non-Local Means on Salt and Pepper Noise

## References
- A. Buades, B. Coll and J. -. Morel, "A non-local algorithm for image denoising," 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05), San Diego, CA, USA, 2005, pp. 60-65 vol. 2, doi: 10.1109/CVPR.2005.38.