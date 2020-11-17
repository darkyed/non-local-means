# non-local-means 

Implementation of Non-Local Means algorithm, and its comparison with Gaussian Filter, on:
- Gaussian Noise
- Salt and Pepper Noise

## Code Structure
To execute the code on a particular image, locate the variable `image_test` in `main.py` and change it to your image's name(default: `"Image3.jpg"`). Make sure the image is present in the `Data` directory.

Non-Local Means program: `utilfuncs/nlmeans.py`

## Results
After `main.py` finishes executing, three directories will be spawned in the current directory
- **gray**: grayscale version of original images
- **noisy**: grayscale images with noise added to them
  - **gauss**: Gaussian Noise
  - **saltnpepp**: Salt and Pepper Nouse
- **results**: Images after denoising
  - **images of format**: deno_\<algorithm>_\<noise_type>.jpg
  - **result.txt**: MSE and PSNR values

And a file, `time.txt`:
- Contains the details about the time taken by each of the alogorithms on different types of noises
