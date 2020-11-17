import numpy as np
from numba import jit


def get_relative_centers(small_winsize, large_winsize):
    # enlist all the centers of small window inside large window
    # except the current pixel(relative, i.e., assumed (0,0) for now)
    
    center_small, center_large = small_winsize//2, large_winsize//2
    range_var = center_large - center_small
    rrange = range(-range_var, range_var + 1)
    
    # List comprehensions are compact and faster
    rel_center_coords = [[i,j] for i in rrange for j in rrange if (i,j)!=(0,0)] # excluding the current position (0,0)
    return np.array(rel_center_coords)


def get_all_small_windows(mat, curr_pos, small_r, small_c, rel_centers):
    # rel_centers contain the centers of neighbourhood windows
    # here we add current center to get all of the neighbourhood windows
    
    ci,cj = curr_pos
    all_windows = []
    for (i,j) in rel_centers:
        all_windows += [mat[small_r+ci+i, small_c+cj+j]]
    
    return np.array(all_windows)


def get_rel_pos(mat, curr_pos, rel_centers):
    # here we get the relative positions of the window cells
    # by taking differecne between current pixel and pixel at 
    # relative distance (i,j) from current pixel
    curr_arr = np.array(curr_pos)
    ci,cj = curr_pos
    shifted_arr = curr_arr + rel_centers
    shifte_i, shifted_j = shifted_arr[:,0], shifted_arr[:,0]
    rel_pos = mat[ci, cj] - mat[shifte_i,shifted_j]

    return rel_pos


@jit(nopython=True)
def get_distance(current_window, nbr_window, diff_center, hh, nw):
    # here we take the weights and estimated pixel(distance)
    diff = np.sum((current_window - nbr_window)**2) - diff_center
    denom = hh*nw
    w = np.exp(-1*(diff/denom))
    r,c = nbr_window.shape
    return w*(nbr_window[r//2, c//2]), w


@jit(nopython=True)
def get_net_vals(rel_poss, curr_wind, nbr_winds, hh, nw):
    # here we sum up the weighted pixel(sum(weight*estimated_pixel) --> net_distance) and weights(sum(weight) --> net_weight)
    net_distance = net_weight = 0
    for ip,pos in enumerate(rel_poss):
        dist,wt = get_distance(curr_wind, nbr_winds[ip], pos, hh, nw)
        net_distance = net_distance+dist
        net_weight = net_weight+wt
    return net_distance, net_weight


def non_local_means(noisy_img, large_winsize=21, small_winsize=7, h=10):
    '''
    input:
    ------------------------------------------------------------------------
    noisy_img: noisy image (in grayscale) dimensions: (m x n)
    large_winsize: size of the large window (provide only one dimension)
    small_winsize: size of the smaller window (provide only one diemension)
    h: filtering parameter (generally fixed to 10 x sigma)
    
    output:
    ------------------------------------------------------------------------
    new_img: Denoised image with the same dimensions i.e. (m x n)
    '''
    nw = small_winsize**2
    hh = h*h
    rh,cw = noisy_img.shape

    # our output image
    new_image = np.zeros((rh,cw))

    # padding symmetrically to avoid out of bounds error at edges and corners
    padding = large_winsize//2
    padded_img = np.pad(noisy_img,padding,mode='edge')

    # centers will be used to get absolute position(by adding relative positions) using difference
    center_small, center_large = small_winsize//2, large_winsize//2

    # row matrix, col matrix of coordiantes of small window (remove center to get centralized relative corrdinates)
    rows_small, cols_small = np.indices((small_winsize, small_winsize)) - center_small
    rel_centers = get_relative_centers(small_winsize,large_winsize) # big_diff

    # Iterate for every pixel in the image(before padding)
    for i in range(padding, padding+rh):
        for j in range(padding, padding+cw):

            current_window = padded_img[i+rows_small, j+cols_small]

            # all small neighbour windows inside the big window(centered at current pixel)
            neighbour_windows = get_all_small_windows(padded_img, (i,j), rows_small, cols_small, rel_centers) # windows

            # get the original coordinates of the small windows by adding current coordinates to relative coordinates
            rel_positions = get_rel_pos(padded_img, (i,j), rel_centers)

            # get the weighted pixel(net_distance, i.e sum(w*color)) and total weight(net_weight, i.e. sum(w))
            net_distance, net_weight = get_net_vals(rel_positions,current_window,neighbour_windows, hh, nw)

            # assign the value to current pixel(recall we have padded indices here --> so subtract them)
            new_image[i-padding, j-padding] = np.clip(net_distance/net_weight,0,255)

    return new_image