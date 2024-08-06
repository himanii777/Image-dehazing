import cv2
import math
from cv2 import imshow
from cv2 import PSNR
import numpy as np
import csv
import timeit


def calculate_psnr(img1, img2):
    img1 = img1.astype('float64')/255
    img2 = img2.astype('float64')/255
    img1=cv2.resize(img1,(400,600))
    img2=cv2.resize(img2,(400,600))
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    img1=cv2.resize(img1,(400,600))
    img2=cv2.resize(img2,(400,600))
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()    
def calculate_ssim(img1, img2):
    img1=cv2.resize(img1,(400,600))
    img2=cv2.resize(img2,(400,600))
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
        
def dark_channel(im, size):
    if len(im.shape) == 2:  # If the image is single-channel
        dc = im.copy()
    else:
        b, g, r = cv2.split(im)
        dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.erode(dc, kernel)

def atmospheric_light(im, dc):
    flat = dc.reshape(dc.shape[0] * dc.shape[1])
    flat = np.argsort(flat)
    idx = flat[int(len(flat) * 0.95)]
    return im[idx // im.shape[1], idx % im.shape[1]]

def transmission_estimate(im, al, size, omega, t0):
    t_b = 1 - omega * dark_channel(im[:, :, 0] / al[0], size)
    t_g = 1 - omega * dark_channel(im[:, :, 1] / al[1], size)
    t_r = 1 - omega * dark_channel(im[:, :, 2] / al[2], size)
    return cv2.min(cv2.min(t_b, t_g), t_r)

def guided_filter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    return mean_a * im + mean_b

def recover(im, t, al, t0):
    res = np.empty_like(im)
    for ind in range(3):
        res[:, :, ind] = ((im[:, :, ind] - al[ind]) / np.maximum(t, t0)) + al[ind]
    return res

##############CHANGEDONE#################
def multi_scale_dark_channel(im, scales=[3, 5, 7]):
    combined_dc = None
    for scale in scales:
        # Resize the image to the current scale
        resized_im = cv2.resize(im, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_LINEAR)
        # Dark channel at the current scale
        dc = dark_channel(resized_im, scale)
        # Resizing dark channel back to the original size
        dc_resized = cv2.resize(dc, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LINEAR)
        if combined_dc is None:
            combined_dc = dc_resized
        else:
            # Take the minimum value among the darkchannels
            combined_dc = np.minimum(combined_dc, dc_resized)
    return combined_dc

def dehaze(image, omega=0.80, win_size=15, eps=0.001, t0=0.4):
    dc = multi_scale_dark_channel(image, scales=[3, 5, 7])
    al = atmospheric_light(image, dc)
    transmission = transmission_estimate(image, al, win_size, omega, t0)
    refined_t = guided_filter(image[:, :, 0], transmission, 60, eps)
    result = recover(image.astype(np.float64), refined_t, al, t0)
    result = np.uint8(np.clip(result, 0, 255))    
    return result


if __name__ == '__main__':
    import timeit
    import os

    starttime = timeit.default_timer()
    starting_path = "./Hazy Images/"
    ending_path = "_outdoor_hazy.jpg"
    starting_path_for_saving = "./dehazed images/"
    ending_path_for_saving = "__dehazed.png"
    starting_path_for_realimg = "./Ground Truth/"
    ending_path_for_realimg = "_outdoor_GT.jpg"

    os.makedirs(starting_path_for_saving, exist_ok=True)

    f = open('values.csv', 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(['Image Index', 'PSNR', 'SSIM'])

    for i in range(1, 17):
        k = f"{i:02d}"
        name = starting_path + str(k) + ending_path
        input_image = cv2.imread(name)
        if input_image is None:
            print(f"Image {name} could not be loaded.")
            continue

        I = input_image.astype('float64') / 255
        dehazed_image = dehaze(input_image)
        
        path_to_save = starting_path_for_saving + str(k) + ending_path_for_saving
        cv2.imwrite(path_to_save, dehazed_image)

        path_for_ground_truth = starting_path_for_realimg + str(k) + ending_path_for_realimg
        final = cv2.imread(path_for_ground_truth)
        if final is None:
            print(f"Ground truth image {path_for_ground_truth} could not be loaded.")
            continue

        psnr_value = calculate_psnr(dehazed_image, final)
        ssim_value = calculate_ssim(dehazed_image, final)
        
        writer.writerow([int(k), psnr_value, ssim_value])

    f.close()
    print("The time taken is:", timeit.default_timer() - starttime, "s")
