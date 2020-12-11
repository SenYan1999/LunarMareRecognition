import numpy as np
import pandas as pd
import tifffile as tiff
import math
import os
import h5py
import multiprocessing

from tqdm import tqdm, trange
from scipy.io import loadmat

def convert_longtitude_latitude(long, la, x_len, y_len, pbar, gap=0.01):
    y_min = math.floor((long / 180 + 1) / 2 * y_len)
    x_min = math.floor((-la / 90 + 1) / 2 * x_len)

    y_max = math.floor(((long + gap) / 180 + 1) / 2 * y_len)
    x_max = math.floor(((-la + gap) / 90 + 1) / 2 * x_len)

    pbar.update(1)
    return (x_min, x_max), (y_min, y_max)

def segment_image(raw_image, total_number, out_dir):
    split_number = math.floor(math.sqrt(total_number))
    width, height = math.floor(raw_image.shape[0] / split_number), math.floor(raw_image.shape[1] / split_number)
    for i in trange(split_number):
        for j in range(split_number):
            # get the image to be output
            x_start, x_end = i * width, (i + 1) * width
            y_start, y_end = i * height, (i + 1) * height
            to_be_output = raw_image[x_start: x_end, y_start: y_end]

            # output the images to out directory
            out_file = os.path.join(out_dir, '%d.tif' % (i * split_number + j))
            tiff.imwrite(out_file, to_be_output)

def load_txt_annotated(file):
    # read the annotated data
    annotated_data = pd.read_csv(file, sep=' ', header=None).dropna()
    columns = ['longtitude', 'latitude']
    annotated_data.columns = columns
    for column in columns:
        annotated_data[column] = annotated_data[column].astype(float)
    return annotated_data

def load_dat_annotated(file):
    # read the annotated data

    # data = loadmat(file)
    # mareMask = data['MareMask']
    # lat = data['latlat']
    # lon = data['lonlon']

    mareMask = h5py.File(file)['MareMask']

    return mareMask

def annotate_raw_iamge(raw_image, annotate_file):
    # load mat data
    mareMask = load_dat_annotated(annotate_file)

    # read the raw image using tifffile
    raw_image = tiff.imread(raw_image)

    # annotate to original image
    x_len, y_len = raw_image.shape

    mask_lon, mask_lat = np.where(np.array(mareMask) == 1)
    pbar = tqdm(total = mask_lon.shape[0])
    lat = np.linspace(-90, 90, 18001)
    lon = np.linspace(-180, 180, 36001)

    for i, j in zip(mask_lon, mask_lat):
        x_inter, y_inter = convert_longtitude_latitude(lon[i], lat[j], x_len, y_len, pbar)
        raw_image[x_inter[0]: x_inter[1], y_inter[0]: y_inter[1]] = 255


    # use multiprocessing to quickly annotate
    # with multiprocessing.Pool(processes=os.cpu_count()) as pool:
    #     result = []
    #     for i, j in zip(mask_lon, mask_lat):
    #         result.append(pool.apply_async(convert_longtitude_latitude, (lon[i], lat[j], x_len, y_len, pbar)))
    #     # for i in range(18001):
    #     #     for j in range(36001):
    #     #         # if not mare
    #     #         if mareMask[j][i] == 0:
    #     #             pbar.update(1)
    #     #             continue
    #     #
    #     #         # get the block to be annotated
    #     #         result.append(pool.apply_async(convert_longtitude_latitude, (lon[j], lat[i], x_len, y_len)))
    #
    #     # consider the boundary cases
    #     for r in result:
    #         x_inter, y_inter = r.get()
    #         # raw_image[x_inter[0]: x_inter[1], y_inter[0]: y_inter[1]] = 255

    # split the data
    # segment_image(raw_image, 225, '../data/annotated_split_images')

    # use opencv to fill poly
    # print('Using OpenCV to fill poly')
    # cv2.fillPoly(np.array(raw_image), [np.array(pts),], (255, 255))
    # print('Done.')
    
    # print('Saving annonated images...')
    tiff.imwrite('tmp_final.tif', raw_image)
    # print('Done.')

def test():
    raw_image = tiff.imread('../data/raw_data/lunar.tif')

if __name__ == '__main__':
    raw_image = '../data/raw_data/lunar.tif'
    annotate_file = '../data/raw_data/mask.mat'
    annotate_raw_iamge(raw_image, annotate_file)

