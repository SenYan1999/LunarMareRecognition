import tifffile as tiff
import random
import numpy as np
import os
import json

LABEL2COLOR={'Mare': 255, 'Other': 0}

# now we define the IMAGE and IMAGE_ANNOTATED to avoid pass too large arguments in multiprocessing
print('Loading original lunar image and annotated lunar image')
IMAGE = np.load('data/raw_data/lunar.npy')
IMAGE_ANNOTATED = np.load('data/raw_data/lunar_annotated.npy')
print('done...')

def random_sample_window(x_len, y_len, window_size):
    begin_x = random.randint(0, x_len - window_size - 1)
    begin_y = random.randint(0, y_len - window_size - 1)

    return (begin_x, begin_x + window_size), (begin_y, begin_y + window_size)

def is_select_balanced(x_range, y_range, thresholds):
    area = IMAGE_ANNOTATED[x_range[0]: x_range[1], y_range[0]: y_range[1]]

    if thresholds[0] < np.sum(area == 255) / (area.shape[0] * area.shape[1]) < thresholds[1]:
        return True
    else:
        return False

def is_select_most_mare(x_range, y_range, thresholds):
    area = IMAGE_ANNOTATED[x_range[0]: x_range[1], y_range[0]: y_range[1]]

    ratio = np.sum(area == 255) / (area.shape[0] * area.shape[1])

    if ratio > thresholds[1]:
        return True
    else:
        return False

def is_select_little_mare(x_range, y_range, thresholds):
    area = IMAGE_ANNOTATED[x_range[0]: x_range[1], y_range[0]: y_range[1]]

    ratio = np.sum(area == 255) / (area.shape[0] * area.shape[1])

    if thresholds[0] > ratio:
        return True
    else:
        return False

def find_area_job(x_len, y_len, window_size, balanced_thresholds, unbalanced_thresholds):
    while True:
        x_range, y_range = random_sample_window(x_len, y_len, window_size=window_size)
        if is_select_balanced(x_range, y_range, balanced_thresholds):
            return 0, (x_range, y_range)
        elif is_select_most_mare(x_range, y_range, unbalanced_thresholds):
            return 1, (x_range, y_range)
        elif is_select_little_mare(x_range, y_range, unbalanced_thresholds):
            return 2, (x_range, y_range)
        else:
            continue

def retrieve_areas(window_size, balanced_thresholds, unbalanced_thresholds, num_balanced, num_unbalanced, out_dir):
    assert num_balanced > 0
    assert num_unbalanced > 0

    # append areas selected
    balanced_segments = []
    most_mare_segments = []
    little_mare_segments = []
    x_len, y_len = IMAGE_ANNOTATED.shape[0], IMAGE_ANNOTATED.shape[1]
    idx = 0
    while len(balanced_segments) != num_balanced or len(most_mare_segments) != num_unbalanced // 2 or len(little_mare_segments) != num_unbalanced // 2:
        signal, area = find_area_job(x_len, y_len, window_size, balanced_thresholds, unbalanced_thresholds)
        if len(balanced_segments) != num_balanced and signal == 0:
            balanced_segments.append(area)
            write_images(area, out_dir, idx)
            idx += 1
        elif len(most_mare_segments) != num_unbalanced  // 2 and signal == 1:
            most_mare_segments.append(area)
            write_images(area, out_dir, idx)
            idx += 1
        elif len(most_mare_segments) != num_unbalanced  // 2 and signal == 2:
            most_mare_segments.append(area)
            write_images(area, out_dir, idx)
            idx += 1
        else:
            continue

    print('Now dumping segments to json files')
    with open(os.path.join(out_dir, 'balanced_segments.json'), 'w') as f:
        json.dump(balanced_segments, f)
    with open(os.path.join(out_dir, 'most_mare_segments.json'), 'w') as f:
        json.dump(most_mare_segments, f)
    with open(os.path.join(out_dir, 'little_mare_segments.json'), 'w') as f:
        json.dump(little_mare_segments, f)
    print('done...')

def write_images(segment, out_dir, id):
    input_dir = os.path.join(out_dir, 'input')
    label_dir = os.path.join(out_dir, 'label')

    if not os.path.exists(input_dir):
        os.mkdir(input_dir)
    if not os.path.exists(label_dir):
        os.mkdir(label_dir)

    # write input image
    area = IMAGE[segment[0][0]: segment[0][1], segment[1][0]: segment[1][1]]
    tiff.imwrite(os.path.join(input_dir, str(id) + '.tiff'), data=area)

    # write label image
    area_annotated = IMAGE_ANNOTATED[segment[0][0]: segment[0][1], segment[1][0]: segment[1][1]]
    pixel_annotated = np.where(area_annotated == LABEL2COLOR['Mare'])
    area = np.zeros_like(area_annotated)
    area[pixel_annotated] = LABEL2COLOR['Mare']
    tiff.imwrite(os.path.join(label_dir, str(id) + '.tiff'), data=area)

def main():
    window_size = 512
    balanced_thresholds = (0.2, 0.6)
    unbalanced_thresholds = (0.1, 0.9)
    balanced_num = 14000
    unbalanced_num = 6000
    out_dir = 'data/all_segments'
    print('Now we begin retrieve selected areas')
    retrieve_areas(window_size, balanced_thresholds, unbalanced_thresholds, balanced_num, unbalanced_num, out_dir)
    print('done...')

if __name__ == '__main__':
    main()
