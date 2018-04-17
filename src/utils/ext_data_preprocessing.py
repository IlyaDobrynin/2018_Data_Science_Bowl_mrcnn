import os
import numpy as np # linear algebra
from skimage.io import *
from collections import Counter
from dirs import ROOT_DIR, make_dir
import matplotlib.pyplot as plt
from src.utils import data_exploration as de
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")  # Warnings off


def split_image_labels(image=None, labels=None, size=(500, 500)):
    """
    Split external data from 1000x1000px to 500z500px
    :param image: image file
    :param labels: mask file, corresponded to this image
    :param size: desired split image size
    :return: splited imag and labels
    """
    i_h, i_w, i_ch = image.shape
    l_a, l_h, l_w = labels.shape
    if i_h == l_h and i_w == l_w:
        h, w = i_h, i_w
    else:
        h, w = 0, 0
        print("Image and labels sizes mismatch")

    total = int(h/size[0]) * int(w/size[1])
    output_image = np.ndarray((total, size[0], size[1], i_ch), dtype=np.uint16)
    output_labels = np.ndarray((total, l_a, size[0], size[1]), dtype=np.uint16)

    count = 0
    for i in range(0, h, size[0]):
        for j in range(0, w, size[1]):
            output_image[count, :, :, :] = image[i:(i + size[0]), j:(j + size[1]), :]
            output_labels[count, :, :, :] = labels[:, i:(i + size[0]), j:(j + size[1])]
            count += 1

    return output_image, output_labels


def save_data(image, labels, image_id=None, out_path=None):
    """
    Save splited images and labels
    :param image: image to save
    :param labels: label to save
    :param image_id: name for image to save
    :param out_path: path for saved images
    :return:
    """
    if image.shape[0] == labels.shape[0]:
        fold = image.shape[0]
    else:
        fold = None
        print("Number of splits mismatch")

    for f in range(0, fold):
        split_image_id = "%s-%s" % (image_id, f+1)
        out_images_dir = make_dir(os.path.join(out_path, r'{}/images'.format(split_image_id)))
        out_masks_dir = make_dir(os.path.join(out_path, r'{}/masks'.format(split_image_id)))
        imsave(os.path.join(out_images_dir, r'{}.png'.format(split_image_id)), image[f, :, :])
        for l in range(0, labels.shape[1]):
            label = labels[f, l, :, :]
            if np.max(label) > 0:
                imsave(os.path.join(out_masks_dir, r'mask_{}.png'.format(l+1)), label)


def check_split(path):
    """
    Util function for check if the split went well
    :param path: path to splited images
    :return:
    """
    external_ids = next(os.walk(path))[1]
    random_images = np.random.choice(external_ids, 2)
    fig = plt.figure(1, figsize=(15, 15))
    for i, image_id in enumerate(random_images):
        image_file = os.path.join(ROOT_DIR, path, r'{}/images/{}.png'.format(image_id, image_id))
        image = imread(image_file)
        mask_file = os.path.join(ROOT_DIR, path, r'{}/masks/*.png'.format(image_id))
        masks = imread_collection(mask_file).concatenate()
        height, width, _ = image.shape
        num_masks = masks.shape[0]
        labels = np.zeros((height, width), np.uint16)
        for index in range(0, num_masks):
            labels[masks[index] > 0] = index + 1

        ax = fig.add_subplot(2, len(random_images), i * 2 + 1)
        ax.imshow(labels)
        ay = fig.add_subplot(2, len(random_images), i * 2 + 2)
        ay.imshow(image)

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()


if __name__ == '__main__':
    external_ids = de.get_external_ids()
    out_path = r'data/external/extra_data_splited'

    # for ext_id in tqdm(external_ids, total=len(external_ids)):
    #     image, masks, _ = de.read_image_labels(image_id=ext_id, img_type='ext')
    #     output_image, output_labels = split_image_labels(image, masks)
    #     save_data(image=output_image, labels=output_labels, image_id=ext_id, out_path=out_path)
    path = os.path.join(ROOT_DIR, r'data/internal_external/train')
    check_split(path=path)

