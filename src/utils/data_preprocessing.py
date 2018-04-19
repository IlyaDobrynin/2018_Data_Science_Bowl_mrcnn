import os
import sys
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize

if os.name is 'nt':
    dir_splitter = '\\'
else:
    dir_splitter = '/'

# Set some parameters
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
TRAIN_PATH = '../data/images/train'
TEST_PATH = '../data/images/test'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

# Get train and test IDs
def get_id():
    train_ids = next(os.walk(TRAIN_PATH))[1]
    test_ids = next(os.walk(TEST_PATH))[1]

    return train_ids, test_ids

def get_and_resize(train_ids, test_ids, type):
    # Get and resize train images and masks
    out_npy_path = '../out_files/npy/256_256'
    if type == 'train':
        X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
        Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        print('Getting and resizing train images and masks ... ')
        sys.stdout.flush()
        for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)): #len(train_ids)):
            path = os.path.join(TRAIN_PATH, id_)
            img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            X_train[n] = img
            mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_ = imread(path + '/masks/' + mask_file)
                mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                              preserve_range=True), axis=-1)
                mask = np.maximum(mask, mask_)
            Y_train[n] = mask

        np.save(os.path.join(out_npy_path, 'X_train'), X_train)
        np.save(os.path.join(out_npy_path, 'Y_train'), Y_train)

    elif type == 'test':
        X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
        print('Getting and resizing test images ... ')
        sys.stdout.flush()
        for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
            path = os.path.join(TEST_PATH, id_)
            img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            X_test[n] = img

        np.save(os.path.join(out_npy_path, 'X_test'), X_test)
        np.save(os.path.join(out_npy_path, 'X_test'), X_test)


def check_npy(npy_path):
    npy = np.load(npy_path)
    print("-"*30 + ' CHECKING NPY ' + '-'*30)
    print(npy.shape)


def show_train_img(x_path, y_path):
    images = np.load(x_path)
    masks = np.load(y_path)
    ix = random.randint(0, len(train_ids))
    # imshow(np.squeeze(images[ix]))
    fig = plt.figure(1, figsize=(10, 10))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(images[ix])
    ay = fig.add_subplot(1, 2, 2)
    ay.imshow(np.squeeze(masks[ix]))
    plt.show()

if __name__ == '__main__':
    train_ids, test_ids = get_id()
    get_and_resize(train_ids=train_ids, test_ids=test_ids, type='train')
    get_and_resize(train_ids=train_ids, test_ids=test_ids, type='test')
    # show_train_img('../out_files/npy/default/X_train.npy', '../out_files/npy/default/Y_train.npy')
    # check_npy('../out_files/npy/default/X_train.npy')


