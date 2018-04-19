import os
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import random
from dirs import ROOT_DIR, TRAIN_DATASET_DIR, TEST_DATASET_DIR, EXTERNAL_DATA


def get_id():
    """
    Get train and test ids
    :return: train_ids and test_ids
    """
    train_ids = next(os.walk(TRAIN_DATASET_DIR))[1]
    test_ids = next(os.walk(TEST_DATASET_DIR))[1]

    return train_ids, test_ids


def split_test_val(source_ids, val_split_factor=0.2):
    """
    Split source ids array into train and validation subarrays
    :param source_ids: soure full size array of ids
    :param val_split_factor: proportion for splitting, default = 0.2 (0.8 for training ids and 0.2 for validation)
    :return:  train_ads and val_ids arrays
    """
    train_ids = source_ids[:int(len(source_ids) * (1 - val_split_factor))]
    val_ids = source_ids[int(len(source_ids) * (1 - val_split_factor)):]

    return train_ids, val_ids


def get_external_ids():
    external_ids = next(os.walk(EXTERNAL_DATA))[1]
    return external_ids


def read_image_labels(data_path, image_id, img_type):
    if img_type == 'train' or img_type == 'val':
        image_file = os.path.join(data_path, r'{}/images/{}.png'.format(image_id,image_id))
        mask_file = os.path.join(data_path, r'{}/masks/*.png'.format(image_id))
        image = skimage.io.imread(image_file)
        masks = skimage.io.imread_collection(mask_file).concatenate()
        height, width, _ = image.shape
        num_masks = masks.shape[0]
        labels = np.zeros((height, width), np.uint16)
        for index in range(0, num_masks):
            labels[masks[index] > 0] = index + 1
        return image, labels
    elif img_type == 'ext':
        image_file = os.path.join(data_path, r'{}/images/{}.tif'.format(image_id,image_id))
        mask_file = os.path.join(data_path, r'{}/masks/*.png'.format(image_id))
        image = skimage.io.imread(image_file)
        masks = skimage.io.imread_collection(mask_file).concatenate()
        height, width, _ = image.shape
        num_masks = masks.shape[0]
        labels = np.zeros((height, width), np.uint16)
        for index in range(0, num_masks):
            labels[masks[index] > 0] = index + 1
        return image, masks, labels
    elif img_type == 'test':
        image_file = os.path.join(data_path, r'{}/images/{}.png'.format(image_id, image_id))
        image = skimage.io.imread(image_file)
        return image


def read_labels(preds_path, image_id):
    mask_file = os.path.join(ROOT_DIR, r'{}\{}\*.png'.format(preds_path, image_id))
    masks = skimage.io.imread_collection(mask_file).concatenate()
    _, height, width = masks.shape
    num_masks = masks.shape[0]
    labels = np.zeros((height, width), np.uint16)
    for index in range(0, num_masks):
        labels[masks[index] > 0] = index + int(255/num_masks)
    return labels


def images_shapes_parameters(ids, id_type):
    widths = np.ndarray(shape=(len(ids),), dtype=np.int32)
    heights = np.ndarray(shape=(len(ids),), dtype=np.int32)
    for i, image_id in enumerate(ids):
        if id_type == 'train':
            image, _ = read_image_labels(data_path=TRAIN_DATASET_DIR, image_id=image_id, img_type=id_type)
        else:
            image = read_image_labels(data_path=TEST_DATASET_DIR, image_id=image_id, img_type=id_type)
        widths[i] = image.shape[1]
        heights[i] = image.shape[0]

    unique_widths = np.nonzero(np.bincount(widths))[0]
    unique_heights = np.nonzero(np.bincount(heights))[0]

    print("Unique widths", unique_widths)
    print("Mean width", np.mean(unique_widths))
    print("Unique heights", unique_heights)
    print("Mean height", np.mean(unique_heights))


def read_image(images_path, image_id):
    return skimage.io.imread(os.path.join(ROOT_DIR, images_path, image_id, r'{}.png'.format(image_id)))


def show_images(paths, img_type, img_to_show=1, images_ids_list=None, random_mode=True):
    rand = random.randint(0, 85)
    for n, preds_path in enumerate(paths):
        np.random.seed(rand)
        image_ids = next(os.walk(os.path.join(ROOT_DIR, preds_path)))[1]
        if random_mode and not images_ids_list:
            images = np.random.choice(image_ids, img_to_show)
        else:
            if images_ids_list:
                images = images_ids_list
            else:
                print("No images to show.\n You can provide images_ids_list parameter, or set random mode=True")
                break

        print("Images list:\n", images)

        for i, image_id in enumerate(images):
            if img_type == 'test':
                image = read_image_labels(data_path=TEST_DATASET_DIR, image_id=image_id, img_type=img_type)
                labels = read_labels(preds_path, image_id)
                fig = plt.figure(n + 1, figsize=(15, 15))
                ax = fig.add_subplot(2, len(images), (i + 1))
                ax.imshow(labels)
                ay = fig.add_subplot(2, len(images), (i + 1) + len(images))
                ay.imshow(image)
                fig.suptitle(preds_path + '\n{}'.format(image_id))
            else:
                image, labels = read_image_labels(data_path=TRAIN_DATASET_DIR, image_id=image_id, img_type=img_type)
                label = read_labels(preds_path, image_id)
                fig = plt.figure(n + 1, figsize=(15, 15))
                ax = fig.add_subplot(3, len(images), (i + 1))
                ax.imshow(image)
                ax = fig.add_subplot(3, len(images), (i + 1) + len(images))
                ax.imshow(labels)
                ay = fig.add_subplot(3, len(images), (i + 1) + 2 * len(images))
                ay.imshow(label)
                fig.suptitle(preds_path + '\n{}'.format(image_id))

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()


if __name__ == '__main__':

    paths = [
             r'out_files/images/predict/cell20180411T1546-060_ep-45_pep',
             r'out_files/images/predict/cell20180412T2356-080_ep-60_pep',
             # r'out_files/images/postproc/remove_small_obj/cell20180411T1546-060_ep-45_pep',
             # r'out_files/images/postproc/remove_small_obj/cell20180412T2356-080_ep-60_pep',
             ]

    images_ids_list = ['a479b10e9b12970b86bd7f9aef6b1224bd0fc49d39b598bd0c2a67aebff74cec',
                       '4c9f76aa783818aa1c3506820b995391afb06a36fd034de44311364e6e161bc0']
    img_type = 'test'

    show_images(paths=paths,
                img_type=img_type,
                images_ids_list=images_ids_list,
                img_to_show=2)


