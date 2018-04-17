import os
import numpy as np
import skimage.io
from skimage.io import *
from dirs import ROOT_DIR, OUT_FILES, make_dir
from skimage.morphology import *
from src.utils import data_exploration as de
from tqdm import tqdm
import warnings


def read_labels(labels_dir, label_id):
    """
    Read labels (masks) for predicted data
    :param labels_dir: labels directory
    :param label_id: id of label
    :return: all labels from the label directory
    """
    # Get array of all masks for current image
    label_dir = os.path.join(ROOT_DIR, r'{}/{}/*'.format(labels_dir, label_id))
    labels = imread_collection(label_dir).concatenate()

    return labels


def save_labels(labels, out_dir, model_name, image_id):
    """
    Save labels (masks) after postprocessing pipeline
    :param labels: label image
    :param out_dir: output directory
    :param model_name: name of the model, used for this labels predition
    :param image_id:
    :return:
    """
    warnings.filterwarnings("ignore")  # Warnings off
    # Make output directories
    dir_to_save = r'{}/{}'.format(out_dir, model_name)  # Path to return
    current_image_dir = r'{}/{}'.format(dir_to_save, image_id)
    make_dir(current_image_dir)

    for i, image in enumerate(labels):
        imsave(os.path.join(OUT_FILES, current_image_dir, r'{}.png'.format(i)), image)


def overlapping_fix(labels):
    """
    Simple overlapping remove function. Check if the instances on
    the two labels overlaps and subtract one from another
    :param labels: labels file for one image
    :return: subtracted labels
    """
    out_labels = np.ndarray((labels.shape[0], labels.shape[1], labels.shape[2]), dtype=np.uint32)

    for i, label in enumerate(labels):
        new_labels = labels.copy()
        new_labels = np.delete(arr=new_labels, obj=i, axis=0)
        height, width = label.shape

        other_cells = np.zeros((height, width), np.uint16)
        for new_label in new_labels:
            other_cells[new_label > 0] = 255

        overlapping = np.logical_and(label, other_cells)
        overlapping = overlapping.astype(np.uint32) * 255
        without_overlapping = np.logical_xor(label, overlapping)
        without_overlapping = without_overlapping.astype(np.uint32) * 255

        if np.max(overlapping) == 0:
            out_labels[i, :, :] = label
        else:
            out_labels[i, :, :] = without_overlapping

    return out_labels


def remove_small_instances(labels, max_hole_size=56, min_obj_size=30):
    out_labels = np.ndarray((labels.shape[0], labels.shape[1], labels.shape[2]), dtype=np.uint32)
    for i, label in enumerate(labels):
        remove_holes_img = remove_small_holes(label, min_size=max_hole_size)
        remove_objects_img = remove_small_objects(remove_holes_img, min_size=min_obj_size)
        out_labels[i, :, :] = remove_objects_img * 255

    return out_labels


def morfling(labels):
    out_labels = np.ndarray((labels.shape[0], labels.shape[1], labels.shape[2]), dtype=np.uint32)
    for i, label in enumerate(labels):
        dilate_img = dilation(label)
        out_labels[i, :, :] = dilate_img

    return out_labels


def sum_masks(images_dir, image_id):
    warnings.filterwarnings("ignore")
    # Make output directories
    image_dir = os.path.join(ROOT_DIR, r'{}/{}'.format(images_dir, image_id))
    model_name = images_dir.replace("\\", "/").split("/")[-1]
    out_dir = r'out_files/images/postproc/sum_masks/{}/{}'.format(model_name, image_id)
    make_dir(out_dir)
    image_ids = next(os.walk(image_dir))[2]
    image_ids.sort(key=lambda x: int(x.split(".")[0]))

    image = skimage.io.imread(r'{}/{}'.format(image_dir, image_ids[0]))
    height, width = image.shape
    labels = np.zeros((height, width), np.uint16)
    for img_id in image_ids:
        img = imread(r'{}/{}'.format(image_dir, img_id))
        labels[img > 0] = 255

    imsave(os.path.join(ROOT_DIR, out_dir, r'{}.png'.format(image_id)), labels)

if __name__ == "__main__":
    train_ids, test_ids = de.get_id()
    # _, val_ids = de.split_test_val(source_ids=train_ids)

    # Initial parameters
    directory = os.path.join(ROOT_DIR, r'out_files/images/predict_val/mrcnn-60_ep-0.2_vs-coco_iw-heads_l-12_pep')
    image_ids = next(os.walk(os.path.join(ROOT_DIR, directory)))[1]

    mode = 'val'
    if mode == 'test':
        # image_ids = test_ids
        out_dir = r'out_files/images/postproc'
    elif mode == 'val':
        # image_ids = val_ids
        out_dir = r'out_files/images/postproc_val'
    else:
        # image_ids = []
        out_dir = ''
        print("Set mode in 'test' or 'val'")

    model_name = directory.replace("\\", "/").split("/")[-1]

    for image_id in tqdm(image_ids, total=len(image_ids)):
        labels = read_labels(directory, image_id)
        morfling_labels = morfling(labels=labels)
        removed_instances_labels = remove_small_instances(labels=morfling_labels)
        overlap_fix_labels = overlapping_fix(labels=removed_instances_labels)
        save_labels(labels=overlap_fix_labels,
                    out_dir=out_dir,
                    process_type='remove_small_obj',
                    model_name=model_name,
                    image_id=image_id)

