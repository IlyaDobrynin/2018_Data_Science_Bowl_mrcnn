import os
import numpy as np
import skimage.io
import skimage.segmentation
from dirs import ROOT_DIR, TRAIN_DATASET_DIR
from tqdm import tqdm


def calculate_image_iou(image_id, true_images_dir, pred_images_dir):
    # Load a single image and its associated masks
    file = os.path.join(ROOT_DIR, true_images_dir, r"{}/images/{}.png".format(image_id, image_id))
    mfile = os.path.join(ROOT_DIR, true_images_dir, r"{}/masks/*.png".format(image_id))
    image = skimage.io.imread(file)
    masks = skimage.io.imread_collection(mfile).concatenate()
    height, width, _ = image.shape
    num_masks = masks.shape[0]

    # Make a ground truth array and summary label image
    y_true = np.zeros((num_masks, height, width), np.uint16)
    y_true[:,:,:] = masks[:,:,:] // 255  # Change ground truth mask to zeros and ones

    pred_mfile = os.path.join(ROOT_DIR, pred_images_dir, r'{}/*'.format(image_id))
    pred_masks = skimage.io.imread_collection(pred_mfile).concatenate()
    num_pred_masks = pred_masks.shape[0]
    y_pred = np.zeros((num_pred_masks, height, width), np.uint16)
    y_pred[:, :, :] = pred_masks[:, :, :] // 255

    # Compute number of objects
    num_true = len(y_true)
    num_pred = len(y_pred)

    iou = []
    for predicted in range(num_pred):
        bol = 0  # best overlap
        bun = 1e-9  # corresponding best union
        for true in range(num_true):
            olap = y_pred[predicted] * y_true[true]  # Intersection points
            osz = np.sum(olap)  # Add the intersection points to see size of overlap
            if osz > bol:  # Choose the match with the biggest overlap
                bol = osz
                bun = np.sum(np.maximum(y_pred[predicted], y_true[true]))  # Union formed with sum of maxima
        iou.append(bol / bun)

    # Loop over IoU thresholds
    p = 0
    for t in np.arange(0.5, 1.0, 0.05):
        matches = iou > t
        tp = np.count_nonzero(matches)  # True positives
        fp = num_pred - tp  # False positives
        fn = num_true - tp  # False negatives
        p += tp / (tp + fp + fn)

    return p / 10


if __name__ == "__main__":

    # For standalone metric calculation change postproc_images_dir to your path to data
    postproc_images_dir = r'out_files/images/predict_val/cell20180412T2356-080_ep-60_pep'


    postproc_images_ids = next(os.walk(os.path.join(ROOT_DIR, postproc_images_dir)))[1]
    true_images_dir = TRAIN_DATASET_DIR
    p = 0
    for image_id in tqdm(postproc_images_ids, total=len(postproc_images_ids)):
        p += calculate_image_iou(image_id=image_id,
                                 true_images_dir=true_images_dir,
                                 pred_images_dir=postproc_images_dir)
    mean_p = p / len(postproc_images_ids)

    print('\n\nTotal IoU for validation set: {:1.3f}'.format(mean_p))