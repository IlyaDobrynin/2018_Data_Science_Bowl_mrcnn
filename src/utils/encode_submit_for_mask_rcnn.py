import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from skimage.io import imread
from skimage.morphology import label
from dirs import ROOT_DIR, make_dir
from src.utils.data_exploration import get_id


def preds_test(test_ids, predict_files_path):
    preds_test_images = []
    for i, ids in tqdm(enumerate(test_ids), total=len(test_ids)):
        pred_image = np.load(os.path.join(ROOT_DIR, predict_files_path + r'/{}.npy'.format(ids)))
        preds_test_images.append(np.squeeze(pred_image))

    return preds_test_images


def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b

    return run_lengths


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


def create_submit(files_path, model_name, submit_path):
    _, test_ids = get_id()

    new_test_ids = []
    rles = []

    for image_id in tqdm(test_ids, total=len(test_ids)):
        preds_test_images = []
        ids = next(os.walk(os.path.join(files_path, '{}'.format(image_id))))
        for i, image_index in enumerate(ids[2]):
            pred_image = imread(os.path.join(ids[0], image_index))
            preds_test_images.append(pred_image)

        for idx, img_id in enumerate(ids[2]):
            rle = list(prob_to_rles(preds_test_images[idx]))
            rles.extend(rle)
            new_test_ids.extend([image_id] * len(rle))

    # Create submission DataFrame
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv(os.path.join(submit_path, r'{}_sub.csv'.format(model_name)), index=False)


if __name__ == '__main__':
    print('-' * 30 + ' Creating submit file... ' + '-' * 30)
    images_to_encode = os.path.join(ROOT_DIR,
                                    r'out_files/images/postproc/remove_small_obj/mrcnn-60_ep-0.2_vs-coco_iw-heads_l-24_pep')
    model_name = images_to_encode.replace("\\", "/").split("/")[-1]
    submit_path = make_dir('sub/{}'.format(model_name))
    create_submit(files_path=images_to_encode, model_name=model_name, submit_path=submit_path)




