import os
import numpy as np
from tqdm import tqdm
import skimage.io
import skimage.color
from imgaug import augmenters as iaa

import src.nn_var.mask_rcnn.mrcnn.model as modellib
import src.nn_var.mask_rcnn.CellsDataset as CD
from src.nn_var.mask_rcnn.mrcnn import utils
from src.utils import data_exploration as de
from src.utils import encode_submit_for_mask_rcnn as esfmr
from src.utils import data_postprocessing as dpost
from src.utils import metric

from dirs import ROOT_DIR, TRAIN_DATASET_DIR, TEST_DATASET_DIR, MODEL_DIR, OUT_FILES, COCO_MODEL_PATH, make_dir

np.random.seed(17)

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


def get_model_name(model, params):
    """
    Create model name in format:
                "(config.NAME)YYYYMMDDTHHMM-EEE_ep-PP_pep"
                where:
                    - config.NAME - NAME parameter from the CD.CellsConfig class
                    - YYYYMMDDTHHMM - time in the given format
                    - EEE - number of epochs overall
                    - PP - numper of epoch for predict
    :param model: model class
    :param params: parameters dictionary
    :return: name of the current model
    """
    # Create model name
    model_path = model.find_last()[1].split('\\')
    time = model_path[-2]
    fact_epohs = model_path[-1][-6:-3]
    if not model_path:
        print("Model is not trained yet")

    if params['epoch_for_predict'].isdigit():
        pred_epoch = params['epoch_for_predict']
    elif params['epoch_for_predict'] == 'last':
        pred_epoch = fact_epohs[1:]
    else:
        time = params['path_to_weights_for_predict'].split('/')[-2]
        pred_epoch = params['path_to_weights_for_predict'].split('/')[-1][-5:-3]
        fact_epohs = "path"

    name = time + '-' \
                 + "{}_ep-".format(fact_epohs) \
                 + "{}_pep".format(pred_epoch)

    return name


def train_model(model, config, params, train_ids, val_ids):
    """
    Train the model
    :param config: model configure class instance
    :param params: parameters dictionary
    :param train_ids: train images ids
    :param val_ids: validation images ids
    :return:
    """
    # Prepare training and validation datasets
    dataset_train = CD.CellsDataset()
    dataset_train.load_cells(dataset_dir=TRAIN_DATASET_DIR, image_ids=train_ids)
    dataset_train.prepare()
    dataset_val = CD.CellsDataset()
    dataset_val.load_cells(dataset_dir=TRAIN_DATASET_DIR, image_ids=val_ids)
    dataset_val.prepare()

    # Choose weights to start with
    init_with = params['init_with']
    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)
    else:
        model.load_weights(os.path.join(ROOT_DIR, params['path_to_weights_for_train']), by_name=True)

    print("Training starts with weights from %s" % init_with)

    # Add some TTA
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # Train the model.
    # You can define your own pipeline via params['learning_pipeline']
    # i.e. {'heads': 30, '4+': 20} dict means that first will be
    # trained 'heads' for 30 epochs, then will be trained '4+'
    # for 20 epochs.
    epochs = 0
    learning_rate = config.LEARNING_RATE
    for i, layer in enumerate(params['learning_pipeline'].keys()):
        epochs += list(params['learning_pipeline'].values())[i]
        model.train(dataset_train, dataset_val,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    augmentation=augmentation,
                    layers=layer
                    )
        learning_rate /= 10


def predict(model, config, params, model_name, images_ids):
    """
    Predict masks for test images
    :param model: model in inference mode
    :param config: model configure class instance
    :param params: parameters dictionary
    :param model_name: the name of the model
    :param images_ids: ids of images to predict (test_ids or val_ids)
    :return: path to the folder with predicted files
    """

    # Choose the weights for predicting
    model_path = ""
    if params['epoch_for_predict'].isdigit():
        model_path = os.path.join(model.find_last()[0], r'mask_rcnn_cell_00{}.h5'.format(params['epoch_for_predict']))
    elif params['epoch_for_predict'] == 'last':
        model_path = model.find_last()[1]
    elif params['epoch_for_predict'] == 'path':
        model_path = os.path.join(ROOT_DIR, params['path_to_weights_for_predict'])
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    # Choose the type of predicting dataset:
    #   - val (validation) - predict validation dataset for model validation score estimation;
    #   - test - predict test dataset.
    predict_type = params['predict_type']
    pred_files_dir = ""
    images_dir = ""
    if predict_type == 'test':
        # Create folder for predicted files
        relative_preds_path = r'out_files\images\predict\{}'.format(model_name)
        pred_files_dir = make_dir(relative_preds_path)
        images_dir = TEST_DATASET_DIR
    elif predict_type == 'val':
        # Create folder for predicted files
        relative_preds_path = r'out_files\images\predict_val\{}'.format(model_name)
        pred_files_dir = make_dir(relative_preds_path)
        images_dir = TRAIN_DATASET_DIR
    assert pred_files_dir != "", "Provide path to predict files"
    assert images_dir != "", "Provide path to source files"

    # Save config file
    config.save_to_file(os.path.join(pred_files_dir))

    # Predict images
    for image_id in tqdm(images_ids, total=len(images_ids)):
        # Create folder for image
        image_dir = os.path.join(pred_files_dir, image_id)
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)

        # Read test image
        test_image = skimage.io.imread(os.path.join(images_dir, r'{}\images\{}.png'.format(image_id, image_id)))
        # If grayscale. Convert to RGB for consistency.
        if test_image.ndim != 3:
            test_image = skimage.color.gray2rgb(test_image)
        elif test_image.shape[2] > 3:
            test_image = test_image[:, :, :3]

        # Predict mask for the given image
        results = model.detect([test_image], verbose=0)
        r = results[0]
        # print(type(r['masks']))
        # Save predicted masks
        # print(r['masks'].shape)
        if r['masks'].shape[0] != 0:
            for i in range(r['masks'].shape[2]):
                import warnings
                warnings.filterwarnings("ignore")
                # if r['masks']:
                r['masks'].astype(np.int32)[:, :, i] *= 255
                skimage.io.imsave('{}\{}.png'.format(image_dir, i), r['masks'][:, :, i])
                # else:
        else:
            pred_image = np.zeros((test_image.shape[0], test_image.shape[1]), dtype=np.int32)
            skimage.io.imsave('{}/0.png'.format(image_dir), pred_image)

    return pred_files_dir


def validate(model, config, params, model_name, val_ids):
    """
    Validation pipeline:
        1. Predict validation dataset
        2. Postprocessing predicted data
        3. Calculate local validation sore (LVS)
    :param model: pretrained model
    :param config: inference configurations
    :param params: parameters dictionary
    :param model_name: name of the model
    :param val_ids: array of validation ids
    :return: mean IoU score for the validation set
    """
    params['predict_type'] = 'val'
    # Predict data
    print('\nStep 1 of 3: Predicting validation data... ')

    predict_images_dir = predict(model=model,
                                 config=config,
                                 params=params,
                                 model_name=model_name,
                                 images_ids=val_ids)

    # Validation data postprocessing
    print('\nStep 2 of 3: Validation data postprocessing... ')
    postproc_out_dir = r'out_files/images/postproc_val'
    postproc_model_name = predict_images_dir.replace("\\", "/").split("/")[-1]
    predict_images_ids = next(os.walk(os.path.join(OUT_FILES, predict_images_dir)))[1]
    for image_id in tqdm(predict_images_ids, total=len(predict_images_ids)):
        labels = dpost.read_labels(predict_images_dir, image_id)
        morfling_labels = dpost.morfling(labels=labels)
        removed_instances_labels = dpost.remove_small_instances(labels=morfling_labels)
        overlap_fix_labels = dpost.overlapping_fix(labels=removed_instances_labels)
        dpost.save_labels(labels=overlap_fix_labels,
                          out_dir=postproc_out_dir,
                          model_name=postproc_model_name,
                          image_id=image_id)

    # Check IoU score for validation data or encode submit for test data
    print('\nStep 3 of 3: Check validation score... ')
    postproc_images_dir = r'{}/{}'.format(postproc_out_dir, postproc_model_name)
    postproc_images_ids = next(os.walk(os.path.join(OUT_FILES, postproc_images_dir)))[1]
    true_images_dir = TRAIN_DATASET_DIR
    p = 0
    for image_id in tqdm(postproc_images_ids, total=len(postproc_images_ids)):
        p += metric.calculate_image_iou(image_id=image_id,
                                        true_images_dir=true_images_dir,
                                        pred_images_dir=postproc_images_dir)
    mean_p = p / len(postproc_images_ids)

    return mean_p


def submit_predict(model, config, params, model_name, val_score, test_ids):
    """
    Predicting and submission generate pipeline:
        1. Predict test data
        2. Postprocessing predicted data
        3. Encode submit
    :param model: pretrained model
    :param config: inference configurations
    :param params: parameters dictionary
    :param model_name: parameters dictionary
    :param val_score: mean IoU score for the validation set
    :param test_ids: array of test ids
    :return:
    """
    params['predict_type'] = 'test'

    # Predict data
    print('\nStep 1 of 3: Predicting test data... ')
    predict_images_dir = predict(model=model,
                                 config=config,
                                 params=params,
                                 model_name=model_name,
                                 images_ids=test_ids)

    # Data postprocessing
    print('\nStep 2 of 3: Test data postprocessing... ')
    postproc_out_dir = r'out_files/images/postproc'
    postproc_model_name = predict_images_dir.replace("\\", "/").split("/")[-1]
    predict_images_ids = next(os.walk(os.path.join(OUT_FILES, predict_images_dir)))[1]
    for image_id in tqdm(predict_images_ids, total=len(predict_images_ids)):
        labels = dpost.read_labels(predict_images_dir, image_id)
        morfling_labels = dpost.morfling(labels=labels)
        removed_instances_labels = dpost.remove_small_instances(labels=morfling_labels)
        overlap_fix_labels = dpost.overlapping_fix(labels=removed_instances_labels)
        dpost.save_labels(labels=overlap_fix_labels,
                          out_dir=postproc_out_dir,
                          model_name=postproc_model_name,
                          image_id=image_id)

    print('\nStep 3 of 3: Creating submit file... ')
    images_to_encode = os.path.join(OUT_FILES, r'{}/{}'.format(postproc_out_dir, postproc_model_name))
    submit_path = make_dir('sub/{}-V{}'.format(postproc_model_name, val_score))
    config.save_to_file(os.path.join(submit_path))
    esfmr.create_submit(files_path=images_to_encode,
                        model_name=postproc_model_name,
                        submit_path=submit_path)


if __name__ == "__main__":
    # Parameters dictionary
    params = {
        'mode': 'training',  # "training", "validate" or "predict"
        'model_type': 'mrcnn',

        # TRAINING PARAMETERS
        'init_with': 'coco',  # "imagenet", "coco", "last" or "path". If "path" is chosen,
                              # get the path from 'path_to_weights_for_train'.
        'path_to_weights_for_train': r'weights/cell20180412T2356/mask_rcnn_cell_0060.h5',
        'learning_pipeline': {'heads': 40,
                              '4+': 80
                              },
                              #'4+': 100},
        'train_dataset': 'ie',  # i - internal train dataset,
                                # e - external train dataset
                                # ie - internal + external train dataset.
        'val_split': 0.1,

        # PREDICT PARAMETERS
        'epoch_for_predict': 'last',  # int, "last" or "path". If "path" is chosen,
                                      # get the path from 'path_to_weights_for_predict'.
        'path_to_weights_for_predict': r'weights/cell20180411T1546/mask_rcnn_cell_0045.h5'
    }

    # Get images ids
    train_ids_old, test_ids = de.get_id()

    np.random.shuffle(train_ids_old)
    train_ids, val_ids = de.split_test_val(train_ids_old, params['val_split'])
    # print(len(test_ids), len(train_ids), len(val_ids))

    if params['mode'] == 'training':
        print('\n' + '-' * 30 + ' TRAIN MODEL... ' + '-' * 30 + '\n')
        train_config = CD.CellsConfig()
        train_config.save_to_file(path=os.path.join(ROOT_DIR, r'weights'))

        # Create model object in "training" mode
        model = modellib.MaskRCNN(mode="training",
                                  config=train_config,
                                  model_dir=MODEL_DIR
                                  )

        # Train the model
        train_model(model=model,
                    params=params,
                    config=train_config,
                    train_ids=train_ids,
                    val_ids=val_ids)

    elif params['mode'] == 'validate':
        pred_config = CD.CellsConfigInference()
        # Create model object in "inference" mode
        model = modellib.MaskRCNN(mode="inference",
                                  config=pred_config,
                                  model_dir=MODEL_DIR)

        # Get model name
        model_name = get_model_name(model=model, params=params)

        # Validate the model
        print('\n' + '-' * 30 + ' VALIDATE MODEL... ' + '-' * 30 + '\n')
        mean_iou_score = validate(model=model,
                                  config=pred_config,
                                  params=params,
                                  model_name=model_name,
                                  val_ids=val_ids)
        print('\n\nTotal IoU for validation set: {:1.3f}'.format(mean_iou_score))

    elif params['mode'] == 'predict':
        pred_config = CD.CellsConfigInference()

        # Create model object in "inference" mode
        model = modellib.MaskRCNN(mode="inference",
                                  config=pred_config,
                                  model_dir=MODEL_DIR)

        # Get model name
        model_name = get_model_name(model=model, params=params)

        # Predict and submit without validating
        print('\n' + '-' * 30 + 'PREDICT AND SUBMIT... ' + '-' * 30 + '\n')
        val_score = 0
        submit_predict(model=model,
                       config=pred_config,
                       params=params,
                       model_name=model_name,
                       val_score='{:1.3f}'.format(val_score),
                       test_ids=test_ids)

