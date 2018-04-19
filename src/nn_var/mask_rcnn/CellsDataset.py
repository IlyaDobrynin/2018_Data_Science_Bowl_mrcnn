import os
import numpy as np
import skimage.io
import configparser
from src.nn_var.mask_rcnn.mrcnn.config import Config
import src.nn_var.mask_rcnn.mrcnn.utils as utils
from dirs import *

config = configparser.ConfigParser()
config.read(os.path.join(ROOT_DIR, 'hyper.cfg'))


class CellsConfig(Config):
    parameters = config['CellsConfig']

    NAME = "cell"
    GPU_COUNT = int(parameters['GPU_COUNT'])
    IMAGES_PER_GPU = int(parameters['IMAGES_PER_GPU'])  # 1
    NUM_CLASSES = int(parameters['NUM_CLASSES'])  # 1 + 1
    STEPS_PER_EPOCH = int(parameters['STEPS_PER_EPOCH']) // (IMAGES_PER_GPU * GPU_COUNT)
    VALIDATION_STEPS = int(parameters['VALIDATION_STEPS']) // (IMAGES_PER_GPU * GPU_COUNT)
    DETECTION_NMS_THRESHOLD = float(parameters['DETECTION_NMS_THRESHOLD'])  # 0.1
    DETECTION_MIN_CONFIDENCE = float(parameters['DETECTION_MIN_CONFIDENCE'])  # 0.1
    BACKBONE = parameters['BACKBONE'] # "resnet101"
    IMAGE_RESIZE_MODE = parameters['IMAGE_RESIZE_MODE']  # "crop"
    IMAGE_MIN_DIM = int(parameters['IMAGE_MIN_DIM'])  # 512
    IMAGE_MAX_DIM = int(parameters['IMAGE_MAX_DIM'])  # 512
    IMAGE_MIN_SCALE = float(parameters['IMAGE_MIN_SCALE'])  # 1.0
    RPN_ANCHOR_SCALES = (int(parameters['RPN_ANCHOR_SCALES_1']),
                         int(parameters['RPN_ANCHOR_SCALES_2']),
                         int(parameters['RPN_ANCHOR_SCALES_3']),
                         int(parameters['RPN_ANCHOR_SCALES_4']),
                         int(parameters['RPN_ANCHOR_SCALES_5']))  # (16, 32, 64, 128, 256)
    POST_NMS_ROIS_TRAINING = int(parameters['POST_NMS_ROIS_TRAINING'])  # 2000
    POST_NMS_ROIS_INFERENCE = int(parameters['POST_NMS_ROIS_INFERENCE'])  # 1000
    RPN_NMS_THRESHOLD = float(parameters['RPN_NMS_THRESHOLD']) # 0.9
    RPN_TRAIN_ANCHORS_PER_IMAGE = int(parameters['RPN_TRAIN_ANCHORS_PER_IMAGE'])  # 64
    LEARNING_RATE = float(parameters['LEARNING_RATE'])  # 0.003
    TRAIN_ROIS_PER_IMAGE = int(parameters['TRAIN_ROIS_PER_IMAGE'])  # 128
    DETECTION_MAX_INSTANCES = int(parameters['DETECTION_MAX_INSTANCES'])  # 300
    MAX_GT_INSTANCES = int(parameters['MAX_GT_INSTANCES'])  # 300


class CellsConfigInference(CellsConfig):
    parameters = config['CellsConfigInference']

    GPU_COUNT = int(parameters['GPU_COUNT'])  # 1
    IMAGES_PER_GPU = int(parameters['IMAGES_PER_GPU'])  # 1
    IMAGE_RESIZE_MODE = parameters['IMAGE_RESIZE_MODE']  # "pad64"
    RPN_NMS_THRESHOLD = float(parameters['RPN_NMS_THRESHOLD'])  # 0.9


class CellsDataset(utils.Dataset):

    def load_cells(self, dataset_dir, image_ids):
        self.add_class("cells", 1, "cell")
        for i in image_ids:
            self.add_image("cells",
                           image_id=i,
                           path=os.path.join(dataset_dir, r"{}/images/{}.png".format(i, i)),
                           path_mask=os.path.join(dataset_dir, i, 'masks'))

    def load_mask(self, image_id):

        instance_masks = []
        path_mask = self.image_info[image_id]['path_mask']
        masks_names = next(os.walk(path_mask))[2]

        for i, mask in enumerate(masks_names):
            if mask.split('.')[-1] != 'png':
                continue
            img = skimage.io.imread(os.path.join(path_mask, mask))
            instance_masks.append(img)

        masks = np.stack(instance_masks, axis=2)
        class_ids = np.ones(shape=(len(masks_names)), dtype=np.int32)

        return masks, class_ids

if __name__ == '__main__':
    cfg = CellsConfig()
