from src.nn_var.mask_rcnn.mrcnn.config import Config
import src.nn_var.mask_rcnn.mrcnn.utils as utils
import os
import numpy as np
import skimage.io


class CellsConfig(Config):
    # Give the configuration a recognizable name
    NAME = "cell"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # background + cells class
    STEPS_PER_EPOCH = 764 // (IMAGES_PER_GPU * GPU_COUNT)
    VALIDATION_STEPS = 10 // (IMAGES_PER_GPU * GPU_COUNT)
    DETECTION_NMS_THRESHOLD = 0.1
    DETECTION_MIN_CONFIDENCE = 0.1
    BACKBONE = "resnet101"
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 1.0
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000
    RPN_NMS_THRESHOLD = 0.9
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64
    # MEAN_PIXEL = np.array([43.53, 39.56, 48.22])
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    LEARNING_RATE = 0.003
    TRAIN_ROIS_PER_IMAGE = 128
    DETECTION_MAX_INSTANCES = 300
    MAX_GT_INSTANCES = 300


class CellsConfigInference(CellsConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize images for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9


class CellsDataset(utils.Dataset):

    def load_cells(self, dataset_dir, image_ids):
        self.add_class("cells", 1, "cell")
        for i in image_ids:
            self.add_image("cells", image_id=i,
                           # width=width, height=height,
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

