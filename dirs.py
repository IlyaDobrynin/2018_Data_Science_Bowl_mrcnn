import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset folder
TRAIN_DATASET_DIR = r'C:/Programming/nuclei/data/train'   # Change for your path
TEST_DATASET_DIR = r'C:/Programming/nuclei/data/test'     # Change for your path
EXTERNAL_DATA = r'C:/Programming/nuclei/data/extra_data'  # Change for your path

# Path to save files
OUT_FILES = ROOT_DIR                                      # Change for your path

# Path to model weights
MODEL_DIR = os.path.join(ROOT_DIR, r'weights')

# Local path to trained coco weights
COCO_MODEL_PATH = os.path.join(ROOT_DIR, r'src/nn_var/mask_rcnn/coco_model/mask_rcnn_coco.h5')


def make_dir(relative_path):
    dirs = relative_path.replace("\\", "/").split('/')
    top_dir = ROOT_DIR
    for dir in dirs:
        top_dir = os.path.join(top_dir, dir)
        if not os.path.exists(top_dir):
            os.mkdir(top_dir)

    return top_dir
