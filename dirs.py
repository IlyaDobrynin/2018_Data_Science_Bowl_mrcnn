import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset folders
DATASET_DIR = r'C:/Programming/nuclei/data'                 # Change to your path
train_folder = 'train'                                      # Change to the name of your train data folder
test_folder = 'test'                                        # Change to the name of your test data folder
extra_data_folder = 'extra_data'                            # Change to the name of your extra data folder

# Path to save files
OUT_FILES = ROOT_DIR                                        # Change to your path

TRAIN_DATASET_DIR = os.path.join(DATASET_DIR, train_folder)
TEST_DATASET_DIR = os.path.join(DATASET_DIR, test_folder)
EXTERNAL_DATA = os.path.join(DATASET_DIR, extra_data_folder)

# Path to model weights
MODEL_DIR = os.path.join(ROOT_DIR, r'weights')
# Local path to trained coco weights
COCO_MODEL_PATH = os.path.join(ROOT_DIR, r'src/nn_var/mask_rcnn/coco_model/mask_rcnn_coco.h5')


def make_dir(relative_path, top_dir=ROOT_DIR):
    dirs = relative_path.replace("\\", "/").split('/')
    for d in dirs:
        top_dir = os.path.join(top_dir, d)
        if not os.path.exists(top_dir):
            os.mkdir(top_dir)

    return top_dir
