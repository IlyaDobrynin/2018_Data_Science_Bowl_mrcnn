import sys
import argparse
from src.nn_var.mask_rcnn.train import *
from dirs import MODEL_DIR

def training_pipeline(s):
    try:
        values = s.split(",")
        layer, epochs = values[0], int(values[1])
        return layer, epochs
    except:
        raise argparse.ArgumentTypeError("Bad format")


parser = argparse.ArgumentParser(description='Mask RCNN impementation for DSB2018')
parser.add_argument("-m", "--mode",
                    default='train',
                    metavar="<mode>",
                    help="'train', 'validate' or 'predict'")
parser.add_argument("-tm", "--training_mode",
                    required=False,
                    default="coco",
                    metavar="",
                    help="Initial weights. Can be 'imagenet', 'coco', 'last' or 'path' (default 'coco')")
parser.add_argument("-tw", "--train_weights_path",
                    required=False,
                    default=None,
                    metavar="",
                    help="Path to initial weights to train with")
parser.add_argument("-tp", "--training_pipeline",
                    nargs="+",
                    type=training_pipeline,
                    default=[("heads", 40), ("4+", 80)],
                    required=False,
                    metavar="",
                    help="Training pipeline in format layer_to_train,epochs "
                         "where layer_to_train - 'heads', '5+', '4+', '3+' or all; "
                         "epochs - overall amount of training epochs "
                         "(default: heads,40 '4+',80)")
parser.add_argument("-vs", "--val_split",
                    required=False,
                    metavar="",
                    default=0.1,
                    help="Train dataset split factor (0 < vs <= 1) to make validation data (default 0.1)")
parser.add_argument("-pm", "--predict_mode",
                    metavar="",
                    required=False,
                    default="last",
                    help="Weights to predict with. Can be integer number, 'last' or 'path' (default 'last')")
parser.add_argument("-pw", "--pred_weights_path",
                    metavar="",
                    default=None,
                    required=False,
                    help="Path to weights to predict with")

args = parser.parse_args()

modes = ('train', 'validate', 'predict')
training_mode = ('coco', 'imagenet', 'last', 'path')
predict_mode = ('last', 'path')

if args.mode not in modes:
    print("ERROR: Parameter 'mode' is incorrect. "
          "Please insert one of these values: 'train', 'validate' or 'predict'")
    sys.exit()
if args.training_mode not in training_mode:
    print("ERROR: parameter 'training_mode' is incorrect. "
          "Please insert one of these values: 'coco', 'imagenet', 'last' or 'path'")
    sys.exit()
if args.predict_mode not in predict_mode and args.pred_weights.isdigit() is False:
    print(args.pred_weights.isdigit(), type(args.pred_weights))
    print("ERROR: parameter 'predict_mode' is incorrect. "
          "Please insert one of these values: integer number, 'last' or 'path'")
    sys.exit()

params = {
    'mode': args.mode,
    'model_type': 'mrcnn',

    # TRAINING PARAMETERS
    'init_with': args.training_mode,
    'learning_pipeline': dict(args.training_pipeline),
    'val_split': args.val_split,

    # VALIDATION/PREDICT PARAMETERS
    'epoch_for_predict': args.predict_mode
}

# Check if all paths are here
if args.mode == modes[0]:
    if args.training_mode == training_mode[3]:
        assert args.train_weights_path, "Arguments --path_to_train_weights is required for training"
        params['path_to_weights_for_train'] = args.train_weights_path
if args.mode == modes[1]:
    if args.predict_mode == predict_mode[1]:
        assert args.pred_weights_path, "Arguments --path_to_pred_weights is required for validation"
        params['path_to_weights_for_predict'] = args.pred_weights_path
if args.mode == modes[2]:
    if args.predict_mode == predict_mode[1]:
        assert args.pred_weights_path, "Arguments --path_to_pred_weights is required for prediction"
        params['path_to_weights_for_predict'] = args.pred_weights_path

print("Running parameters are: {}\n".format(params))

# Get images ids
train_ids_old, test_ids = de.get_id()

np.random.shuffle(train_ids_old)
train_ids, val_ids = de.split_test_val(train_ids_old, params['val_split'])
# print(len(test_ids), len(train_ids), len(val_ids))

if params['mode'] == modes[0]:
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

elif params['mode'] == modes[1]:
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

    # Predict and submit without validation
    print('\n' + '-' * 30 + 'PREDICT AND SUBMIT... ' + '-' * 30 + '\n')
    val_score = 0
    submit_predict(model=model,
                   config=pred_config,
                   params=params,
                   model_name=model_name,
                   val_score='{:1.3f}'.format(val_score),
                   test_ids=test_ids)

