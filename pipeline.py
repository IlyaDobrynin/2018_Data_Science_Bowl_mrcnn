from src.nn_var.mask_rcnn.train import *

params = {
    'mode': 'training',  # "training", "validate" or "predict"
    'model_type': 'mrcnn',

    # TRAINING PARAMETERS
    'init_with': 'coco',  # "imagenet", "coco", "last" or "path". If "path" is chosen,
    # get the path from 'path_to_weights_for_train'.
    'path_to_weights_for_train': r'weights/cell20180412T2356/mask_rcnn_cell_0060.h5',
    'learning_pipeline': {'heads': 40, '4+': 80},
    'train_dataset': 'ie',  # i - internal train dataset,
                            # e - external train dataset
                            # ie - internal + external train dataset.
    'val_split': 0.1,

    # VALIDATION/PREDICT PARAMETERS
    'epoch_for_predict': 'path',  # "int", "last" or "path". If "path" is chosen,
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