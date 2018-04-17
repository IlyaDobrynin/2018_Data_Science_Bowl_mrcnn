1. Put train dataset into "data/internal_external/train" folder
2. Put test dataset into "data/test" folder
3. Put pretrained coco weights into "src/nn_var/mask_rcnn/coco_model"
4. Change paths in dirs.py
5. Train the model: run "src/nn_var/mask_rcnn/train.py" file with params['mode'] = 'training'.
   Set your parameters in params dictionary.
6. Predict: run "src/nn_var/mask_rcnn/train.py" file with params['mode'] = 'predict'