# 0 Intro
This is a Mask-RCNN-based solution to the [Kaggle 2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018)
* Stage 1 public score ~0.490
* Stage 2 private score 0.569 — 34th place

# 1 Hardware requirements
My specs:
* Intel i5-4690
* Nvidia GeForce 970
Training time — ~900s per epoch

# 2 Preparing the data
1. Clone 
1. Put train dataset into "data/internal_external/train" folder
2. Put test dataset into "data/test" folder
3. Put pretrained coco weights into "src/nn_var/mask_rcnn/coco_model"
4. Change paths in dirs.py
5. Train the model: run "src/nn_var/mask_rcnn/train.py" file with params['mode'] = 'training'.
   Set your parameters in params dictionary.
6. Predict: run "src/nn_var/mask_rcnn/train.py" file with params['mode'] = 'predict'

# 3 Training

# 4 Validation

# 5 Predict and Submit
