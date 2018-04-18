# Credits
* Mask-RCNN implementation based on the brilliant [Matterport code](https://github.com/matterport/Mask_RCNN)
* RLE encoding based on the best [rakhlin fast run length encoding](https://www.kaggle.com/rakhlin/fast-run-length-encoding-python)
* Code for metric estimation - [William Cukierski kernel](https://www.kaggle.com/wcukierski/example-metric-implementation)

# 0 Intro
This is a Mask-RCNN-based solution to the [Kaggle 2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018).
* Stage 1 public score ~0.490
* Stage 2 private score 0.569 — 34th place

# 1 Hardware requirements
My specs:
* Intel i5-4690
* Nvidia GeForce 970
Training time ~900s per epoch with pre-defined parameters

# 2 Preparing the data
1. Clone this repo:

      `git clone https://github.com/snakers4/ds_bowl_2018`
2. Dataset.
      2.1 Download dataset from [official page](https://www.kaggle.com/c/data-science-bowl-2018/data) and put files into your directory like this:
```
    ├──data
    │   ├── stage1_test                 <- A folder with stage1 test data
    │   ├── stage2_test                 <- A folder with stage2 test data
    │   └── stage1_train                <- A folder with stage1 train data
    │        ├─ f8e74d4006dd68c1dbe68df7be905835e00d8ba4916f3b18884509a15fdc0b55
    │        │    ├──  images
    │        │    └──  masks

        ...
        

    └────────└─ ff599c7301daa1f783924ac8cbe3ce7b42878f15a39c2d19659189951f540f48
```
    
2. Put test dataset into "data/test" folder
3. Put pretrained coco weights into "src/nn_var/mask_rcnn/coco_model"
4. Change paths in dirs.py
5. Train the model: run "src/nn_var/mask_rcnn/train.py" file with params['mode'] = 'training'.
   Set your parameters in params dictionary.
6. Predict: run "src/nn_var/mask_rcnn/train.py" file with params['mode'] = 'predict'

# 3 Training

# 4 Validation

# 5 Predict and Submit
