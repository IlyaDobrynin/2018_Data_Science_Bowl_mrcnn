# Credits
* Mask-RCNN implementation based on the brilliant [Matterport](https://github.com/matterport/Mask_RCNN) code
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
## 2.1 Datasets
1. Download dataset from [official page](https://www.kaggle.com/c/data-science-bowl-2018/data)
2. Download official external dataset from [this page](https://www.kaggle.com/voglinio/bowl2018-external)
3. Change path in **dirs.py** to your external dataset folder:
`EXTERNAL_DATA = Your external data directory`
4. Run **src/utils/ext_data_preprocessing.py** with your `out_path` parameter to split external images from 1000x1000 px to 500x500 px
5. Make train dataset - put all files from **external_data_splited** to **stage1_train**

**At the end of steps above you shout get similar structure:**
 ``` 
    └─ data
         ├── stage1_test                 <- A folder with stage1 test data
         ├── stage2_test                 <- A folder with stage2 test data
         ├── stage1_train                <- A folder with stage1 train data
         │     ├─ 00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e
         │     │    ├──  images
         │     │    └──  masks
         │     .
         │     .
         │     .
         │     └─ TCGA-RD-A8N9-01A-01-TS1-4
         ├── external_data
         │     ├─ TCGA-18-5592-01Z-00-DX1
         │     │   ├──  images
         │     │   └──  masks
         │     .
         │     .
         │     .
         │     └─ TCGA-RD-A8N9-01A-01-TS1
         └── external_data_splited
               ├─ TCGA-18-5592-01Z-00-DX1-1
               │   ├──  images
               │   └──  masks
               .
               .
               .
               └─ TCGA-RD-A8N9-01A-01-TS1-4
```               
## 2.2 Pretrained weights
1. For the model training process you will need the COCO pretrained weights. You can download them [**here**](https://yadi.sk/d/WBcgk3yA3UWzkt)
2. The model weights from the 34 plae of the competition are [**here**](https://yadi.sk/d/O6FNJ0cd3UX4Pp)

# 3 Training
1. Change paths in **dirs.py** such as:
```
      TRAIN_DATASET_DIR = Your train directory
      TEST_DATASET_DIR =  Your train directory
```
2. Make sure that you had downloaded pretrained COCO weights (**2.2**, 1) and they are in the **src/nn_var/mask_rcnn/coco_model**
3. Set `params['mode'] = training`.
4. Set your parameters in params dictionary. My parameters are:
```
'mode': 'training',
'model_type': 'mrcnn',
'init_with': 'coco', 
'path_to_weights_for_train': None,
'learning_pipeline': {'heads': 40,'4+': 80},
'train_dataset': 'ie',
'val_split': 0.1,
'epoch_for_predict': 'last',
'path_to_weights_for_predict': None
```
4. If you want to start training from pretrained weights, set `params['init_with'] = path` and `params['path_to_weights_for_train'] = Path to pretrained weights`.
5. Run the **train.py** script.

# 4 Validation
Validation allow you locally predict the mean IoU score, achieved with trained model.
1. Set `params['mode'] = validate`.
2. Set `params['epoch_for_predict'] = last/NN/path`, where:
      * last - predict validation masks using the last trained weights of your model
      * NN - predict validation masks using trained weights with given number
      * path - predict validation masks using weights from the path. If you set this parameter, also change `params['path_to_weights_for_predict']`
3. Run the **train.py** script.

# 5 Predict and Submit
1. Set `params['mode'] = predict`.
2. Set `params['epoch_for_predict']` just like in the Validation step.
3. The output submut will be in the **sub** folder.

