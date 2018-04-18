![](/images/2018-04-18_16-26-12.png)

# Credits
* Mask-RCNN implementation based on the brilliant [**Matterport**](https://github.com/matterport/Mask_RCNN) code
* RLE encoding based on the best [**rakhlin fast run length encoding**](https://www.kaggle.com/rakhlin/fast-run-length-encoding-python)
* Code for metric estimation - [**William Cukierski kernel**](https://www.kaggle.com/wcukierski/example-metric-implementation)

# 0 Intro
This is a Mask-RCNN-based solution to the [**Kaggle 2018 Data Science Bowl**](https://www.kaggle.com/c/data-science-bowl-2018).
* Stage 1 public score ~0.490
* Stage 2 private score 0.569 — 34th place:
![](/images/2018-04-18_16-14-00.png)

# 1 Hardware requirements
My specs:
* Intel i5-4690
* Nvidia GeForce 970

Training time ~900s per epoch with pre-defined parameters

# 2 Preparing the data
## 2.1 Datasets
1. Download dataset from [**official page**](https://www.kaggle.com/c/data-science-bowl-2018/data) and official external dataset from [**this page**](https://www.kaggle.com/voglinio/bowl2018-external)and put it in your dataset root directory
2. Change path in **dirs.py** to your external dataset folder:
```
DATASET_DIR = Your dataset root directory
EXTERNAL_DATA = os.path.join(DATASET_DIR, Name of the external data folder)
```
4. Run **src/utils/ext_data_preprocessing.py** (`out_path = Name of the out folder`) to split external images from 1000x1000 px to 500x500 px.
5. Make train dataset - put all files from **external_data_splited** to **train**.

**At the end of steps above you shout get similar structure:**
 ``` 
    └─ data
         ├── test                        <- A folder with stage 2 test data
         ├── train                       <- A folder with train data
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
1. For the model training process you will need the COCO pretrained weights. You can download them [**here**](https://yadi.sk/d/WBcgk3yA3UWzkt).
2. The model weights from the 34 place of the competition are [**here**](https://yadi.sk/d/O6FNJ0cd3UX4Pp).

# 3 Training
1. Change paths in **dirs.py** such as:
```
      DATASET_DIR = Your data directory
      train_folder = Name of your train data folder
      test_folder = Name of your test data folder
      extra_data_folder = Xame of your external data folder
```
2. Make sure that you had downloaded pretrained COCO weights (**2.2**, 1) and they are in the **src/nn_var/mask_rcnn/coco_model**.
3. Open **pipeline.py**. Set `params['mode'] = training` for starting in the training mode.
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
5. Run the **pipeline.py**.

# 4 Validation
Validation allow you locally predict the mean IoU score, achieved with trained model.
At the **pipeline.py**:
1. Set `params['mode'] = validate`.
2. Set `params['epoch_for_predict'] = last/NN/path`, where:
      * last - predict validation masks using the last trained weights of your model
      * NN - predict validation masks using trained weights with given number
      * path - predict validation masks using weights from the path. If you set this parameter, also change       `params['path_to_weights_for_predict']`
3. Run the **pipeline.py** script.

# 5 Predict and Submit
At the **pipeline.py**:
1. Set `params['mode'] = predict`.
2. Set `params['epoch_for_predict']` just like in the Validation step.
3. Run the **pipeline.py** script.
4. The output submit will be in the **sub** folder.

