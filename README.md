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
1. Download dataset from [**official page**](https://www.kaggle.com/c/data-science-bowl-2018/data) and official external dataset from [**this page**](https://www.kaggle.com/voglinio/bowl2018-external) and put it in your dataset root directory
2. Change path in **configs.ini** to your external dataset folder:
```
      dataset_dir = Your dataset root directory
      extra_data_folder = Name of the folder with external data
```
4. Run **src/utils/ext_data_preprocessing.py** (see `python src/utils/ext_data_preprocessing.py -h` for parameters) to split external images from 1000x1000 px to 500x500 px.
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
1. For the model training process you will need the COCO pretrained weights. You can download them [**here**](https://yadi.sk/d/WBcgk3yA3UWzkt). Put them into **src/nn_var/mask_rcnn/coco_model**.
2. The model weights from the 34 place of the competition are [**here**](https://yadi.sk/d/O6FNJ0cd3UX4Pp). Put them into **weights** folder.

# 3 Train, validate and predict
1. Change paths in **configs.ini** such as:
```
      dataset_dir = Your data directory
      train_folder = Name of your train data folder
      test_folder = Name of your test data folder
      extra_data_folder = Name of your external data folder
```
2. Make sure that you had downloaded pretrained COCO weights (**2.2**, 1) and they are in the **src/nn_var/mask_rcnn/coco_model**.
3. Run **pipeline.py**:
```
python pipeline.py -m train/validate/predict
```
For the options description run 
```
python pipeline.py -h
```
# Change model hyperparameters
The LB0.569-model hyperparameters are in the **hyper.ini**.
