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
      * Download dataset from [official page](https://www.kaggle.com/c/data-science-bowl-2018/data)
      * Download official external dataset from [this page](https://www.kaggle.com/voglinio/bowl2018-external)
      * Run **src/utils/ext_data_preprocessing.py** to split external images from 1000x1000 px to 500x500 px
      * At the end of steps above you shout get similar structure:
`
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
         │     └─ ff3407842ada5bc18be79ae453e5bdaa1b68afc842fc22fa618ac6e6599d0bb3
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
`
      * Make train dataset - put all files from **external_data_splited** to **stage1_train**
2. Put test dataset into "data/test" folder
3. Put pretrained coco weights into "src/nn_var/mask_rcnn/coco_model"
4. Change paths in dirs.py
5. Train the model: run "src/nn_var/mask_rcnn/train.py" file with params['mode'] = 'training'.
   Set your parameters in params dictionary.
6. Predict: run "src/nn_var/mask_rcnn/train.py" file with params['mode'] = 'predict'

# 3 Training

# 4 Validation

# 5 Predict and Submit
