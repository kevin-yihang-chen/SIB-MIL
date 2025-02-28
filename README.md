# [Under Review] SIB-MIL: Sparsity-Induced Bayesian Neural Network for Robust Multiple Instance Learning on Whole Slide Image Analysis

[//]: # (This repository holds the Pytorch implementation for the ReMix augmentation described in the paper )

[//]: # (> [**ReMix: A General and Efficient Framework for Multiple Instance Learning based Whole Slide Image Classification**]&#40;https://arxiv.org/abs/2207.01805&#41;,  )

[//]: # (> Jiawei Yang, Hanbo Chen, Yu Zhao, Fan Yang,  Yao Zhang, Lei He, and Jianhua Yao    )

[//]: # (> International Conference on Medical Image Computing and Computer Assisted Intervention &#40;MICCAI&#41;, 2022 )



<p align="center">
  <img src="Framework-min.png" width="1000">
</p>


[//]: # (# Installation)

[//]: # ()
[//]: # (We use [Remix]&#40;https://github.com/1st-Yasuo/ReMix&#41; as the original codebase.)

# Data Download
We use three dataset projects in our paper for demonstration: 1) [Camelyon16](https://camelyon16.grand-challenge.org/), 2) [TCGA](https://portal.gdc.cancer.gov/) and 3) [BRACS](https://www.bracs.icar.cnr.it/). 

You may follow the instructions in the websites to download the data.

# Crop Slide and Feature Extraction
We crop slides with magnification parameter set to 20 (level 0) and features are extracted using pretrained KimiaNet. We followed the pipeline of [DSMIL](https://github.com/binli123/dsmil-wsi).

[//]: # (For implementation details, please refer to our previous project [WSI-HGNN]&#40;https://github.com/HKU-MedAI/WSI-HGNN&#41;.)

# Model Training

```shell
python main.py --backbone abmil --num_epochs 100 --dataset BRCA --task staging --feats_size 1024 --extractor Kimia --num_workers 1 --num_rep 1 --wandb
```


