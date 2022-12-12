# HarDNet-BTS: An Efficient 3D CNN for Brain Tumor Segmentation
> BrainLes 2021 Paper : [**HarDNet-BTS: A Harmonic Shortcut Network for Brain Tumor Segmentation**](https://link.springer.com/chapter/10.1007/978-3-031-08999-2_21)

> ICCV 2019 Paper : [**HarDNet: A Low Memory Traffic Network**](https://arxiv.org/abs/1909.00948)

## HarDNet Family
#### For Image Classification : [HarDNet](https://github.com/PingoLH/Pytorch-HarDNet) A Low Memory Traffic Network
#### For Object Detection : [CenterNet-HarDNet](https://github.com/PingoLH/CenterNet-HarDNet) 44.3 mAP / 45 fps on COCO Dataset
#### For Semantic Segmentation : [FC-HarDNet](https://github.com/PingoLH/FCHarDNet)  76.0 mIoU / 53 fps on Cityscapes Dataset
#### For Polyp Segmentation : [HarDNet-MSEG](https://github.com/james128333/HarDNet-MSEG) 90.4% mDice / 119 FPS on Kvasir-SEG @352x352

## Main results
<p align="center"> <img src='imgs/validaitonphase8th.png' align="center" height="300px"> </p>

### Performance on RSNA-ASNR-MICCAI Brain Tumor Segmentation (BraTS) Challenge 2021 Dataset
Validation Phase

| Models       |     |Dice|        |   | HD95   ||  
| :----------: | :----: | :----: | :-----------: | :--------: | :------------: | :---------------: |
|       |**ET**     |**TC**| **WT**       |**ET**   | **TC**   |**WT**| 
|NO.1 | 0.471| 0.597| 0.598 |0.672 | 0.617| 0.894| 
|NO.2 |0.572| 0.690 |0.699 |0.745| 0.725| 0.917| 
|NO.3 |0.733| 0.813 |0.820 |0.861 |0.840 |0.949 |
|NO.4 |0.776| 0.857 |0.855 |0.891| 0.8616 |0.961 |
|NO.5| 0.810 |0.876| 0.862 |0.944| 0.860 |0.968|
|NO.6| 0.810 |0.876| 0.862 |0.944| 0.860 |0.968|
|NO.7| 0.810 |0.876| 0.862 |0.944| 0.860 |0.968|
|**HarDNet-BTS** | **0.8442**   |  **0.8793**| **0.9260**| **12.592**| **7.073**| **3.884**|
