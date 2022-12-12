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
|NO.1 | -| - | - | - | - | - | 
|NO.2 |0.858 | 0.885 |0.926 |6.016| 5.831| 3.770| 
|NO.3 |0.847 | 0.878 |0.928 |9.335 |7.564 |3.470 |
|NO.4 |0.8480| 0.8796 | 0.9253 | 14.178| 5.862 | 3.455 |
|NO.5| 0.8197   |0.87837| 0.92723 | - | - | - |
|NO.6| 0.8451    |0.8781|0.9275 |20.73 | 7.623 |3.47|
|NO.7| 0.8600      |0.8868| 0.9265 |9.0541| 5.8409 |3.6009|
|**HarDNet-BTS** | **0.8442**   |  **0.8793**| **0.9260**| **12.592**| **7.073**| **3.884**|
