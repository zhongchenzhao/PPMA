# Polyline Path Masked Attention for Vision Transformer

Official codes for **Polyline Path Masked Attention for Vision Transformer (PPMA)**





## Updates

* ***2025.06.12*** We have released the complete training and inference code and training logs!









## Getting started

**TL;DR:** **We adaptes Mamba2's structured mask to 2D scaning and integrates it into the self-attention mechanism of ViTs as an explicit positional encoding.**



![contribution](./figures/contribution.png)











## Results



#### Image Classification Results on ImageNet-1K 224x224

|   Model    | #Params | FLOPs |  Acc  |                         Training log                         | Checkpoint |
| :--------: | :-----: | :---: | :---: | :----------------------------------------------------------: | :--------: |
| PPMA-Tiny  |   14M   | 2.7G  | 82.6% | [PPMA-Tiny](./training_logs/classification/ppma_tiny_log.txt) |            |
| PPMA-Small |   27M   | 4.9G  | 84.2% | [PPMA-Small](./training_logs/classification/ppma_small_log.txt) |            |
| PPMA-Base  |   54M   | 10.6G | 85.0% | [PPMA-Base](./training_logs/classification/ppma_base_log.txt) |            |



#### Object Detection and Instance Segmentation Results on COCO with Mask R-CNN Method (1× schedule)

|  Backbone  | Pretrained Model | #Params | FLOPs | box mAP | mask mAP |                            Config                            |                         Training log                         | Checkpoint |
| :--------: | :--------------: | :-----: | :---: | :-----: | :------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :--------: |
| PPMA-Tiny  |                  |   33M   | 218G  |  47.1   |   42.4   | [PPMA-Tiny](./detection/configs/mask_rcnn_ppma_tiny_fpn_1x_coco.py) | [PPMA-Tiny](./training_logs/detection/mask_rcnn_ppma_tiny_fpn_1x_coco.log) |            |
| PPMA-Small |                  |   46M   | 263G  |  49.2   |   43.8   | [PPMA-Small](./detection/configs/mask_rcnn_ppma_small_fpn_1x_coco.py) | [PPMA-Small](./training_logs/detection/mask_rcnn_ppma_small_fpn_1x_coco.log) |            |
| PPMA-Base  |                  |   73M   | 374G  |  51.1   |   45.5   | [PPMA-Base](./detection/configs/mask_rcnn_ppma_base_fpn_1x_coco.py) | [PPMA-Base](./training_logs/detection/mask_rcnn_ppma_base_fpn_1x_coco.log) |            |



#### Semantic Segmentation Results on ADE20K with UPerNet Method (batch size=16)

|  Backbone  | Pretrained Model | Input Size | #Params | FLOPs | mIoU (SS) | mIoU (MS) |                            Config                            |                         Training log                         | Checkpoint |
| :--------: | :--------------: | :--------: | :-----: | :---: | :-------: | :-------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :--------: |
| PPMA-Tiny  |                  |  512×512   |   43M   | 983G  |   48.7    |   49.1    | [PPMA-Tiny](./segmentation/configs/upernet_ppma_tiny_512x512_160k_ade20k_ss.py) | [PPMA-Tiny](./training_logs/segmentation/upernet_ppma_tiny_512x512_160k_ade20k_ss.log) |            |
| PPMA-Small |                  |  512×512   |   56M   | 984G  |   51.1    |   52.0    | [PPMA-Small](./segmentation/configs/upernet_ppma_small_512x512_160k_ade20k_ss.py) | [PPMA-Small](./training_logs/segmentation/upernet_ppma_small_512x512_160k_ade20k_ss.log) |            |
| PPMA-Base  |                  |  512×512   |   83M   | 1137G |   52.3    |   53.0    | [PPMA-Base](./segmentation/configs/upernet_ppma_base_512x512_160k_ade20k_ss.py) | [PPMA-Base](./training_logs/segmentation/upernet_ppma_base_512x512_160k_ade20k_ss.log) |            |



#### Semantic Segmentation Results on Cityscapes with UPerNet Method (batch size=8)

|  Backbone  | Pretrained Model | Input Size | #Params | FLOPs | mIoU (SS) | mIoU (MS) |                            Config                            |                         Training log                         | Checkpoint |
| :--------: | :--------------: | :--------: | :-----: | :---: | :-------: | :-------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :--------: |
| PPMA-Tiny  |                  | 1024×1024  |   43M   | 983G  |   82.7    |   83.5    | [PPMA-Tiny](./segmentation/configs/cityscapes/upernet_ppma_tiny_1024x1024_160k_cityscapes_ss.py) | [PPMA-Tiny](./training_logs/segmentation/upernet_ppma_tiny_1024x1024_160k_cityscapes_ss.log) |            |
| PPMA-Small |                  | 1024×1024  |   56M   | 984G  |   83.7    |   84.0    | [PPMA-Small](./segmentation/configs/cityscapes/upernet_ppma_small_1024x1024_160k_cityscapes_ss.py) | [PPMA-Small](./training_logs/segmentation/upernet_ppma_small_1024x1024_160k_cityscapes_ss.log) |            |
| PPMA-Base  |                  | 1024×1024  |   83M   | 1137G |   83.9    |   84.3    | [PPMA-Base](./segmentation/configs/cityscapes/upernet_ppma_base_1024x1024_160k_cityscapes_ss.py) | [PPMA-Base](./training_logs/segmentation/upernet_ppma_base_1024x1024_160k_cityscapes_ss.log) |            |



#### Mask Visualization



![contribution](./figures/Mask_visualization.png)



#### Masked Attention Visualization

![contribution](./figures/ILSVRC2012_val_00018280_atten_map.gif)



![contribution](./figures/MaskedAttentionMap.png)







## How to use

#### 1. Requirements

```
conda create -n zzc_ppma python=3.8.20 -y
conda activate zzc_ppma
chmod +x install_requirements.sh 
./install_requirements.sh 
```



#### 2. Training on ImageNet-1K

```
cd classification/

# PPMA-Tiny
torchrun --nproc_per_node=8 --master_port=29501 main.py --warmup_epochs 5 --model PPMAViT_T --data_path <path-to-imagenet> --num_workers 16 --batch_size 128 --drop_path 0.05 --epoch 300 --dist_eval --output_dir ./exp/PPMAViT_T_202505010000

# PPMA-Small
torchrun --nproc_per_node=8 --master_port=29501 main.py --warmup_epochs 5 --model PPMAViT_S --data_path <path-to-imagenet> --num_workers 16 --batch_size 128 --drop_path 0.05 --epoch 300 --dist_eval --output_dir ./exp/PPMAViT_S_202505010000

# PPMA-Base
torchrun --nproc_per_node=8 --master_port=29501 main.py --warmup_epochs 5 --model PPMAViT_B --data_path <path-to-imagenet> --num_workers 16 --batch_size 128 --drop_path 0.05 --epoch 300 --dist_eval --output_dir ./exp/PPMAViT_B_202505010000
```



#### 2. Training on COCO2017

```
cd detection/

# 1x schedule
bash dist_train.sh ./configs/mask_rcnn_ppma_tiny_fpn_1x_coco.py 8
bash dist_train.sh ./configs/mask_rcnn_ppma_small_fpn_1x_coco.py 8
bash dist_train.sh ./configs/mask_rcnn_ppma_base_fpn_1x_coco.py 8

# 3x schedule
bash dist_train.sh ./configs/mask_rcnn_ppma_tiny_fpn_3x_coco.py 8
bash dist_train.sh ./configs/mask_rcnn_ppma_small_fpn_3x_coco.py 8
bash dist_train.sh ./configs/mask_rcnn_ppma_base_fpn_3x_coco.py 8
```



##### Evaluation on COCO2017

```
cd detection/

# 1x schedule
bash dist_test.sh ./configs/mask_rcnn_ppma_tiny_fpn_1x_coco.py <checkpoint-path> 8 --eval bbox segm
bash dist_test.sh ./configs/mask_rcnn_ppma_small_fpn_1x_coco.py <checkpoint-path> 8 --eval bbox segm
bash dist_test.sh ./configs/mask_rcnn_ppma_base_fpn_1x_coco.py <checkpoint-path> 8 --eval bbox segm

# 3x schedule
bash dist_test.sh ./configs/mask_rcnn_ppma_tiny_fpn_3x_coco.py <checkpoint-path> 8 --eval bbox segm
bash dist_test.sh ./configs/mask_rcnn_ppma_small_fpn_3x_coco.py <checkpoint-path> 8 --eval bbox segm
bash dist_test.sh ./configs/mask_rcnn_ppma_base_fpn_3x_coco.py <checkpoint-path> 8 --eval bbox segm
```



#### 3. Training on ADE20K

```
cd segmentation/

bash dist_train.sh ./configs/upernet_ppma_tiny_512x512_160k_ade20k_ss.py 8
bash dist_train.sh ./configs/upernet_ppma_small_512x512_160k_ade20k_ss.py 8
bash dist_train.sh ./configs/upernet_ppma_base_512x512_160k_ade20k_ss.py 8
```



#####  Evaluation on ADE20K

```
cd segmentation/

# Single-scale Evaluation
bash dist_test.sh ./configs/upernet_ppma_tiny_512x512_160k_ade20k_ss.py <checkpoint-path> 8 --eval mIoU
bash dist_test.sh ./configs/upernet_ppma_small_512x512_160k_ade20k_ss.py <checkpoint-path> 8 --eval mIoU
bash dist_test.sh ./configs/upernet_ppma_base_512x512_160k_ade20k_ss.py <checkpoint-path> 8 --eval mIoU

# Multi-scale Evaluation
bash dist_test.sh ./configs/upernet_ppma_tiny_512x512_160k_ade20k_ss.py <checkpoint-path> 8 --eval mIoU --aug-test
bash dist_test.sh ./configs/upernet_ppma_small_512x512_160k_ade20k_ss.py <checkpoint-path> 8 --eval mIoU --aug-test
bash dist_test.sh ./configs/upernet_ppma_base_512x512_160k_ade20k_ss.py <checkpoint-path> 8 --eval mIoU --aug-test
```



#### 4. Training on Cityscapes

```
cd segmentation/
bash dist_train.sh ./configs/cityscapes/upernet_ppma_tiny_1024x1024_160k_cityscapes_ss.py 8
bash dist_train.sh ./configs/cityscapes/upernet_ppma_small_1024x1024_160k_cityscapes_ss.py 8
bash dist_train.sh ./configs/cityscapes/upernet_ppma_base_1024x1024_160k_cityscapes_ss.py 8
```



#####  Evaluation on Cityscapes

```
cd segmentation/

# Single-scale Evaluation
bash dist_test.sh ./configs/cityscapes/upernet_ppma_tiny_1024x1024_160k_cityscapes_ss.py <checkpoint-path> 8 --eval mIoU
bash dist_test.sh ./configs/cityscapes/upernet_ppma_small_1024x1024_160k_cityscapes_ss.py <checkpoint-path> 8 --eval mIoU
bash dist_test.sh ./configs/cityscapes/upernet_ppma_base_1024x1024_160k_cityscapes_ss.py <checkpoint-path> 8 --eval mIoU

# Multi-scale Evaluation
bash dist_test.sh ./configs/cityscapes/upernet_ppma_base_1024x1024_160k_cityscapes_ss.py <checkpoint-path> 8 --eval mIoU --aug-test
bash dist_test.sh ./configs/cityscapes/upernet_ppma_base_1024x1024_160k_cityscapes_ss.py <checkpoint-path> 8 --eval mIoU --aug-test
bash dist_test.sh ./configs/cityscapes/upernet_ppma_base_1024x1024_160k_cityscapes_ss.py <checkpoint-path> 8 --eval mIoU --aug-test
```





## Acknowledgement

* *This project is built using [timm](https://github.com/huggingface/pytorch-image-models), [MMDetection](https://github.com/open-mmlab/mmdetection), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) libraries, and [DeiT](https://github.com/facebookresearch/deit), [RMT](https://github.com/qhfan/RMT), [TransNeXt](https://github.com/DaiShiResearch/TransNeXt/tree/main) repositories. We express our heartfelt gratitude for the contributions of these open-source projects.*
* *We also want to express our gratitude for some articles introducing this project and derivative implementations based on this project.*





## License

This project is released under the Apache 2.0 license. Please see the [LICENSE](/LICENSE) file for more information.





## Citation

If you use PPMA in your research, please consider the following BibTeX entry and giving us a star:
```BibTeX

```
