#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : coco-stuff164k.py
@Desc  :
from https://github.com/yunxiangfu2001/SegMAN/tree/main/segmentation/local_configs/_base_/datasets/stuff164k.py
"""

# dataset settings
dataset_type = 'COCOStuffDataset'
# data_root = 'data/coco_stuff164k'
data_root = '/mnt/nvme_data/zzc/datasets/cocostuff'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='AlignedResize', keep_ratio=True, size_divisor=32),         # 20250604
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=50,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='images/train2017',
            ann_dir='annotations/train2017',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val2017',
        ann_dir='annotations/val2017',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val2017',
        ann_dir='annotations/val2017',
        pipeline=test_pipeline))
