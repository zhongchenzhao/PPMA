_base_ = [
    '../_base_/models/upernet_base.py',
    '../_base_/datasets/cityscapes_1024x1024_repeat.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

crop_size = (1024, 1024)        # cityscapes
# optimizer
model = dict(
    backbone=dict(
        pretrained="../checkpoints/classification/PPMA_T_202502251000/best.pth",
        type='ppma_tiny',
        num_classes=150,
        embed_dims=[64, 128, 256, 512],  # tiny
    ),
    decode_head=dict(
        in_channels=[64, 128, 256, 512],
        num_classes=150
    ),
    auxiliary_head=dict(
        in_channels=256,
        num_classes=150
    ),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(768, 768)),
)

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'query_embedding': dict(decay_mult=0.),
                                                 'relative_pos_bias_local': dict(decay_mult=0.),
                                                 'cpb': dict(decay_mult=0.),
                                                 'temperature': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=1)            # total batch size 8
# data=dict(samples_per_gpu=4)            # models are trained on 4 GPUs with 4 images per GPU
evaluation = dict(interval=4000, metric='mIoU')     # cityscapes
