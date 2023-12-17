# --------------------------------------------------------
# Based on timm, mmseg, setr, xcit and swin code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/fudan-zvg/SETR
# https://github.com/facebookresearch/xcit/
# https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------'
# recommand use this config for BEiT models which are self-supervised pretrained on imagenet
_base_ = [
    '../_base_/models/upernet_beit.py', '../_base_/datasets/endovissub2017.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
crop_size = (512, 512)
num_classes_cholecseg8k = 8 #include _background_
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters=True
model = dict(
    type='EncoderDecoder',
    pretrained='./pretrain/pretrained_pvt.pth',
    backbone=dict(
        _delete_=True,
        type='PVT', # PVT Small
        img_size=512,
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        depths=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        sr_ratios=[8, 4, 2, 1],
        #fpn_out_dim=[64, 128, 320, 512],
        drop_path_rate=0.1,
		fpn_out_dim=768,
		is_fpn_out_layer=False,
        out_indices=[2, 6, 12, 15]
    ),
    decode_head=dict(
        in_channels=[512, 512, 512, 512],
        #in_channels=[64, 128, 320, 512],
        num_classes=num_classes_cholecseg8k,
        channels=512,
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=50000)
    ),
    auxiliary_head=dict(
        in_channels=512,
        num_classes=num_classes_cholecseg8k,
    ), 
    test_cfg = dict(mode='slide', crop_size=crop_size, stride=(341, 341))
)


optimizer = dict(_delete_=True, type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05,
                 constructor='LayerDecayOptimizerConstructor', 
                 paramwise_cfg=dict(num_layers=16, layer_decay_rate=1.))



lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=8)
"""custom_hooks = [
    dict(type='EMAHook', start_iters=1000, priority='NORMAL')
]"""

#runner = dict(type='IterBasedRunnerAmp')

# do not use mmdet version fp16
"""fp16 = True
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=False,
)"""
evaluation = dict(interval=1000, metric='mIoU')