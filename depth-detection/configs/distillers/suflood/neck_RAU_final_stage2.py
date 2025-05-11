dataset_type = 'FLOODDataset'
data_root = '/public/DATA/lxy/'
classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
           'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
           'scissors', 'teddy bear', 'hair drier', 'toothbrush')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='TriLoadImageFromFile',use_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='Collect', keys=['img', 't_img', 'gt_labels','mask'])
]
test_pipeline = [
    dict(type='TriLoadImageFromFile',use_mask=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img', 't_img' ]),
            dict(type='Collect', keys=['img', 't_img' ])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='FLOODDataset',
        classes=('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                 'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
        ann_file=
        './suflood/final_v_train.json',
        img_prefix='',
        pipeline=[
            dict(type='TriLoadImageFromFile',use_mask=True),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=False),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='Collect', keys=['img', 't_img', 'gt_labels','mask'])
        ]),
    val=dict(
        type='FLOODDataset',
        classes=('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                 'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
        ann_file=
        './suflood/final_v_test.json',
        img_prefix='',
        pipeline=[
            dict(type='TriLoadImageFromFile',use_mask=True),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img', 't_img' ]),
                    dict(type='Collect', keys=['img', 't_img', 'gt_labels'],)
                ])
        ]),
    test=dict(
        type='FLOODDataset',
        classes=('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                 'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
        ann_file=
        './suflood/final_v_test.json',
        img_prefix='',
        pipeline=[
            dict(type='TriLoadImageFromFile',use_mask=True),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=False),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img', 't_img' ]),
                    dict(type='Collect', keys=['img', 't_img', 'gt_labels' ])
                ])
        ]))
evaluation = dict(interval=1, metric='bbox')
optimizer = dict(type='AdamW', lr=1.4e-5, weight_decay=1e-3,capturable=True)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    gamma=0.1,
    warmup_iters=300,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=60)
checkpoint_config = dict(interval=4)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
temp = 0.5
alpha_fgd = 0.001
beta_fgd = 0.0005
gamma_fgd = 0.0005
lambda_fgd = 5e-06
distiller = dict(
    type='DetectionDistiller',
    teacher_pretrained=
    'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_1x_coco/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth',
    init_student=True,
    bbox=False,
    line_loss=False,
    distill_cfg=[
        dict(
            student_module='neck.fpn_convs.4.conv',
            teacher_module='neck.fpn_convs.4.conv',
            output_hook=True,
            methods=[
                dict(
                    type='FeatureLoss',
                    name='loss_fgd_fpn_4',
                    student_channels=256,
                    teacher_channels=256,
                    temp=0.5,
                    alpha_fgd=0.001,
                    beta_fgd=0.0005,
                    gamma_fgd=0.0005,
                    lambda_fgd=5e-06)
            ]),
        dict(
            student_module='neck.fpn_convs.3.conv',
            teacher_module='neck.fpn_convs.3.conv',
            output_hook=True,
            methods=[
                dict(
                    type='FeatureLoss',
                    name='loss_fgd_fpn_3',
                    student_channels=256,
                    teacher_channels=256,
                    temp=0.5,
                    alpha_fgd=0.001,
                    beta_fgd=0.0005,
                    gamma_fgd=0.0005,
                    lambda_fgd=5e-06)
            ]),
        dict(
            student_module='neck.fpn_convs.2.conv',
            teacher_module='neck.fpn_convs.2.conv',
            output_hook=True,
            methods=[
                dict(
                    type='FeatureLoss',
                    name='loss_fgd_fpn_2',
                    student_channels=256,
                    teacher_channels=256,
                    temp=0.5,
                    alpha_fgd=0.001,
                    beta_fgd=0.0005,
                    gamma_fgd=0.0005,
                    lambda_fgd=5e-06)
            ]),
        dict(
            student_module='neck.fpn_convs.1.conv',
            teacher_module='neck.fpn_convs.1.conv',
            output_hook=True,
            methods=[
                dict(
                    type='FeatureLoss',
                    name='loss_fgd_fpn_1',
                    student_channels=256,
                    teacher_channels=256,
                    temp=0.5,
                    alpha_fgd=0.001,
                    beta_fgd=0.0005,
                    gamma_fgd=0.0005,
                    lambda_fgd=5e-06)
            ]),
        dict(
            student_module='neck.fpn_convs.0.conv',
            teacher_module='neck.fpn_convs.0.conv',
            output_hook=True,
            methods=[
                dict(
                    type='FeatureLoss',
                    name='loss_fgd_fpn_0',
                    student_channels=256,
                    teacher_channels=256,
                    temp=0.5,
                    alpha_fgd=0.001,
                    beta_fgd=0.0005,
                    gamma_fgd=0.0005,
                    lambda_fgd=5e-06)
            ])])
student_cfg = 'configs/distillers/retinanet_r50_RAU_head.py'
teacher_cfg = 'configs/retinanet/retinanet_x101_64x4d_fpn_1x_coco.py'
work_dir = './work_dirs/stage2'
gpu_ids = range(0, 1)
