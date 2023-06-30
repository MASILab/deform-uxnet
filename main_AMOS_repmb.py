#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 11:06:19 2021

@author: leeh43
"""

from monai.utils import set_determinism, first
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    SpatialPadd,
    ResizeWithPadOrCropd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    KeepLargestConnectedComponentd,
    Spacingd,
    ToTensord,
    ToTensor,
    RandAffined,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    RandRotate90d
)
from UXNet_3D_reparamv2 import UXNET
from monai.networks.layers import Norm
from monai.metrics import DiceMetric, compute_meandice
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
# from sliding_window_test import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import numpy as np
from tqdm import tqdm
import json
os.environ["CUDA_VISIBLE_DEVICES"]='0'

if_pretrain = False
out_classes = 16
task_json = os.path.join('/nfs/masi/leeh43/transformer_convnet/task1_dataset.json')
with open(task_json, 'r') as f:
    data = json.load(f)

train_images = []
train_labels = []
for item in data["training"]:
    train_images.append(os.path.join(item['image']))
    train_labels.append(os.path.join(item['label']))


all_images = sorted(train_images)
all_labels = sorted(train_labels)


data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(all_images, all_labels)]

train_files, val_files = data_dicts[:180], data_dicts[180:]
set_determinism(seed=0)

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(
            1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        # ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(256,256,128), mode=("constant")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-125, a_max=275,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=2,
            image_key="image",
            image_threshold=0,
        ),
        # user can also add other random transforms
        # RandFlipd(
        #     keys=["image", "label"],
        #     spatial_axis=[0],
        #     prob=0.10,
        # ),
        # RandFlipd(
        #     keys=["image", "label"],
        #     spatial_axis=[1],
        #     prob=0.10,
        # ),
        # RandFlipd(
        #     keys=["image", "label"],
        #     spatial_axis=[2],
        #     prob=0.10,
        # ),
        # RandRotate90d(
        #     keys=["image", "label"],
        #     prob=0.10,
        #     max_k=3,
        # ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
        RandAffined(
            keys=['image', 'label'],
            mode=('bilinear', 'nearest'),
            prob=1.0, spatial_size=(96, 96, 96),
            rotate_range=(0, 0, np.pi/30),
            scale_range=(0.1, 0.1, 0.1)),
        ToTensord(keys=["image", "label"]),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(
            1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        # ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(256,256,128), mode=("constant")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-125, a_max=275,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ]
)

## Check Dataset
# check_ds = Dataset(data=train_files, transform=train_transforms)
# check_loader = DataLoader(check_ds, batch_size=4)
# check_data = first(check_loader)
# image, label = (check_data["image"][2][0], check_data["label"][2][0])
# print(f"image shape: {image.shape}, label shape: {label.shape}")
# plot the slice [:, :, 80]
# fig = plt.figure("check", (12, 6))
# plt.subplot(1, 2, 1)
# plt.title("image")
# plt.imshow(image[:, :, 110], cmap="gray")
# plt.subplot(1, 2, 2)
# plt.title("label")
# ax = plt.imshow(label[:, :, 110])
# fig.colorbar(ax)
# plt.show()

train_ds = CacheDataset(
    data=train_files, transform=train_transforms,
    cache_rate=0, num_workers=1)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=6, pin_memory=True)

val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_rate=0, num_workers=1)
# val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

device = torch.device("cuda:0")

if if_pretrain == True:
    from monai.networks.blocks import UnetOutBlock
    model = UXNET(
        in_chans=1,
        out_chans=5,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        spatial_dims=3,
    ).to(device)
    model_dir = os.path.join('/nfs/masi/leeh43/monai_seg/FLARE_repcovxnet_MIDL_lr_0.0001_AdamW_all_bz_1_sample_2_21x21x21_conv_matrix_c2')
    model_pth = os.path.join(model_dir, 'best_metric_model.pth')
    model.load_state_dict(torch.load(model_pth))
    model.out = UnetOutBlock(spatial_dims=3, in_channels=48, out_channels=out_classes)
    model = model.to(device)
    print('----------Have Pretrained Weights! Start Transfer Learning for {} classes----------'.format(out_classes))
else:
    model = UXNET(
        in_chans=1,
        out_chans=out_classes,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        spatial_dims=3,
    )
    # model_dir = os.path.join('/nfs/masi/leeh43/monai_seg/AMOS_deformuxnet_MIDL_lr_0.0001_AdamW_k=3_offsetk=5_depthwise_THW')
    # model_pth = os.path.join(model_dir, 'best_metric_model.pth')
    # model.load_state_dict(torch.load(model_pth))
    model = model.to(device)
    print('----------No Pretained Weights! Start Training From Scratch----------')


from ptflops import get_model_complexity_info
with torch.cuda.device(0):
    macs, params = get_model_complexity_info(model, (1, 96, 96, 96), as_strings=True,
            print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


# model_pth = os.path.join('/nfs/masi/leeh43/monai_seg/FLARE_repcovxnet_MIDL_lr_0.0001_bz_1_sample_2_3x3x3_21x21x21_SGD_c2/best_metric_model.pth')
# weights = torch.load(model_pth)
# for item in weights:
#     print(item)
# model.load_from(torch.load(model_pth))
# model = torch.nn.DataParallel(model).cuda()

## Set multi-branch learning rate
layer_names = []
for idx, (name, param) in enumerate(model.named_parameters()):
    layer_names.append(name)
    # print(f'{idx}:{name}')

parameters = []
lr = 0.0001
# ratio = 7 / 21
for idx, name in enumerate(layer_names):
    if 'branch' in name:
        # print(name)
        parameters += [{'params': [p for n, p in model.named_parameters() if n == name and p.requires_grad],
                    'lr':     lr}]
    else:
        parameters += [{'params': [p for n, p in model.named_parameters() if n == name and p.requires_grad],
                    'lr':     lr}]

# loss_function = DiceLoss(to_onehot_y=True, softmax=True)
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
# dice_metric = DiceMetric(include_background=False, reduction='mean')
optimizer = torch.optim.AdamW(parameters)
# optimizer = torch.optim.SGD(parameters, momentum=0.99, weight_decay=3e-5, nesterov=True)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=1000)

# max_epochs = 600
# val_interval = 1
# best_metric = -1
# best_metric_epoch = -1
# epoch_loss_values = []
# metric_values = []
# post_pred = Compose([ToTensor(), AsDiscrete(argmax=True, to_onehot=True, n_classes=num_classes )])
# post_label = Compose([ToTensor(), AsDiscrete(to_onehot=True, n_classes=num_classes )])

root_dir = os.path.join('/nfs/masi/leeh43/monai_seg/AMOS_deformuxnet_MIDL_lr_0.0001_AdamW_k=3_offsetk=5_depthwise_THW')
if os.path.exists(root_dir) == False:
    os.makedirs(root_dir)
    
t_dir = os.path.join(root_dir, 'tensorboard')
if os.path.exists(t_dir) == False:
    os.makedirs(t_dir)
writer = SummaryWriter(log_dir=t_dir)

def validation(epoch_iterator_val):
    # model_feat.eval()
    model.eval()
    dice_vals = list()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            # val_outputs = model(val_inputs)
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 2, model)
            # val_outputs = model_seg(val_inputs, val_feat[0], val_feat[1])
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            dice = dice_metric.aggregate().item()
            dice_vals.append(dice)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice)
            )
        dice_metric.reset()
    mean_dice_val = np.mean(dice_vals)
    writer.add_scalar('Validation Segmentation Loss', mean_dice_val, global_step)
    return mean_dice_val


def train(global_step, train_loader, dice_val_best, global_step_best):
    # model_feat.eval()
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        # with torch.no_grad():
        #     g_feat, dense_feat = model_feat(x)
        # logit_map, kd_loss = model(x)
        logit_map = model(x)
        # avg_kd = torch.stack(kd_loss, dim=0).sum(dim=0) / len(kd_loss)
        # print('Knowledge Distillation Loss = {}'.format(avg_kd * 0.01))
        loss = loss_function(logit_map, y) 
        # + avg_kd * 0.01
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
        )
        if (
            global_step % eval_num == 0 and global_step != 0
        ) or global_step == max_iterations:
            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            dice_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(
                    model.state_dict(), os.path.join(root_dir, "best_metric_model.pth")
                )
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
                # scheduler.step(dice_val)
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
                # scheduler.step(dice_val)
        writer.add_scalar('Training Segmentation Loss', loss.data, global_step)
        global_step += 1
    return global_step, dice_val_best, global_step_best


max_iterations = 40000
eval_num = 500
post_label = AsDiscrete(to_onehot=out_classes)
post_pred = AsDiscrete(argmax=True, to_onehot=out_classes)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(
        global_step, train_loader, dice_val_best, global_step_best
    )





