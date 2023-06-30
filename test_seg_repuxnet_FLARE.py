#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 14:54:09 2021

@author: leeh43
"""

from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscreted,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
    Activationsd,
    EnsureChannelFirstd,
    Invertd,
    Resized,
    SaveImaged,
    ScaleIntensityd,
    EnsureTyped,
    KeepLargestConnectedComponentd,
)
from monai.transforms.utils import allow_missing_keys_mode
# from monai.networks.nets import UNet, UNETR, SwinUNETR
from UXNet_3D_reparamv2 import UXNET
# from Unet3D import UNet3D
from monai.networks.layers import Norm
from monai.metrics import compute_meandice
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]='1'

def imshows(ims):
    nrow = len(ims)
    ncol = len(ims[0])
    fig, axes = plt.subplots(nrow, ncol, figsize=(
        ncol * 3, nrow * 3), facecolor='white')
    for i, im_dict in enumerate(ims):
        for j, (title, im) in enumerate(im_dict.items()):
            if isinstance(im, torch.Tensor):
                im = im.detach().cpu().numpy()
            im = np.mean(im, axis=0)  # average across channels
            if len(ims) == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]
            ax.set_title(f"{title}\n{im.shape}")
            im_show = ax.imshow(im)
            ax.axis("off")
            fig.colorbar(im_show, ax=ax)

# img_path = os.path.join('/nfs/masi/leeh43/RandPatch_Peter/test/INPUTS_imageva_nc/cropped_test/images')
# img_path = os.path.join('/nfs/masi/leeh43/style_transfer_pv_nc/registration_pv_to_nc/registered_PV_nii')
# img_path = os.path.join('/nfs/masi/leeh43/RandPatch_Peter/preprocessing/INPUTS/cropped_-4_5/images_test')
# img_path = os.path.join('/nfs/masi/leeh43/RandPatch_Peter/preprocessing/INPUTS_OKD/cropped/images')
# img_path = os.path.join('/nfs/masi/leeh43/RandPatch_Peter/test/INPUTS/cropped/images')
# label_path = os.path.join('/nfs/masi/leeh43/RandPatch_Peter/preprocessing/INPUTS/cropped_-4_5/labels_12')

# out_classes = 8
# mial_img_dir = os.path.join('/nfs/masi/leeh43/feta_2021/images_mial')
# irtk_img_dir = os.path.join('/nfs/masi/leeh43/feta_2021/images_irtk')

# mial_valid_img = sorted(glob.glob(os.path.join(mial_img_dir, '*.nii.gz')))[20:30]
# irtk_valid_img = sorted(glob.glob(os.path.join(irtk_img_dir, '*.nii.gz')))[20:30]

# print(sorted(valid_images))
# all_images = sorted(train_images) + sorted(valid_images)
# # print(len(train_images), len(valid_images))
# all_labels = sorted(train_labels) + sorted(valid_labels)
# print(len(train_labels), len(valid_labels))


out_classes = 5
img_path = os.path.join('/nfs/masi/leeh43/FLARE2021/TRAIN_IMG')
label_path = os.path.join('/nfs/masi/leeh43/FLARE2021/TRAIN_MASK')
output_dir = os.path.join('/nfs/masi/leeh43/transformer_convnet/Datasets/FLARE_split/ablation/BF_alpha/alpha_6')

label_list = []
img_list = []
for img in os.listdir(img_path):
    label_name = img.split('_0000.')[0] + '.nii.gz'
    label_file = os.path.join(label_path, label_name)
    img_file = os.path.join(img_path, img)

    img_list.append(img_file)

test_images = img_list[341:]
# test_labels = label_list[241:301]


all_images = test_images
# test_labels = sorted(glob.glob(os.path.join(label_path, "*.nii.gz")))

# all_images = test_images 
# # all_labels = test_labels

data_dicts = [
    {"image": image_name}
    for image_name in zip(all_images)
]

test_files = data_dicts

set_determinism(seed=0)

## Define transformation
test_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(
            1.0, 1.0, 1.2), mode=("bilinear")),
        # ResizeWithPadOrCropd(keys=["image"], spatial_size=(168,168,128), mode=("constant")),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-125, a_max=275,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        ToTensord(keys=["image"]),
    ]
)

## Load Data & Apply Transform
test_ds = CacheDataset(data=test_files, transform=test_transforms,
                       cache_rate=1.0, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

## Define post transforms
post_transforms = Compose([
        EnsureTyped(keys="pred"), 
        Activationsd(keys="pred", softmax=True),
        Invertd(
            keys="pred",  # invert the `pred` data field, also support multiple fields
            transform=test_transforms,
            orig_keys="image",  # get the previously applied pre_transforms information on the `img` data field,
                              # then invert `pred` based on this information. we can use same info
                              # for multiple fields, also support different orig_keys for different fields
            meta_keys="pred_meta_dict",  # key field to save inverted meta data, every item maps to `keys`
            orig_meta_keys="image_meta_dict",  # get the meta data from `img_meta_dict` field when inverting,
                                             # for example, may need the `affine` to invert `Spacingd` transform,
                                             # multiple fields can use the same meta data to invert
            meta_key_postfix="meta_dict",  # if `meta_keys=None`, use "{keys}_{meta_key_postfix}" as the meta key,
                                           # if `orig_meta_keys=None`, use "{orig_keys}_{meta_key_postfix}",
                                           # otherwise, no need this arg during inverting
            nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
                                   # to ensure a smooth output, then execute `AsDiscreted` transform
            to_tensor=True,  # convert to PyTorch Tensor after inverting
        ),
        AsDiscreted(keys="pred", argmax=True, n_classes=5),
        KeepLargestConnectedComponentd(keys='pred', applied_labels=[1,3]),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=output_dir, output_postfix="seg", output_ext=".nii.gz", resample=True),
])

## Define model, loss & optimizer
device = torch.device('cuda:0')
model = UXNET(
        in_chans=1,
        out_chans=out_classes,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        spatial_dims=3
).to(device)

model_dir = os.path.join('/nfs/masi/leeh43/monai_seg/MICCAI2023/Ablation/FLARE_repcovxnet_MIDL_lr_0.0001_AdamW_all_bz_1_sample_2_21x21x21_conv_matrix_alpha_6')
model_path = os.path.join(model_dir, 'best_metric_model.pth')

model.load_state_dict(torch.load(model_path))
model.eval()
with torch.no_grad():
    for i, test_data in enumerate(test_loader):
        images = test_data["image"].to(device)
        roi_size=(96, 96, 96)
        sw_batch_size = 4
        test_data['pred'] = sliding_window_inference(
            images, roi_size, sw_batch_size, model)
        test_data = [post_transforms(i) for i in decollate_batch(test_data)]
        
        
        
        
        
































