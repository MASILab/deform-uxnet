### [Scaling Up 3D Kernels with Bayesian Frequency Re-parameterization for Medical Image Segmentation](https://arxiv.org/abs/2303.05785)

Official Pytorch implementation of 3D RepUX-Net, from the following paper:

[Scaling Up 3D Kernels with Bayesian Frequency Re-parameterization for Medical Image Segmentation](https://arxiv.org/abs/2303.05785). MICCAI 2023 (Provisional Accepted, top 14%) \
Ho Hin Lee, Quan Liu, Shunxing Bao, Qi Yang, Xin Yu, Leon Y. Cai, Thomas Li, [Yuankai Huo](https://hrlblab.github.io/), [Xenofon Koutsoukos](https://engineering.vanderbilt.edu/bio/xenofon-koutsoukos), [Bennet A. Landman](https://my.vanderbilt.edu/masi/people/bennett-landman-ph-d/) \
Vanderbilt University \
[[`arXiv`](https://arxiv.org/abs/2303.05785)]

---

<p align="center">
<img src="screenshots/Figure_1.png" width=100% height=40% 
class="center">
</p>

<p align="center">
<img src="screenshots/Figure_2.png" width=100% height=40% 
class="center">
</p>


We propose **3D RepUX-Net**, a pure volumetric convolutional network that effectively adapts current largest 3D kernel sizes (e.g., 21x21x21) with spatial frequency modeling as Bayesian prior for weight re-parameterization during training.

## Installation
 Please look into the [INSTALL.md](INSTALL.md) for creating conda environment and package installation procedures.

 ## Training Tutorial
 - [x] FLARE 2021 Training Code [TRAINING.md](TRAINING.md)
 - [x] AMOS 2022 Finetuning Code [TRAINING.md](TRAINING.md)
 
<!-- ✅ ⬜️  -->

## Results
### FLARE 2021 Train From Scratch Models (5-folds cross-validation)
| Methods | resolution | #params | FLOPs | Mean Dice | Model
|:---:|:---:|:---:|:---:| :---:|:---:|
| nn-UNet | 96x96x96 | 31.2M | 743.3G | 0.926 | |
| TransBTS | 96x96x96 | 31.6M | 110.4G | 0.902 | | 
| UNETR | 96x96x96 | 92.8M | 82.6G | 0.886 | |
| nnFormer | 96x96x96 | 149.3M | 240.2G | 0.906 | |
| SwinUNETR | 96x96x96 | 62.2M | 328.4G | 0.929 | |
| 3D UX-Net (k=7) | 96x96x96 | 53.0M | 639.4G | 0.934 | |
| 3D UX-Net (k=21) | 96x96x96 | 65.9M | 757.6G | 0.930 | |
| 3D RepUX-Net | 96x96x96 | 65.8M | 757.4G | 0.944 | |



 ### AMOS 2022 Models (T.F.S: Train From Scratch, F.T: Fine-Tuning)
 | Methods | resolution | #params | FLOPs | Mean Dice (T.F.S) | Mean Dice (F.T)
|:---:|:---:|:---:|:---:| :---:| :---:|
| nn-UNet | 96x96x96 | 31.2M | 743.3G | 0.850 | 0.878 |
| TransBTS | 96x96x96 | 31.6M | 110.4G | 0.783 | 0.792 |
| UNETR | 96x96x96 | 92.8M | 82.6G | 0.740 | 0.762 |
| nnFormer | 96x96x96 | 149.3M | 240.2G | 0.785 | 0.790|
| SwinUNETR | 96x96x96 | 62.2M | 328.4G | 0.871 |  0.880|
| 3D UX-Net (k=7) | 96x96x96 | 53.0M | 639.4G | 0.890 | 0.900|
| 3D UX-Net (k=21) | 96x96x96 | 65.9M | 757.6G | 0.891 | 0.898|
| 3D RepUX-Net | 96x96x96 | 65.8M | 757.4G | 0.902 | 0.911 |

 ### External Testing of FLARE-trained Model with 4 Different Datasets 
 | Methods | MSD Spleen | KiTS Kidney | LiTS Liver | TCIA Pancreas |
|:---:|:---:|:---:|:---:| :---:|
| nn-UNet | 0.917 | 0.829 | 0.935 | 0.739 | 
| TransBTS | 0.881 | 0.797 | 0.926 | 0.699 |
| UNETR | 0.857 | 0.801 | 0.920 | 0.679 | 
| nnFormer | 0.880 | 0.774 | 0.927 | 0.690 | 
| SwinUNETR | 0.901 | 0.815 | 0.933 | 0.736| 
| 3D UX-Net (k=7) | 0.926 | 0.836 | 0.939 | 0.750 |
| 3D UX-Net (k=21) | 0.908 | 0.808 | 0.929 | 0.720 |
| 3D RepUX-Net | 0.932 | 0.847 | 0.949 | 0.779 |




<!-- ✅ ⬜️  -->
## Training
Training and fine-tuning instructions are in [TRAINING.md](TRAINING.md). Pretrained model weights will be uploaded for public usage later on.

<!-- ✅ ⬜️  -->
## Evaluation
Efficient evaulation can be performed for the above three public datasets as follows:
```
python test_seg.py --root path_to_image_folder --output path_to_output \
--dataset flare --network REPUXNET --trained_weights path_to_trained_weights \
--mode test --sw_batch_size 4 --overlap 0.7 --gpu 0 --cache_rate 0.2 \
```

## Acknowledgement
This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library.

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Citation
If you find this repository helpful, please consider citing:
```
@Article{lee2023scaling,
  author  = {Lee, Ho Hin and Liu, Quan and Bao, Shunxing and Yang, Qi and Yu, Xin and Cai, Leon Y and Li, Thomas and Huo, Yuankai and Koutsoukos, Xenofon and Landman, Bennett A},
  title   = {Scaling Up 3D Kernels with Bayesian Frequency Re-parameterization for Medical Image Segmentation},
  journal = {arXiv preprint arXiv:2303.05785},
  year    = {2023}
}
```