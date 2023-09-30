# Training

We provide FeTA2021 training, FLARE2021 training, and AMOS2022 Finetuning commands here.
Please check [INSTALL.md](INSTALL.md) for installation instructions first.

## FLARE 2021 & AMOS 2022 Train From Scratch Scenario
3D RepUX-Net training on FLARE/AMOS with a single GPU:
```
python main_train.py --root root_folder_path --output output_folder_path \
--dataset flare/amos --network RepUXNET --mode train --pretrain False \
--batch_size 1 --crop_sample 2 --lr 0.0001 --optim AdamW --max_iter 60000 \ 
--eval_step 500 --gpu 0 --cache_rate 0.2 --num_workers 2
```

- If the error "Out of GPU memory" is popped out, please reduce the number of crop_sample or cache_rate 
- We perform 60000 iterations for training, and validation is performed in every 500 step.
- If you want to run our code with your dataset, please look into [load_datasets_transforms.py](load_datasets_transforms.py) and you can directly create new transforms following the similar format for your own dataset. 

## AMOS 2022 Finetuning
3D RepUX-Net finetuning on AMOS 2022 with a single GPU:
```
python main_finetune.py --root root_folder_path --output output_folder_path \
--dataset amos --network REPUXNET --mode train --pretrain True \
--pretrained_weights path_to_pretrained_weights --pretrain_classes 5 (FLARE number of classes) \
--batch_size 1 --crop_sample 2 --lr 0.0001 --optim AdamW --max_iter 60000 \ 
--eval_step 500 --gpu 0 --cache_rate 0.2 --num_workers 2
```

- The main_finetune.py allows us to finetune on models that pretrained with different datasets, as long as we know the output classes and the network structure of the pretrained model.
- For the finetuning scenario in the paper, we use the best fold model in 5-fold cross-validations with FLARE 2021 as the pretrained weights for finetuning.

## Network Choices
We also aim to perform benchmarking in the latest public multi-organ/tissue segmentation datasets with volumetric transformer network SOTA. This GitHub allows to perform fair comparisons for network by using same data augmentation and preprocessing. Here is the summary of the available network in our code:
- 3D U-Net
- SegResNet
- TransBTS
- UNETR
- nnFormer
- SwinUNETR
- 3D UX-Net

Feel free to provide recommendations of adding latest volumetric transformer or CNN networks and we can further implement it for benchmarking.

