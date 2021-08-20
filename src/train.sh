#!/usr/bin/env bash
python train.py \
  --saved_fn 'test_multiscale' \
  --arch 'darknet' \
  --cfgfile ./config/cfg/complex_yolov4.cfg \
  --batch_size 1 \
  --lr 2e-4 \
  --num_workers 8 \
  --no-val \
  --gpu_idx 0 \
  --num_epochs 60 \
  --checkpoint_freq 20 \
  --multiscale_training \
  --img_size 320 \
  #--pretrained_path  ../checkpoints/complex_yolov4_no_aug/Model_complex_yolov4_no_aug_epoch_40.pth \