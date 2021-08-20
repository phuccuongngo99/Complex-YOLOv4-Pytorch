#!/usr/bin/env bash
python test.py \
  --saved_fn 'complex_yolov4_vis' \
  --arch 'darknet' \
  --cfgfile ./config/cfg/complex_yolov4.cfg \
  --batch_size 1 \
  --num_workers 1 \
  --gpu_idx 0 \
  --pretrained_path ../checkpoints/complex_yolov4_vis/Model_complex_yolov4_vis_epoch_300.pth \
  --img_size 320 \
  --conf_thresh 0.1 \
  --nms_thresh 0.2 \
  --show_image \
#  --save_test_output \
  --output_format 'image' \
