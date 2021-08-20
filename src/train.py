import time
import numpy as np
import sys
import random
import os
import warnings
import cv2
import copy
from easydict import EasyDict as edict

warnings.filterwarnings("ignore", category=UserWarning)

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from tqdm import tqdm

sys.path.append('./')

import config.kitti_config as cnf
from data_process import kitti_data_utils, kitti_bev_utils
from data_process.kitti_dataloader import create_train_dataloader, create_val_dataloader
from models.model_utils import create_model, make_data_parallel, get_num_parameters
from utils.evaluation_utils import post_processing_v2, rescale_boxes
from utils.train_utils import create_optimizer, create_lr_scheduler, get_saved_state, save_checkpoint
from utils.train_utils import reduce_tensor, to_python_float, get_tensorboard_log
from utils.misc import AverageMeter, ProgressMeter
from utils.logger import Logger
from config.train_config import parse_train_configs
from utils.visualization_utils import show_image_with_boxes, merge_rgb_to_bev, predictions_to_kitti_format
from evaluate import evaluate_mAP


def main():
    configs = parse_train_configs()

    # Re-produce results
    if configs.seed is not None:
        random.seed(configs.seed)
        np.random.seed(configs.seed)
        torch.manual_seed(configs.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if configs.gpu_idx is not None:
        print('You have chosen a specific GPU. This will completely disable data parallelism.')

    if configs.dist_url == "env://" and configs.world_size == -1:
        configs.world_size = int(os.environ["WORLD_SIZE"])

    configs.distributed = configs.world_size > 1 or configs.multiprocessing_distributed

    if configs.multiprocessing_distributed:
        configs.world_size = configs.ngpus_per_node * configs.world_size
        mp.spawn(main_worker, nprocs=configs.ngpus_per_node, args=(configs,))
    else:
        main_worker(configs.gpu_idx, configs)


def main_worker(gpu_idx, configs):
    configs.gpu_idx = gpu_idx
    configs.device = torch.device('cpu' if configs.gpu_idx is None else 'cuda:{}'.format(configs.gpu_idx))

    if configs.distributed:
        if configs.dist_url == "env://" and configs.rank == -1:
            configs.rank = int(os.environ["RANK"])
        if configs.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            configs.rank = configs.rank * configs.ngpus_per_node + gpu_idx

        dist.init_process_group(backend=configs.dist_backend, init_method=configs.dist_url,
                                world_size=configs.world_size, rank=configs.rank)
        configs.subdivisions = int(64 / configs.batch_size / configs.ngpus_per_node)
    else:
        configs.subdivisions = int(64 / configs.batch_size)

    configs.is_master_node = (not configs.distributed) or (
            configs.distributed and (configs.rank % configs.ngpus_per_node == 0))

    if configs.is_master_node:
        logger = Logger(configs.logs_dir, configs.saved_fn)
        logger.info('>>> Created a new logger')
        logger.info('>>> configs: {}'.format(configs))
        tb_writer = SummaryWriter(log_dir=os.path.join(configs.logs_dir, 'tensorboard'))
    else:
        logger = None
        tb_writer = None

    # model
    model = create_model(configs)

    # load weight from a checkpoint
    if configs.pretrained_path is not None:
        assert os.path.isfile(configs.pretrained_path), "=> no checkpoint found at '{}'".format(configs.pretrained_path)
        model.load_state_dict(torch.load(configs.pretrained_path))
        if logger is not None:
            logger.info('loaded pretrained model at {}'.format(configs.pretrained_path))

    # resume weights of model from a checkpoint
    if configs.resume_path is not None:
        assert os.path.isfile(configs.resume_path), "=> no checkpoint found at '{}'".format(configs.resume_path)
        model.load_state_dict(torch.load(configs.resume_path))
        if logger is not None:
            logger.info('resume training model from checkpoint {}'.format(configs.resume_path))

    # Data Parallel
    model = make_data_parallel(model, configs)

    # Make sure to create optimizer after moving the model to cuda
    optimizer = create_optimizer(configs, model)
    lr_scheduler = create_lr_scheduler(optimizer, configs)
    configs.step_lr_in_epoch = True if configs.lr_type in ['multi_step'] else False

    # resume optimizer, lr_scheduler from a checkpoint
    if configs.resume_path is not None:
        utils_path = configs.resume_path.replace('Model_', 'Utils_')
        assert os.path.isfile(utils_path), "=> no checkpoint found at '{}'".format(utils_path)
        utils_state_dict = torch.load(utils_path, map_location='cuda:{}'.format(configs.gpu_idx))
        optimizer.load_state_dict(utils_state_dict['optimizer'])
        lr_scheduler.load_state_dict(utils_state_dict['lr_scheduler'])
        configs.start_epoch = utils_state_dict['epoch'] + 1

    if configs.is_master_node:
        num_parameters = get_num_parameters(model)
        logger.info('number of trained parameters of the model: {}'.format(num_parameters))

    if logger is not None:
        logger.info(">>> Loading dataset & getting dataloader...")
    # Create dataloader
    train_dataloader, train_sampler = create_train_dataloader(configs)
    if logger is not None:
        logger.info('number of batches in training set: {}'.format(len(train_dataloader)))

    if configs.evaluate:
        val_dataloader = create_val_dataloader(configs)
        precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs, None)
        print('Evaluate - precision: {}, recall: {}, AP: {}, f1: {}, ap_class: {}'.format(precision, recall, AP, f1,
                                                                                          ap_class))
        print('mAP {}'.format(AP.mean()))
        return

    for epoch in range(configs.start_epoch, configs.num_epochs + 1):
        if logger is not None:
            logger.info('{}'.format('*-' * 40))
            logger.info('{} {}/{} {}'.format('=' * 35, epoch, configs.num_epochs, '=' * 35))
            logger.info('{}'.format('*-' * 40))
            logger.info('>>> Epoch: [{}/{}]'.format(epoch, configs.num_epochs))

        if configs.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, epoch, configs, logger, tb_writer)
        if not configs.no_val and epoch > 20:
            val_dataloader = create_val_dataloader(configs)
            print('number of batches in val_dataloader: {}'.format(len(val_dataloader)))
            precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs, logger)
            val_metrics_dict = {
                'precision': precision.mean(),
                'recall': recall.mean(),
                'AP': AP.mean(),
                'f1': f1.mean(),
                'ap_class': ap_class.mean()
            }
            if tb_writer is not None:
                tb_writer.add_scalars('Validation', val_metrics_dict, epoch)

        # Save checkpoint
        if configs.is_master_node and ((epoch % configs.checkpoint_freq) == 0):
            model_state_dict, utils_state_dict = get_saved_state(model, optimizer, lr_scheduler, epoch, configs)
            save_checkpoint(configs.checkpoints_dir, configs.saved_fn, model_state_dict, utils_state_dict, epoch)
            
            # Add tensorboard visualization here
            # Validation
            if epoch != 0:
                vis_images = inference_val(configs, model, num_img = 12)
            
                if tb_writer is not None:
                    tb_writer.add_images(
                        "Validation",
                        vis_images,
                        epoch,
                        dataformats='NHWC'
                    )

        if not configs.step_lr_in_epoch:
            lr_scheduler.step()
            if tb_writer is not None:
                tb_writer.add_scalar('LR', lr_scheduler.get_lr()[0], epoch)

    if tb_writer is not None:
        tb_writer.close()
    if configs.distributed:
        cleanup()


def cleanup():
    dist.destroy_process_group()

def inference_val(configs, model, num_img):
    model.eval()

    cfg = copy.deepcopy(configs)

    cfg.batch_size = 1
    
    val_dataloader = create_val_dataloader(cfg)

    batch_images = []
    
    for batch_idx, (img_paths, imgs_bev, targets) in enumerate(val_dataloader):
        input_imgs = imgs_bev.to(device=cfg.device).float()
        outputs = model(input_imgs)
        detections = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)

        img_detections = []  # Stores detections for each image index
        img_detections.extend(detections)

        img_bev = imgs_bev.squeeze() * 255
        img_bev = img_bev.permute(1, 2, 0).numpy().astype(np.uint8)
        img_bev = cv2.resize(img_bev, (cfg.img_size, cfg.img_size))
        for detections in img_detections:
            if detections is None:
                continue
            # Rescale boxes to original image
            detections = rescale_boxes(detections, cfg.img_size, img_bev.shape[:2])
            for x, y, w, l, im, re, *_, cls_pred in detections:
                yaw = np.arctan2(im, re)
                # Draw rotated box
                kitti_bev_utils.drawRotatedBox(img_bev, x, y, w, l, yaw, cnf.colors[int(cls_pred)])

        img_rgb = cv2.imread(img_paths[0])
        calib = kitti_data_utils.Calibration(img_paths[0].replace(".png", ".txt").replace("image_2", "calib"))
        objects_pred = predictions_to_kitti_format(img_detections, calib, img_rgb.shape, cfg.img_size)
        img_rgb = show_image_with_boxes(img_rgb, objects_pred, calib, False)

        img_bev = cv2.flip(cv2.flip(img_bev, 0), 1)

        out_img = merge_rgb_to_bev(img_rgb, img_bev, cfg.img_size)

        # Turn it into rgb again cos cv2 is in bgr form
        out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)

        batch_images.append(out_img)

        if len(batch_images) == num_img:
            break
    
    return np.asarray(batch_images)

def train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, epoch, configs, logger, tb_writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(len(train_dataloader), [batch_time, data_time, losses],
                             prefix="Train - Epoch: [{}/{}]".format(epoch, configs.num_epochs))

    num_iters_per_epoch = len(train_dataloader)

    # switch to train mode
    model.train()
    start_time = time.time()
    for batch_idx, batch_data in enumerate(tqdm(train_dataloader)):
        data_time.update(time.time() - start_time)
        _, imgs, targets = batch_data
        global_step = num_iters_per_epoch * (epoch - 1) + batch_idx + 1

        batch_size = imgs.size(0)

        targets = targets.to(configs.device, non_blocking=True)
        imgs = imgs.to(configs.device, non_blocking=True)
        total_loss, outputs = model(imgs, targets)

        # For torch.nn.DataParallel case
        if (not configs.distributed) and (configs.gpu_idx is None):
            total_loss = torch.mean(total_loss)

        # compute gradient and perform backpropagation
        total_loss.backward()
        if global_step % configs.subdivisions == 0:
            optimizer.step()
            # Adjust learning rate
            if configs.step_lr_in_epoch:
                lr_scheduler.step()
                if tb_writer is not None:
                    tb_writer.add_scalar('LR', lr_scheduler.get_lr()[0], global_step)
            # zero the parameter gradients
            optimizer.zero_grad()

        if configs.distributed:
            reduced_loss = reduce_tensor(total_loss.data, configs.world_size)
        else:
            reduced_loss = total_loss.data
        losses.update(to_python_float(reduced_loss), batch_size)
        # measure elapsed time
        # torch.cuda.synchronize()
        batch_time.update(time.time() - start_time)

        if tb_writer is not None:
            if (global_step % configs.tensorboard_freq) == 0:
                tensorboard_log = get_tensorboard_log(model)
                tb_writer.add_scalar('avg_loss', losses.avg, global_step)
                for layer_name, layer_dict in tensorboard_log.items():
                    tb_writer.add_scalars(layer_name, layer_dict, global_step)

        # Log message
        if logger is not None:
            if (global_step % configs.print_freq) == 0:
                logger.info(progress.get_message(batch_idx))

        start_time = time.time()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        try:
            cleanup()
            sys.exit(0)
        except SystemExit:
            os._exit(0)
