# Copyright (c) Youngwan Lee (ETRI) All Rights Reserved.
"""
VoVNet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import cv2
import logging
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.utils.visualizer import Visualizer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data.datasets.coco import load_coco_json
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    COCOEvaluator, 
    COCOPanopticEvaluator,
    SemSegEvaluator,
    DatasetEvaluators,
    verify_results
    )
from detectron2.data import DatasetCatalog, MetadataCatalog

from vovnet import add_vovnet_config
import torch
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data.datasets import register_coco_instances
import pycocotools

# 数据集路径
TRAIN_ROOT = '/home/wzz/Downloads/T-LESS/myData/trainSet'
VAL_ROOT = '/home/wzz/Downloads/T-LESS/myData/val_20'
#TRAIN_PATH = os.path.join(TRAIN_ROOT, 'image')
TRAIN_PATH = os.path.join(TRAIN_ROOT, 'aug_image')
VAL_PATH = VAL_ROOT
TRAIN_JSON = os.path.join(TRAIN_ROOT, 'train.json')
VAL_JSON = os.path.join(VAL_ROOT, 'annotations.json')

## try COCO
#TRAIN_PATH = '/home/wzz/Downloads/detectron2/datasets/coco/train2017'
#VAL_PATH = '/home/wzz/Downloads/detectron2/datasets/coco/val2017'
#TRAIN_JSON = '/home/wzz/Downloads/detectron2/datasets/coco/annotations/instances_train2017.json'
#VAL_JSON = '/home/wzz/Downloads/detectron2/datasets/coco/annotations/instances_val2017.json'
register_coco_instances("coco_my_train", {}, TRAIN_JSON, TRAIN_PATH)
register_coco_instances("coco_my_val", {}, VAL_JSON, VAL_PATH)
MetadataCatalog.get("coco_my_train")
DatasetCatalog.get("coco_my_train")
MetadataCatalog.get("coco_my_val")
DatasetCatalog.get("coco_my_val")

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        
        return DatasetEvaluators(evaluator_list)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_vovnet_config(cfg)
    args.config_file = "/home/wzz/Downloads/T-LESS/vovnet-detectron2/configs/mask_rcnn_V_57_FPN_3x.yaml"
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # 更改配置参数
    cfg.DATASETS.TRAIN = ("coco_my_train",) # 训练数据集名称
    cfg.DATASETS.TEST = ("coco_my_val",)
    cfg.DATALOADER.NUM_WORKERS = 10  # 单线程

    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.MAX_SIZE_TRAIN = 500 # 训练图片输入的最大尺寸
    cfg.INPUT.MIN_SIZE_TRAIN = (100, 500) # 训练图片输入的最小尺寸，可以吃定为多尺度训练
    cfg.INPUT.MAX_SIZE_TEST = 2560 # 测试数据输入的最大尺寸
    cfg.INPUT.MIN_SIZE_TEST = 1920
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = 'range'


    #fix size
    #cfg.INPUT.MIN_SIZE_TRAIN = 500
    #cfg.INPUT.MAX_SIZE_TRAIN = 500

    #cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    #cfg.MODEL.WEIGHTS = "/home/wzz/Downloads/T-LESS/vovnet-detectron2/checkpoints/v39_bg_aug/model_0270566.pth"
    cfg.SOLVER.IMS_PER_BATCH = 8  # batch_size=2; iters_in_one_epoch = dataset_imgs/batch_size

    # 根据训练数据总数目以及batch_size，计算出每个epoch需要的迭代次数
    ITERS_IN_ONE_EPOCH = int(600 / cfg.SOLVER.IMS_PER_BATCH)

    # 指定最大迭代次数
    cfg.SOLVER.MAX_ITER = 275000
    #cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * 100) - 1 # 12 epochs，
    # 初始学习率
    cfg.SOLVER.BASE_LR = 0.0000002
    # 优化器动能
    cfg.SOLVER.MOMENTUM = 0.9
    #权重衰减
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    # 学习率衰减倍数
    cfg.SOLVER.GAMMA = 0.1
    # 迭代到指定次数，学习率进行衰减
    cfg.SOLVER.STEPS = (7000,)
    # 在训练之前，会做一个热身运动，学习率慢慢增加初始学习率
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    # 热身迭代次数
    cfg.SOLVER.WARMUP_ITERS = 1000

    cfg.SOLVER.WARMUP_METHOD = "linear"
    # 保存模型文件的命名数据减1
    cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH - 1

    # 迭代到指定次数，进行一次评估
    #cfg.TEST.EVAL_PERIOD = ITERS_IN_ONE_EPOCH
    cfg.TEST.EVAL_PERIOD = 200


    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
