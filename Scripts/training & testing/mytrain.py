#This script calls the modules from detectron2 and train a Mask R-CNN model

#import all the modules
import os
import cv2
import logging
from collections import OrderedDict
import detectron2.utils.comm as comm
from detectron2.utils.visualizer import Visualizer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
import torch
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data.datasets import register_coco_instances
import pycocotools
from detectron2.data.datasets import register_coco_instances

#Set the path to our datasets
TRAIN_ROOT = '/path/to/training/set/root'
VAL_ROOT = '/path/to/validation/root'
TRAIN_PATH = os.path.join(TRAIN_ROOT, 'aug_image') #<or> TRAIN_PATH = os.path.join(TRAIN_ROOT, 'image')
VAL_PATH = VAL_ROOT
TRAIN_JSON = os.path.join(TRAIN_ROOT, 'train.json')
VAL_JSON = os.path.join(VAL_ROOT, 'annotations.json')

#Register our dataset in detectron2
register_coco_instances("coco_my_train", {}, TRAIN_JSON, TRAIN_PATH)
register_coco_instances("coco_my_val", {}, VAL_JSON, VAL_PATH)
MetadataCatalog.get("coco_my_train")
DatasetCatalog.get("coco_my_train")
MetadataCatalog.get("coco_my_val")
DatasetCatalog.get("coco_my_val")

PREDEFINED_SPLITS_DATASET = {
    "coco_my_train": (TRAIN_PATH, TRAIN_JSON),
    "coco_my_test": (VAL_PATH, VAL_JSON),
}

# check the label info of our dataset
def checkout_dataset_annotation(name="coco_my_val"):
    dataset_dicts = load_coco_json(TRAIN_JSON, TRAIN_PATH)
    print(len(dataset_dicts))
    for i, d in enumerate(dataset_dicts,0):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(name), scale=1.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imwrite('out/'+str(i) + '.jpg',vis.get_image()[:, :, ::-1])
        if i == 200:
            break

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
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
        elif evaluator_type == "cityscapes":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg() # 拷贝default config副本
    add_vovnet_config(cfg)
    #set the config file, determine the backbone structure
    args.config_file = "/home/wzz/Downloads/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

    args.num_gpus = 2
    cfg.merge_from_file(args.config_file)   
    cfg.merge_from_list(args.opts)          # the setting can also merge from terminal input

    # Set the training parameters manually
    cfg.DATASETS.TRAIN = ("coco_my_train",) # Name of training/testing dataset
    cfg.DATASETS.TEST = ("coco_my_val",)
    cfg.DATALOADER.NUM_WORKERS = 10  

    cfg.INPUT.CROP.ENABLED = True         #Enable croping(part of online augmentation)
    cfg.INPUT.MAX_SIZE_TRAIN = 500        #Max size of training image
    cfg.INPUT.MIN_SIZE_TRAIN = (100, 500) #size range of training image

    ## fix size
    #cfg.INPUT.MAX_SIZE_TRAIN = 500 
    #cfg.INPUT.MIN_SIZE_TRAIN = 500 

    cfg.INPUT.MAX_SIZE_TEST = 2560 # max size of testing image
    cfg.INPUT.MIN_SIZE_TEST = 1920
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = 'range' #set image input mode to 'range'

    #class name = ['obj_1', 'obj_2', 'obj_3']
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    #use pretrained model
    cfg.MODEL.WEIGHTS = "path/to/pretrain/model"
    #can also use:
    #model_path = "/home/wzz/Downloads/T-LESS/t_less_demo/successful outputs/new_res101/no_aug"
    #cfg.MODEL.WEIGHTS = os.path.join(model_path, "model_0001331.pth")

    cfg.SOLVER.IMS_PER_BATCH = 8  # batch_size=8; iters_in_one_epoch = dataset_imgs/batch_size

    # calculate the iteration times of each epoch
    ITERS_IN_ONE_EPOCH = int(600 / cfg.SOLVER.IMS_PER_BATCH)

    # set the max iterations
    cfg.SOLVER.MAX_ITER = 4000
    #cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * 100) - 1 # 12 epochs，
    # Initial learning rate
    cfg.SOLVER.BASE_LR = 0.0002
    # optimizer momentum
    cfg.SOLVER.MOMENTUM = 0.9
    # weight decay parameters
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    # learning rate decay parameter
    cfg.SOLVER.GAMMA = 0.1
    # Learning rate decay beginning step
    cfg.SOLVER.STEPS = (7000,)
    # Warm up factor before training
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    # Warmup iterations
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.WARMUP_METHOD = "linear"
    
    cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH - 1

    # The period of cross evaluation
    cfg.TEST.EVAL_PERIOD = 200


    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    # check annotations
    checkout_dataset_annotation()

    # if we use the model for evaluation:
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