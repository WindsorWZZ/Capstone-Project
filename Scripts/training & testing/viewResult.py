#This script uses the well-trained model to infer an image
#The result will be visualized

#
import detectron2
import os
from detectron2.data.datasets import register_coco_instances
#auto_anno
#register_coco_instances("T-LESS", {}, "./data/train.json", "./data/image")
#manual anno
VAL_ROOT = '/path/to/validation/directory'
VAL_JSON = os.path.join(VAL_ROOT, 'annotations.json')
register_coco_instances("T-LESS", {}, VAL_JSON, VAL_ROOT)

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
T_LESS_metadata = MetadataCatalog.get("T-LESS")
dataset_dicts = DatasetCatalog.get("T-LESS")

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import os
cfg=get_cfg()

cfg.merge_from_file("path/to/model/config/files")
#cfg.merge_from_file("/home/wzz/Downloads/T-LESS/vovnet-detectron2/configs/mask_rcnn_V_57_FPN_3x.yaml")

cfg.DATASETS.TRAIN = ("T-LESS",)
cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0002
cfg.SOLVER.MAX_ITER = (
    20000
)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    128
)  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 3 classes (obj_1, obj_2, obj_3)

model_path = "path/to/well/trained/model"
#cfg.MODEL.WEIGHTS = os.path.join(model_path, "model_final.pth")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("T-LESS", )
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode

from detectron2.utils.visualizer import Visualizer
import random
import cv2

name = "001.jpg"
addr = "/path/to/testing/image" + name
im = cv2.imread(addr)
outputs = predictor(im)
v = Visualizer(im[:, :, ::-1],
           metadata=T_LESS_metadata, 
           scale=0.8, 
           instance_mode=ColorMode.IMAGE#_BW   # remove the colors of unsegmented pixels
)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
res_img =  v.get_image()[:, :, ::-1]
cv2.imshow('show',cv2.resize(res_img, (1280, 960)))
cv2.waitKey(0)
cv2.imwrite(model_path + "/inference/"+name, res_img, [int( cv2.IMWRITE_JPEG_QUALITY), 100])
