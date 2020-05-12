#注册数据集
import detectron2
import os
from detectron2.data.datasets import register_coco_instances
#auto_anno
#register_coco_instances("fruits_nuts", {}, "./data/train.json", "./data/image")
#manual anno
VAL_ROOT = '/home/wzz/Downloads/T-LESS/myData/val_09'
VAL_JSON = os.path.join(VAL_ROOT, 'annotations.json')
register_coco_instances("fruits_nuts", {}, VAL_JSON, VAL_ROOT)

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
fruits_nuts_metadata = MetadataCatalog.get("fruits_nuts")
dataset_dicts = DatasetCatalog.get("fruits_nuts")


from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import os
cfg=get_cfg()
cfg.merge_from_file(
    "/home/wzz/Downloads/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)
cfg.DATASETS.TRAIN = ("fruits_nuts",)
cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0002
cfg.SOLVER.MAX_ITER = (
    20000
)  # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    128
)  # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 3 classes (data, fig, hazelnut)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("fruits_nuts", )
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
import random
import cv2

name = "0011.jpg"
addr = "/home/wzz/Downloads/T-LESS/canon/09/rgb/" + name
im = cv2.imread(addr)
outputs = predictor(im)
v = Visualizer(im[:, :, ::-1],
           metadata=fruits_nuts_metadata, 
           scale=0.8, 
           instance_mode=ColorMode.IMAGE#_BW   # remove the colors of unsegmented pixels
)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
res_img =  v.get_image()[:, :, ::-1]
cv2.imshow('show',cv2.resize(res_img, (1280, 960)))
cv2.waitKey(0)
#cv2.imwrite("output/results_09/res_"+name, res_img, [int( cv2.IMWRITE_JPEG_QUALITY), 100])
