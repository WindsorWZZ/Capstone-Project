# Instance segmentation for texture-less objects  
## Introduction
In this project, a novel instance segmentation solution for texture-less object is designed and implemented. Mask-RCNN is chosen as the deep learning model and T-LESS  is used as the training and testing dataset. The work in this project mainly focus on three points:   
1. The training set have only one object in each image, but the model will be fed with images with a bunch of piled up objects in testing stage. Data Augmentation is used to address this problem.    
2. Instead of manual labelling, all the training data are labelled automatically by image processing algorithms.
3. The initial backbone resnet is replaced by a the new vovnet, and the model reaches a higher accuracy.   

## Environment Settings
This project is based on [Detectron2](https://github.com/facebookresearch/detectron2). Please follow the instruction of detectron2 and set up a virtual environment on your linux system.   

## Model Training & Testing
After setting the environment, you can activate the virtual environment and run python scripts to train or use the models on detectron2.    
1. mytrain.py
Train and test script on Mask-RCNN, uses the basic resnet backbones in Detectron2. You can change different backbones and model hyper-parameters by editing the variables in the script.    
***training:***
```
python mytrain.py
```
***only evaluating:***
```
python mytrain.py --eval-only
```   
2. train_net.py: Same as ``mytrain.py``, using vovnet backbone.   
3. viewResult.py: Visualize the inference results.    

***Reference:***
[https://blog.csdn.net/weixin_39916966/article/details/103299051](https://blog.csdn.net/weixin_39916966/article/details/103299051)    
[https://www.dlology.com/blog/how-to-train-detectron2-with-custom-coco-datasets/](https://www.dlology.com/blog/how-to-train-detectron2-with-custom-coco-datasets/)    

## Dataset Generating
The source images are all from [T-LESS](http://cmp.felk.cvut.cz/t-less/) dataset. Auto-labelling algorithm is applied to the training set, which has only one object in each image. For testing data, manual labelling is necessary.    
### Training Set
Use morphological transformation to extract the contour of each object. Store the contour in a binary ```.png``` image. Then follow these instructions to create COCO style dataset.    
English edition: [https://patrickwasp.com/create-your-own-coco-style-dataset/](https://patrickwasp.com/create-your-own-coco-style-dataset/)    
Chinese edition: [https://blog.csdn.net/u010684651/article/details/101678268](https://blog.csdn.net/u010684651/article/details/101678268)    

### Testing Set
Use labelme to label the instances in T-LESS, than transform the json file into COCO style. [https://www.jianshu.com/p/4242171ea780](https://www.jianshu.com/p/4242171ea780)    

### Data Augmentation
There are two ways of data augmentation: Offline and online. In this project, offline augmentation is to simply change the background of the image. Online augmentation is to randomly cut or resize the image right before training. Due to limited time, augmentation methods here are not mature enough, but they did show their impact on model training. You can apply more techniques in both offline or online ways.    

## File description:
**mytrain.py:** Mask-RCNN with resnet training & testing.    
**train_net.py:** Mask-RCNN with vovnet training & testing.    
**viewResult.py:** Visualization of inference results.
**crop.py:** Cropping source images.    
**convert.py:** Contour extraction of a source image.    
**add_bg.py:** Demo of image background replacement.    
**shapes_to_coco.py:** COCO style dataset transformation in auto-labelling.    
**labelme2coco.py:** COCO style dataset transformation in manual-labelling.    
**add_bg.py:** Demo of image background replacement.