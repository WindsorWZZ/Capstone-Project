#This script crops the pictures, inorder to get more information during training.
import cv2 as cv
import os
crop_path = '/path/to/output/directory'
path = '/path/to/input/directory'

file_names = os.listdir(path)
for name in file_names:
    src = cv.imread(path + '/' + name)
    crop = src[400:1400, 400:1400]
    cv.imwrite(crop_path +'/' + name[0:4] + '.jpg',crop)