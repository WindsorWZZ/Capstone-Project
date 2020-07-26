import cv2
import  numpy as np
import os
OUT_ROOT = '/home/wzz/Downloads/T-LESS/rawData/train/trainSet/BG_image'
IMG_ROOT = '/home/wzz/Downloads/T-LESS/rawData/train/trainSet/Image'

img_WOOD=cv2.imread('/home/wzz/Downloads/T-LESS/rawData/BG_aug/WOOD.jpg')
img_QR=cv2.imread('/home/wzz/Downloads/T-LESS/rawData/BG_aug/QR.png')
rows,cols,channels = img_WOOD.shape

#遍历替换
center=[0,0]#在新背景图片中的位置

file_names = os.listdir(IMG_ROOT)
for name in file_names:
    img = cv2.imread(IMG_ROOT + '/' + name)
    num = int(name[0:4])
    if num % 2 == 1:
        cv2.imwrite(OUT_ROOT +'/' + name,img)
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, binary =  cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    #dilate and erode
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    binary = cv2.dilate(binary, kernel)
    binary = cv2.erode(binary, kernel)

    if num % 4 == 0:
        for i in range(rows):
            for j in range(cols):
                if binary[i,j]==0:#0代表黑色的点
                    img[center[0]+i,center[1]+j]=img_WOOD[i,j]#此处替换颜色，为BGR通道
    else :
        for i in range(rows):
            for j in range(cols):
                if binary[i,j]==0:#0代表黑色的点
                    img[center[0]+i,center[1]+j]=img_QR[i,j]#此处替换颜色，为BGR通道
    

    cv2.imwrite(OUT_ROOT +'/' + name,img)

