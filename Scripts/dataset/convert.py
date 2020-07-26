#This script converts the pictures to binary mode, which stores their contour information
import cv2
import  numpy as np
import os
OUT_ROOT = '/path/to/output/directory'
IMG_ROOT = '/path/to/source/image/directory'
img_WOOD = cv2.imread('/path/to/wooden/background/picture')
img_QR = cv2.imread('/path/to/QRcode/background/picture')
rows, cols, channels = img_WOOD.shape

#Traverse and replace each pixel
center=[0, 0]  #The relative position between source image and background image
file_names = os.listdir(IMG_ROOT)
for name in file_names:
    img = cv2.imread(IMG_ROOT + '/' + name)
    num = int(name[0 : 4])
    if num % 2 == 1:
        cv2.imwrite(OUT_ROOT + '/' + name, img)
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
                if binary[i, j] == 0:
                    img[center[0] + i,center[1] + j] = img_WOOD[i, j] 

    else :
        for i in range(rows):
            for j in range(cols):
                if binary[i,j] == 0:  
                    img[center[0] + i, center[1] + j]=img_QR[i, j] 
    

    cv2.imwrite(OUT_ROOT +'/' + name,img)