from scipy import ndimage
from scipy import signal
import numpy as np
import pandas as pd
import cv2

def Test():
    img1 = cv2.imread('img/1-1.tif', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('img/1-2.tif', cv2.IMREAD_GRAYSCALE)
    img1 = cv2.resize(img1, (512,512), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (512,512), interpolation=cv2.INTER_AREA)
    line_img1 = np.zeros((512,512,1), np.uint8)
    line_img2 = np.zeros((512,512,1), np.uint8)

    #kpimg1 = None
    #kpimg2 = None

    filtered_img1 = ndimage.gaussian_laplace(img1, sigma=2)
    filtered_img2 = ndimage.gaussian_laplace(img2, sigma=2)
    filtered_img1 = cv2.Canny(filtered_img1, 200, 256, apertureSize=7)
    filtered_img2 = cv2.Canny(filtered_img2, 200, 256, apertureSize=7)

    lines1 = cv2.HoughLinesP(filtered_img1, 1, np.pi/360, 120, 150, 5)
    lines2 = cv2.HoughLinesP(filtered_img2, 1, np.pi/360, 120, 150, 5)

    for i in range(len(lines1)):
        for x1,y1,x2,y2 in lines1[i]:
            cv2.line(line_img1, (x1, y1), (x2, y2), (255, 255, 255), 7)
    for i in range(len(lines2)):
        for x1,y1,x2,y2 in lines2[i]:
            cv2.line(line_img2, (x1, y1), (x2, y2), (255, 255, 255), 7)

    blur_img1 = cv2.GaussianBlur(line_img1, (7,7), 10)
    blur_img2 = cv2.GaussianBlur(line_img2, (7,7), 10)

    cc = signal.correlate2d(blur_img1, blur_img2)
    print(cc)

    #cv2.imshow('filtered iamge 1', filtered_img1)
    cv2.imshow('blurred lines 1', blur_img1)
    #cv2.imshow('filtered iamge 2', filtered_img2)
    cv2.imshow('blurred lines 2', blur_img2)



    #orb = cv2.ORB_create(nfeatures=10000, scoreType=cv2.ORB_FAST_SCORE)
    #orb = cv2.ORB_create(nfeatures=10000)
    #kp1, des1 = orb.detectAndCompute(filtered_img1, None)

    #kpimg1 = cv2.drawKeypoints(blank_img, kp1, kpimg1, (0, 0, 255), flags=0)
    #kpimg2 = cv2.drawKeypoints(filtered_img2, kp2, kpimg2, (0, 0, 255), flags=0)

    #cv2.imshow('filter1', filtered_img1)
    #cv2.imshow('ORB1', kpimg1)
    #cv2.imshow('ORB2', kpimg2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

Test()

#TODO
# 여러 이미지를 불러들인다.
# ORB나 여러 방법으로 특징점 추출한다.
# 각 이미지들의 특징점 매트릭스들간의 correlation을 구한다.
# 특징점 간의 correlation을 어떤 식으로 구해야하나