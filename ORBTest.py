import numpy as np
import cv2

def ORB():
    img1 = cv2.imread('img/1-1.jpg')
    img2 = cv2.imread('img/1-1_a.jpg')
    kpimg1 = None
    kpimg2 = None

    #orb = cv2.ORB_create(nfeatures=10000, scaleFactor=2, scoreType=cv2.ORB_FAST_SCORE)
    #orb = cv2.ORB_create(nfeatures=10000, scoreType=cv2.ORB_FAST_SCORE)
    orb = cv2.ORB_create(nfeatures=10000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    kpimg1 = cv2.drawKeypoints(img1, kp1, kpimg1, (0, 0, 255), flags=0)
    kpimg2 = cv2.drawKeypoints(img2, kp2, kpimg2, (0, 0, 255), flags=0)

    cv2.imshow('ORB1', kpimg1)
    cv2.imshow('ORB2', kpimg2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

ORB()

#TODO
# 여러 이미지를 불러들인다.
# ORB나 여러 방법으로 특징점 추출한다.
# 각 이미지들의 특징점 매트릭스들간의 correlation을 구한다.
# 특징점 간의 correlation을 어떤 식으로 구해야하나