from scipy import ndimage
import numpy as np
import cv2

def Test():
    img1 = cv2.imread('img/1-1.tif')
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #img2 = cv2.imread('img/1-2.tif')
    kpimg1 = None
    #kpimg2 = None

    filtered_img1 = ndimage.gaussian_laplace(gray, sigma=2)
    filtered_img1 = cv2.Canny(filtered_img1, 350, 360, apertureSize=5)
    #filtered_img2 = ndimage.gaussian_laplace(img2, sigma=1)

    lines1 = cv2.HoughLinesP(filtered_img1, 1, np.pi/360, 140, 250, 10)

    for i in range(len(lines1)):
        for x1,y1,x2,y2 in lines1[i]:
            cv2.line(img1, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('dd', img1)
    cv2.imshow('filter1', filtered_img1)



    orb = cv2.ORB_create(nfeatures=10000, scoreType=cv2.ORB_FAST_SCORE)
    #orb = cv2.ORB_create(nfeatures=10000)
    kp1, des1 = orb.detectAndCompute(filtered_img1, None)

    kpimg1 = cv2.drawKeypoints(filtered_img1, kp1, kpimg1, (0, 0, 255), flags=0)
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