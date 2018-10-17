import numpy as np
import cv2

def ORB():
    img = cv2.imread('img/test.tif')
    img2 = None

    orb = cv2.ORB_create(nfeatures=5000)
    kp, des = orb.detectAndCompute(img, None)

    img2 = cv2.drawKeypoints(img, kp, img2, (0, 0, 255), flags=0)

    cv2.imshow('ORB', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

ORB()