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

    filtered_img1 = ndimage.gaussian_laplace(img1, sigma=2)
    filtered_img2 = ndimage.gaussian_laplace(img2, sigma=2)
    filtered_img1 = cv2.Canny(filtered_img1, 200, 256, apertureSize=7)
    filtered_img2 = cv2.Canny(filtered_img2, 200, 256, apertureSize=7)

    lines1 = cv2.HoughLinesP(filtered_img1, 1, np.pi/360, 120, 150, 5)
    lines2 = cv2.HoughLinesP(filtered_img2, 1, np.pi/360, 120, 150, 5)


    # TODO
    # 검출한 직선들의 기울기를 구하여 array를 만들고
    # 각 기울기 간의 차이를 가지고 2d array를 만듦
    # 이미지마다 만들어진 2d array 간의 correlation을 구해서
    # 같은 크랙은 큰 상관관계를 가지고
    # 다른 크랙은 작은 상관관계를 가지는지 확인
    # np.correlate(A,B,'full') -> 2d (cross-)correlation

    num_lines1 = len(lines1)
    slopes1 = np.zeros(num_lines1)
    for i in range(num_lines1):
        for x1,y1,x2,y2 in lines1[i]:
            slopes1[i] = (y1-y2)/(x1-x2)
    
    mat_slopes1 = np.zeros([num_lines1,num_lines1])
    for i in range(num_lines1):
        for j in range(num_lines1):
            mat_slopes1[i][j] = slopes1[i] - slopes1[j]
            

    num_lines2 = len(lines2)
    slopes2 = np.zeros(num_lines2)
    for i in range(num_lines2):
        for x1,y1,x2,y2 in lines2[i]:
            slopes2[i] = (y1-y2)/(x1-x2)

    mat_slopes2 = np.zeros([num_lines2,num_lines2])
    for i in range(num_lines2):
        for j in range(num_lines2):
            mat_slopes2[i][j] = slopes2[i] - slopes2[j]
    
    
    ccorr = signal.correlate2d(mat_slopes1, mat_slopes2, mode='full')
    print(ccorr)
    






    ### 직선 검출 후 직선 이미지 생성하는 코드 및 ORB로 특징점 추출하는 코드

    ########################################################################
    # line_img1 = np.zeros((512,512,1), np.uint8)
    # line_img2 = np.zeros((512,512,1), np.uint8)

    # for i in range(len(lines1)):
    #     for x1,y1,x2,y2 in lines1[i]:
    #         cv2.line(line_img1, (x1, y1), (x2, y2), (255, 255, 255), 7)
    # for i in range(len(lines2)):
    #     for x1,y1,x2,y2 in lines2[i]:
    #         cv2.line(line_img2, (x1, y1), (x2, y2), (255, 255, 255), 7)

    # blur_img1 = cv2.GaussianBlur(line_img1, (7,7), 10)
    # blur_img2 = cv2.GaussianBlur(line_img2, (7,7), 10)

    # cv2.imshow('filtered iamge 1', filtered_img1)
    # cv2.imshow('blurred lines 1', blur_img1)
    # cv2.imshow('filtered iamge 2', filtered_img2)
    # cv2.imshow('blurred lines 2', blur_img2)

    # orb = cv2.ORB_create(nfeatures=10000, scoreType=cv2.ORB_FAST_SCORE)
    # orb = cv2.ORB_create(nfeatures=10000)
    # kp1, des1 = orb.detectAndCompute(filtered_img1, None)

    # kpimg1 = None
    # kpimg2 = None

    # kpimg1 = cv2.drawKeypoints(blank_img, kp1, kpimg1,(0, 0, 255), flags=0)
    # kpimg2 = cv2.drawKeypoints(filtered_img2, kp2, kpimg2, (0, 0, 255), flags=0)

    # cv2.imshow('ORB1', kpimg1)
    # cv2.imshow('ORB2', kpimg2)
    ########################################################################




    # cv2.imshow('filter1', filtered_img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

Test()