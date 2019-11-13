#!/usr/bin/env python
import sys
import numpy as np
import cv2 as cv
import math


def delta(cX, cY, center):
    return (cX - center[0])**2 + (cY - center[1])**2

def is_on_border(contour, is_source = False):
    
    borders_x = set([i for i in range(int(img.shape[1] * 0.98), img.shape[1]+1)])
    borders_y = set([i for i in range(int(img.shape[0] * 0.98), img.shape[0]+1)])
    borders_x.update([i for i in range(0, int(img.shape[1] * 0.02)+1)])
    borders_y.update([i for i in range(0, int(img.shape[0] * 0.02)+1)])
    x_val = set()
    y_val = set()
    
    for i in range(len(contour)):
        x_val.add(contour[i][0][0])
        y_val.add(contour[i][0][1])
    
    flag_crossed = len(y_val.intersection(borders_y)) != 0
    if not is_source:
            flag_crossed = flag_crossed or len(x_val.intersection(borders_x)) != 0
            
    if flag_crossed:
        return True
    else: 

        return False

def drop_trash(image, is_source = False):
    contours0, hierarchy = cv.findContours( image.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    mask = np.ones(image.shape[:2], dtype="uint8") * 255
    for ind ,cnt in enumerate(contours0):
        if is_on_border(cnt, is_source):
            cv.drawContours(mask, [cnt], -1, 0, -1)
    image_result = cv.bitwise_and(image, image, mask=mask)
    return image_result


def ContOnBord(AllContours):
    res = []
    for ind, cnt in enumerate(AllContours):
        if isOnBorder(cnt):
            res.append(ind)
    return res


def find_contour(image, min_area = 0): 
    contours0, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # перебираем все найденные контуры в цикле (поиск прямоугольника)
    XCenter = image.shape[1] / 2
    YCenter = image.shape[0] / 2
    lust_delta = 99999999
    X_Final = 0
    Y_Final = 0
    final_box = 0

    for cnt in contours0:
        rect = cv.minAreaRect(cnt) # пытаемся вписать прямоугольник
        box = cv.boxPoints(rect) # поиск четырех вершин прямоугольника
        box = np.int0(box) # округление координат
        area = int(rect[1][0]*rect[1][1]) # вычисление площади
        if area > min_area:
            cX = (box[0][0] + box[2][0] + box[1][0] + box[3][0]) / 4 
            cY = (box[0][1] + box[2][1] + box[1][1] + box[3][1]) / 4
            cur_delta = delta(cX, cY, [XCenter, YCenter])
            if(cur_delta < lust_delta):
                X_Final = cX
                Y_Final = cY
                final_box = box
    return final_box, (X_Final, Y_Final)

def get_target_center(gray_image):
    image = gray_image.copy()
    image[image < 105] = 0
    image[image > 255] = 0
    
    normed = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
    kernel = cv.getStructuringElement(shape=cv.MORPH_ELLIPSE, ksize=(5,5))
    opened = cv.morphologyEx(normed, cv.MORPH_OPEN, kernel, iterations = 2)
    kernel = np.ones((2,2),np.uint8)
    closedd = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel, iterations = 3)
    kernel = cv.getStructuringElement(shape=cv.MORPH_ELLIPSE, ksize=(5,5))
    
    th2 = cv.adaptiveThreshold(closedd, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 33, 0)
    kernel = cv.getStructuringElement(shape=cv.MORPH_ELLIPSE, ksize=(5,5))
    final_img = cv.morphologyEx(th2, cv.MORPH_CLOSE, kernel, iterations = 5)
    edges = cv.Canny(final_img, 0, 255)
    
    res_img = drop_trash(edges, True)
    
    return find_contour(res_img, 150000)

def get_resistor_center(gray_image):
    image = gray_image.copy()
    image[image < 80] = 0
    image[image > 145] = 0
    
    normed = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
    kernel = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(3,3))
    opened = cv.morphologyEx(normed, cv.MORPH_OPEN, kernel, iterations = 3)
    kernel = np.ones((3,3),np.uint8)
    closedd = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel, iterations = 5)

    th2 = cv.adaptiveThreshold(closedd, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 51, 0)

    kernel = cv.getStructuringElement(shape=cv.MORPH_ELLIPSE, ksize=(3,3))
    final_img = cv.morphologyEx(th2, cv.MORPH_OPEN, kernel, iterations = 5)
    kernel = np.ones((3,3),np.uint8)
    final_img = cv.morphologyEx(final_img, cv.MORPH_CLOSE, kernel, iterations = 2)
    kernel = np.ones((20,20),np.uint8)
    final_img = cv.morphologyEx(final_img, cv.MORPH_OPEN, kernel)
    kernel = cv.getStructuringElement(shape=cv.MORPH_ELLIPSE, ksize=(10,10))
    final_img = cv.morphologyEx(final_img, cv.MORPH_CLOSE, kernel, iterations = 20)
    
    res_img = drop_trash(final_img)
    
    return find_contour(res_img, 80000)

def show_result_img(image, boxes = [], centers = []):
    img = image.copy()
    delta = 0
    for box in boxes:
        delta = delta + 150
        cv.drawContours(img,[box],0,(255,delta,delta),2)
    delta = 0
    for center in centers:
        delta = delta + 150
        cv.circle(img, tuple(map(lambda x: int(x),center)), 5, (255,delta,delta))
        
    #cv.imshow('Color all', img) # вывод обработанного кадра в окно
    #cv.waitKey()
    #cv.destroyAllWindows()
    return img


def get_length(first_center=[0, 0], second_center=[0, 0]):
    return math.sqrt((second_center[0] - first_center[0])**2 + (second_center[1] - first_center[1])**2)

def get_direction_vector(first_point, second_point):
    return second_point - first_point

def get_angle_between_lines(line_1, line_2):
    pass

def get_image_with_reolagram,
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    box_target, center_target = get_target_center(gray_image)
    box_mini, center_resistor = get_resistor_center(gray_image)
    return show_result_img(img, [box_target, box_mini], [center_target, center_resistor])