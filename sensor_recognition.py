import sys
import numpy as np
import cv2 as cv
import math


def delta(cX, cY, center):
    return (cX - center[0])**2 + (cY - center[1])**2


def find_contour(image, min_area = 0, max_area = 9999999,
                     is_resistor = False): 
    contours0, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL,
                                             cv.CHAIN_APPROX_SIMPLE)
    # перебираем все найденные контуры в цикле (поиск прямоугольника)
    XCenter = image.shape[1] / 2
    YCenter = image.shape[0] / 2
    lust_delta = 99999999
    x_final = 0
    y_final = 0
    final_box = 0

    for cnt in contours0:
        rect = cv.minAreaRect(cnt) # пытаемся вписать прямоугольник
        box = cv.boxPoints(rect) # поиск четырех вершин прямоугольник
        box_int = np.int0(box) # округление координат
        area = int(rect[1][0]*rect[1][1]) # вычисление площади

        if (area > min_area) & (area < max_area):
            if is_resistor is True:
                box_int[2:4][:,1] -= 4

            cX = (box[0][0] + box[2][0] + box[1][0] + box[3][0]) / 4 
            cY = (box[0][1] + box[2][1] + box[1][1] + box[3][1]) / 4
            cur_delta = delta(cX, cY, [XCenter, YCenter])
            if(cur_delta < lust_delta):
                x_final = cX
                y_final = cY
                final_box_accurate = box
                final_box = box_int
                
    if is_resistor is False:    
        x_axis = (tuple(((final_box_accurate[0] + final_box_accurate[1]) / 2)),
                    tuple(((final_box_accurate[2] + final_box_accurate[3]) / 2)))
        y_axis = (tuple(((final_box_accurate[1] + final_box_accurate[2]) / 2)),
                  tuple(((final_box_accurate[0] + final_box_accurate[3]) / 2))) 
        axes = (x_axis, y_axis)
        return final_box, (x_final, y_final), axes
    
    return final_box, (x_final, y_final)


def get_resistor_center(gray_image):
    normed = cv.normalize(gray_image, None, 0, 255, cv.NORM_MINMAX,
                            cv.CV_8UC1)
    normed[normed > 148] = 0
    
    kernel = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(2,2))
    eroded = cv.erode(normed, kernel, iterations = 4)
    
    kernel = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(3,3))
    opened = cv.morphologyEx(eroded, cv.MORPH_OPEN, kernel, iterations = 3)
    
    kernel = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(6,6))
    closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel, iterations = 1)
    
    closed[closed > 0] = 255
    th2 = closed    
    edges = cv.Canny(th2, 0, 255)
    
    return find_contour(edges, 55000, 120000, True) 


def get_target_center(gray_image):    
    normed = cv.normalize(gray_image, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
    
    threshold = 90
    normed[normed < 90] = 0 # 90
    while len(normed[normed > 0]) >= 380000:
        normed[normed < threshold] = 0
        threshold += 11

    kernel = cv.getStructuringElement(shape=cv.MORPH_ELLIPSE, ksize=(7,7))
    opened_1 = cv.morphologyEx(normed, cv.MORPH_OPEN, kernel, iterations = 1)
    
    kernel = cv.getStructuringElement(shape=cv.MORPH_ELLIPSE, ksize=(3,3))
    opened_2 = cv.morphologyEx(opened_1, cv.MORPH_OPEN, kernel, iterations = 4)
    
    kernel = np.ones((2,2),np.uint8)
    closed = cv.morphologyEx(opened_2, cv.MORPH_CLOSE, kernel, iterations = 2)
    
    th2 = cv.adaptiveThreshold(closed, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv.THRESH_BINARY, 33, 0)
  
    kernel = cv.getStructuringElement(shape=cv.MORPH_ELLIPSE, ksize=(5,5))
    final_img = cv.morphologyEx(th2, cv.MORPH_CLOSE, kernel, iterations = 4)
    edges = cv.Canny(final_img, 0, 255)

    return find_contour(edges, 150000)


def get_result_img(image, boxes=(), centers=()):
    offset = 0
    for box in boxes:
        offset += 150
        cv.drawContours(image, [box], 0, (255, offset, offset), 2)

    offset = 0
    for center in centers:
        offset += 150
        cv.circle(image, tuple([int(coord) for coord in center]),
                     5, (255, offset, offset))
    return image


def get_length(first_center=(0, 0), second_center=(0, 0),
                 is_axis = False, is_x = False):
    result = math.sqrt((second_center[0] - first_center[0])**2 + (second_center[1] - first_center[1])**2)
    
    if is_axis is True:
        if (is_x is True) & (second_center[1] > first_center[1]):
            result = -1 * result
        elif (is_x is False) & (second_center[0] < first_center[0]):
            result = -1 * result         
    return result

def get_intersection_perpendicular_line(line_points, point):
    # y = kx + c
    k_line = (line_points[1][1] - line_points[0][1]) \
                / (line_points[1][0] - line_points[0][0])
    c_line = line_points[1][1] - k_line * line_points[1][0]
    
    # perpendicular's k = -1/k
    k_perpendicular = -1/k_line
    c_perpendicular = point[1] - k_perpendicular * point[0]
    
    x_result = (c_line - c_perpendicular) / (k_perpendicular - k_line)
    y_result = k_line * x_result + c_line
    return (x_result, y_result)

def find_displacement_of_centers(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    box_target, center_target, axes = get_target_center(gray_image)
    
    if isinstance(box_target, int):
        return image, (9999, 9999)

    black_img = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    black_with_ROI =  cv.fillPoly(black_img, [box_target], (255, 255, 255))
    mask_ROI = cv.cvtColor(black_with_ROI, cv.COLOR_BGR2GRAY)
    image_with_ROI = cv.bitwise_and(gray_image, gray_image, mask = mask_ROI)

    box_resistor, center_resistor = get_resistor_center(image_with_ROI)
    if isinstance(box_resistor, int):
        return image, (9999, 9999)

    cv.line(image, axes[0][0], axes[0][1],
            (0,0,0), thickness = 1)
    cv.line(image, axes[1][0], axes[1][1],
            (0,0,0), thickness = 1)

    cv.putText(image, "X-axis", axes[0][1], cv.FONT_HERSHEY_SIMPLEX,
                1, (0,0,0), thickness = 2)
    cv.putText(image, "Y-axis", axes[1][0], cv.FONT_HERSHEY_SIMPLEX,
                1, (0,0,0), thickness = 2)

    intersection_x_axis = get_intersection_perpendicular_line(axes[0],
                                                                center_resistor)
    intersection_y_axis = get_intersection_perpendicular_line(axes[1],
                                                                center_resistor)
    int_center_resistor = tuple([int(coord) for coord in center_resistor])

    cv.line(image, tuple([int(coord) for coord in intersection_x_axis]),
            int_center_resistor, (0,0,255), thickness = 1)
    cv.line(image, tuple([int(coord) for coord in intersection_y_axis]),
            int_center_resistor, (0,0,255), thickness = 1)

    delta_y = get_length(intersection_x_axis, center_resistor,
                            is_axis=True, is_x=True)
    delta_x = get_length(intersection_y_axis, center_resistor,
                            is_axis=True, is_x=False)

    result_image = get_result_img(image, [box_resistor, box_target],
                                    [center_resistor, center_target])
    return result_image, (delta_x, delta_y)
