import sys
import numpy as np
import cv2 as cv
import math
import os
import matplotlib
import imutils
import keras



def prepare_to_SVM(source_image):
    return source_image.reshape(1800) / 255


def angle_bn_lines(line_1, line_2):
    input_l1 = np.array(line_1)
    input_l2 = np.array(line_2)
    
    vec_1 = input_l1[1] - input_l1[0]
    vec_2 = input_l2[1] - input_l2[0]
    
    res_dot = np.dot(vec_1, vec_2)
    res_cos = res_dot / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))
    
    return np.degrees(np.arccos(res_cos))


def get_angle_3_points(y_point, center_point, target_point):
    a = np.array(y_point)
    b = np.array(center_point)
    c = np.array(target_point)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle)
    if a[0] > c[0]:
        angle_deg = 360 - angle_deg
        
    return angle_deg


#Filter for BGR
def filter_color(input_img, color_for_filter):
    res_img = input_img.copy()
    for ind, value in enumerate(color_for_filter):
        if value is 0:
            res_img[:,:,ind] = 0
        else:
            res_img[(res_img[:,:,ind] < value - 8) 
                    | (res_img[:,:,ind] > value + 8)] = 0
    return cv.cvtColor(res_img, cv.COLOR_BGR2GRAY)


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
        if get_length(box[0], box[1]) < get_length(box[1], box[2]):
            box = np.reshape(np.append(box[1:], box[0]), (4,2))
        box_int = np.int0(box) # округление координат
        area = int(rect[1][0]*rect[1][1]) # вычисление площади
        if (area > min_area) & (area < max_area):
            #Поправки на неточность алгоритма
            if is_resistor is True:
                box_int[[1],0] -= 1
                box_int[[1],1] -= 1
                box_int[[0,2,3],0] -= 2
                box_int[[0,3],1] -= 2
                box[[0,2,3],0] -= 2
                box[[0,3],1] -= 2
                box[[1],0] -= 1
                box[[1],1] -= 1
            else:
                box_int[[3],0] -= 1
                box_int[[0, 3],1] -= 2
                box[[3],0] -= 1
                box[[0, 3],1] -= 2
                

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
    thresh = 124
    normed[normed > thresh] = 0
    while len(normed[normed > 0]) >= 1100:
        normed[normed > thresh] = 0
        thresh -= 1
    
    kernel = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(1,1))
    eroded = cv.erode(normed, kernel, iterations = 4)
    
    kernel = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(2,2))
    opened = cv.morphologyEx(eroded, cv.MORPH_OPEN, kernel, iterations = 2)
    
    kernel = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(10,5))
    closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel, iterations = 1)
    
    '''
    closed[closed > 0] = 255
    cv.imshow('result11', normed) 
    cv.waitKey(0)        
    cv.destroyAllWindows()
    cv.imshow('result11', eroded) 
    cv.waitKey(0)        
    cv.destroyAllWindows()
    cv.imshow('result11', opened) 
    cv.waitKey(0)        
    cv.destroyAllWindows()
    cv.imshow('result11', closed) 
    cv.waitKey(0)        
    cv.destroyAllWindows()
    '''
    edges = cv.Canny(closed, 0, 255)
    
    
    return find_contour(edges, 1000, 2000, True) 


def get_target_center(gray_image):    
    normed = cv.normalize(gray_image, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
    thresh = 185
    normed[normed > thresh] = 0 # 90
    while len(normed[normed > 0]) >= 3400:
        normed[normed > thresh] = 0
        thresh -= 1
    #print(len(normed[normed > 0]))
    
    kernel = cv.getStructuringElement(shape=cv.MORPH_ELLIPSE, ksize=(1,1))
    eroded = cv.erode(normed, kernel, iterations = 5)
    
    kernel = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(2,2))
    opened = cv.morphologyEx(eroded, cv.MORPH_OPEN, kernel, iterations = 2)
    
    
    kernel = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(5,5))
    closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel, iterations = 1)
    
    '''
    cv.imshow('result11', normed) 
    cv.waitKey(0)        
    cv.destroyAllWindows()
    cv.imshow('result11', eroded) 
    cv.waitKey(0)        
    cv.destroyAllWindows()
    cv.imshow('result11', opened) 
    cv.waitKey(0)        
    cv.destroyAllWindows()
    cv.imshow('result11', closed) 
    cv.waitKey(0)        
    cv.destroyAllWindows()
    '''
    
    closed[closed > 0] = 255   
    edges = cv.Canny(closed, 0, 255)
    
    return find_contour(edges, 2000, 4400, False) 


def get_result_img(image, boxes=(), centers=()):
    res_img = image.copy()
    offset = 0
    for box in boxes:
        offset += 150
        cv.drawContours(res_img, [box], 0, (0, offset, offset), 2)

    offset = 0
    for center in centers:
        offset += 150
        cv.circle(res_img, tuple([int(coord) for coord in center]),
                     3, (0, offset, offset), 2)
    return res_img


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
    
    if k_line == 0:
        return (point[0], c_line)
    
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
    
    angle = angle_bn_lines(axes[1],
                           (box_resistor[1], box_resistor[0])) - 1.5

    result_image = get_result_img(image, [box_resistor, box_target],
                                    [center_resistor, center_target])
    
    return result_image, (delta_x, delta_y), angle


def prepare_to_NN(source_image):
    res_image = source_image.copy()
    res_image = res_image.reshape(60, 30, 1)
    res_image = res_image / 255
    res_image = res_image.astype('float32')
    return np.array([res_image])


def get_model_for_serial_num(path_to_model):
    with open(path_to_model, 'rb') as fid:
        model = pickle.load(fid)
    return model
    

def get_serial_num(img, model):
    frame = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)
    frame = cv.normalize(frame, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
    w1, h1 = int(frame.shape[1] /2- 135), int(frame.shape[0] /2 + 88)
    w2, h2 = int(frame.shape[1] / 2 + 135), int(frame.shape[0] / 2 + 152)
    crop_img = frame[h1:h2, w1:w2]
    shift = 0
    delta = int(crop_img.shape[1] / 8)
    images = []
    answer = ''
    for i in range(8):
        if i is 6:
            continue
        img = crop_img[:,2 + delta*i:2 + delta + delta*i]
        img = cv.resize(img, (30,60))
        img = cv.medianBlur(img, 3)
        res = new_model.predict_classes(prepare_to_NN(img))
        answer += str(res[0])
    return answer


def get_contour_of_element(img):
    # Поиск круглых контуров
    blured = cv.medianBlur(frame, 7)
    circles = cv.HoughCircles(blured, cv.HOUGH_GRADIENT, 2,
                          minDist=1500, minRadius=510, maxRadius = 545)

    # Поиск круглых контуров
    if len(circles) == 1:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            r_element = r
            x_center_element, y_center_element = x, y
        return r_element, x_center_element, y_center_element
    else:
        return "Error"
    
    
def get_guide_point(img, elem_contour):
    #Определение контура уникального отверстия
    r_element, x_center_element, y_center_element = elem_contour
    #Создание маски
    frame = img.copy()
    outer_mask = np.zeros(frame.shape, dtype=np.uint8)
    inner_mask = np.zeros(frame.shape, dtype=np.uint8)
    cv.circle(outer_mask, (x_center_element, y_center_element),
              int(0.85 * r_element), (255,255,255), -1 , 8, 0)
    cv.circle(inner_mask, (x_center_element, y_center_element),
              int(0.69 * r_element), (255,255,255), -1, 8, 0)
    #Морфологические преобразования над серым изображением
    fin = cv.GaussianBlur(frame.copy(),(9, 9), 2, 2)
    fin[fin < 60] = 255
    fin[fin < 255] = 0
    #Наложение масок
    fin_masked = cv.bitwise_and(fin, fin, mask = outer_mask)
    fin_masked = fin_masked - inner_mask
    #Поиск контуров под уникальное отверстие
    contours0, hierarchy = cv.findContours(fin_masked, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        area = cv.contourArea(cnt)
        if (area > 600) and  (area < 4000):
            (_1, _2, w, h) = cv.boundingRect(cnt)
            if (w > 70) or (h > 70):
                continue
            M = cv.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
    x_guide_point = cX
    y_guide_point = cY
    
    return x_guide_point, y_guide_point


def get_normalized_element(img, elem_contour, guide_point):
    #Заданиче координат точек
    r_element, x_center_element, y_center_element = elem_contour
    x_guide_point, y_guide_point = guide_point
    
    frame = img.copy()
    
    start_point = (x_center_element, 1080 - y_center_element)
    y_point = (x_center_element, 1080 - y_center_element + 10)
    guide_point = (x_guide_point, 1080 - y_guide_point)

    mask = np.zeros(frame.shape,dtype=np.uint8)
    cv.circle(mask, (x_center_element,y_center_element),
              int(r_element), (255,255,255), -1, 8, 0)
    imageROI = cv.bitwise_and(frame, frame, mask=mask)
    rotated = imutils.rotate_bound(imageROI, 360 - get_angle_3_points(y_point, start_point, guide_point))

    (x, y, w, h) = cv.boundingRect(rotated)
    rotated = rotated[y:y+h, x:x+w]
    
    return rotated

def get_sep_resistors(norm_elem, resistors_info, gauge_schema):
    resistors_info = element_types[212]['resistors']
    #BGR Filter
    w, h = rotated.shape
    resistor_mask = cv.resize(gauge_schema[:,30:,:], (w,h))
    flag_is_filtered = False
    resistor_imgs = []

    for resistor in resistors_info.items():
        for val_filter in resistor[1]['colors']:
            if flag_is_filtered:
                temp_mask += filter_color(resistor_mask, val_filter)
            else:
                temp_mask = filter_color(resistor_mask, val_filter)
                flag_is_filtered = True

        temp_mask[temp_mask > 0] = 255
        kernel = cv.getStructuringElement(shape=cv.MORPH_ELLIPSE, ksize=(3,3))
        temp_mask = cv.morphologyEx(temp_mask, cv.MORPH_OPEN, kernel, iterations = 3)
        temp_mask = cv.morphologyEx(temp_mask, cv.MORPH_CLOSE, kernel, iterations = 4)
        fin = cv.bitwise_and(norm_elem, norm_elem, mask = temp_mask)
        flag_is_filtered = False
        resistor_imgs.append(fin)
        
    return resistor_imgs


def get_displ_for_several_res(norm_elem, resistors_imgs, resistors_info):
    image_res = norm_elem.copy()
    scale_list = []
    results = {}
    angles = {}

    for ind, result in enumerate(resistor_imgs):
        box_t, center_t, axes = get_target_center(result)
        long_lnth = get_length(box_t[0], box_t[1])
        if long_lnth >= 68 and long_lnth <= 75:
            scale_list.append(7 / long_lnth)
        long_lnth = get_length(box_t[2], box_t[3])
        if long_lnth >= 68 and long_lnth <= 75:
            scale_list.append(7 / long_lnth)

        black_img = np.zeros((norm_elem.shape[0], norm_elem.shape[1], 3), np.uint8)
        black_with_ROI = cv.fillPoly(black_img, [box_t], (255, 255, 255))
        mask_ROI = cv.cvtColor(black_with_ROI, cv.COLOR_BGR2GRAY)
        image_with_ROI = cv.bitwise_and(norm_elem, norm_elem, mask = mask_ROI)
        box_resistor, center_resistor = get_resistor_center(image_with_ROI)
        get_result_img(image_res, [box_t, box_resistor], [center_t, center_resistor])
        intersection_x_axis = get_intersection_perpendicular_line(axes[0],
                                                                    center_resistor)
        intersection_y_axis = get_intersection_perpendicular_line(axes[1],
                                                                    center_resistor)
        int_center_resistor = tuple([int(coord) for coord in center_resistor])

        delta_y = get_length(intersection_x_axis, center_resistor,
                                is_axis=True, is_x=True)
        delta_x = get_length(intersection_y_axis, center_resistor,
                                is_axis=True, is_x=False)

        results[ind+1] = np.array([delta_x, delta_y])
        angles[ind+1] = angle_bn_lines(axes[1],
                                       (box_resistor[1], box_resistor[0])) - 1.5

        scale = np.average(scale_list)

    for item in results.items():
        current_angle = resistors_info[item[0]]['angle']    
        results[item[0]] = item[1] * scale
        if (current_angle > 90) and (current_angle < 270):
            results[item[0]] = -1 * results[item[0]]
    
    return image_res, results, angles

def get_displ_by_element(img, resistors_info, gauge_schema):
    frame = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)
    frame = cv.normalize(frame, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
    elem_contour = get_contour_of_element(frame)
    guide_point = get_guide_point(frame, elem_contour)
    norm_elem = get_normalized_element(frame, elem_contour, guide_point)
    resistors_imgs = get_sep_resistors(norm_elem, resistors_info, gauge_schema)
    image_res, results, angles = get_displ_for_several_res(norm_elem, resistors_imgs,
                                                           resistors_info)
    return image_res, results, angles
