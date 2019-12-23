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
