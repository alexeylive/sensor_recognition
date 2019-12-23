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
