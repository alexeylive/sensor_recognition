import cv2


def start_camera():
    camera_port = 0
    camera = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)
    # Check if the webcam is opened correctly
    if not camera.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        state, image_matrix = camera.read()
        cv2.imshow("Capturing", image_matrix)

    camera.release()
    cv2.destroyAllWindows()


start_camera()
