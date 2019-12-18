import cv2
import cv2.aruco as aruco
import numpy as np
import math

def show(image, title, scale=0.5):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)

    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    cv2.imshow(title, resized)

def perspective_transform(img, corners):
    height = max(np.linalg.norm(corners[0] - corners[3]),
                 np.linalg.norm(corners[1] - corners[2]))
    width = max(np.linalg.norm(corners[0] - corners[1]),
                np.linalg.norm(corners[2] - corners[3]))

    target_ratio = 291/210
    ratio = height/width
    ratio_diff = abs(target_ratio - ratio)

    print(f'ratio is {ratio} diff {ratio_diff}, {width}x{height}')

    layout = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)

    transform_matrix = cv2.getPerspectiveTransform(corners, layout)

    return cv2.warpPerspective(img, transform_matrix, (int(width), int(height)))


def debug_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    frame_markers = aruco.drawDetectedMarkers(image.copy(), corners, ids)

    corners = np.asarray(corners)
    sort_indicies = np.argsort(ids, axis=0)
    corners = corners[sort_indicies]
    points = corners.reshape((math.floor(np.prod(corners.shape) / 2), 2))
    center = np.divide(np.sum(points, axis=0), points.shape[0])

    farthest_corners = []

    for corner in corners:
        corner_points = corner.reshape((math.floor(np.prod(corner.shape) / 2), 2))

        distances = [np.linalg.norm(point - center) for point in corner_points]
        min_index = np.argmax(distances)
        farthest_corners.append(corner_points[min_index])

    farthest_corners = np.asarray(farthest_corners)

    for i in farthest_corners:
        cv2.circle(frame_markers, tuple(i), 30, 255, -1)

    cropped = perspective_transform(frame_markers, farthest_corners)

    show(cropped, "cropped", 0.5)
    show(frame_markers, "debug", 0.5)
