import cv2
import cv2.aruco as aruco
import numpy as np
import math


def average_points(points):
    return


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

    corner = corners[0]
    corner_points = corner.reshape((math.floor(np.prod(corner.shape) / 2), 2))

    distances = [np.linalg.norm(point - center) for point in corner_points]
    min_index = np.argmin(distances)
    closest_corner = corner_points[min_index]

    cv2.imshow("debug", frame_markers)
