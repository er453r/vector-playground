import cv2
import cv2.aruco as aruco
import numpy as np
import math


def show(image, title, scale=0.5):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)

    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    cv2.imshow(title, resized)


def perspective_transform(img, corners, margin=0):
    height = max(np.linalg.norm(corners[0] - corners[3]),
                 np.linalg.norm(corners[1] - corners[2]))
    width = max(np.linalg.norm(corners[0] - corners[1]),
                np.linalg.norm(corners[2] - corners[3]))

    target_ratio = 291/210
    ratio = height/width
    ratio_diff = abs(target_ratio - ratio)

    print(f'ratio is {ratio} diff {ratio_diff}, {width}x{height}')

    layout = np.array([[margin, margin], [width - margin, margin], [width - margin, height - margin], [margin, height - margin]], np.float32)

    transform_matrix = cv2.getPerspectiveTransform(corners, layout)

    return cv2.warpPerspective(img, transform_matrix, (int(width), int(height)))


def markers(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    markers_image = aruco.drawDetectedMarkers(image.copy(), corners, ids)

    corners = np.asarray(corners)
    sort_indicies = np.argsort(ids, axis=0)
    corners = corners[sort_indicies]

    return corners, ids, markers_image


def outer_corners(corners):
    points = corners.reshape((math.floor(np.prod(corners.shape) / 2), 2))
    center = np.divide(np.sum(points, axis=0), points.shape[0])

    farthest_corners = []

    for corner in corners:
        corner_points = corner.reshape((math.floor(np.prod(corner.shape) / 2), 2))

        distances = [np.linalg.norm(point - center) for point in corner_points]
        min_index = np.argmax(distances)
        farthest_corners.append(corner_points[min_index])

    return np.asarray(farthest_corners)


def average_coords(image, width, height, x, y):
    error = 0.8
    margin = (1 - error) / 2

    box_width = int(image.shape[1]) / width
    box_height = int(image.shape[0]) / height

    x1 = (x + 0 + margin) * box_width
    x2 = (x + 1 - margin) * box_width
    y1 = (y + 0 + margin) * box_height
    y2 = (y + 1 - margin) * box_height

    roi = image[int(y1):int(y2), int(x1):int(x2)]

    return np.linalg.norm(cv2.mean(roi))


def debug_image(image):
    corners, ids, frame_markers = markers(image)

    farthest_corners = outer_corners(corners)

    for i in farthest_corners:
        cv2.circle(frame_markers, tuple(i), 30, 255, -1)

    transformed = perspective_transform(image, farthest_corners, margin=32)
    corners, ids, cropped = markers(transformed)
    marker_size = np.linalg.norm(corners[0][0][0][0] - corners[0][0][0][1])
    farthest_corners = outer_corners(corners)

    final = perspective_transform(transformed, farthest_corners)
    final_image = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
    width = int(final_image.shape[1] / marker_size)
    height = int(final_image.shape[0] / marker_size)

    field_width = final_image.shape[1] / width
    field_height = final_image.shape[0] / height

    print(f"Table size {width}x{height} - marker size {marker_size}, image {final_image.shape[1]}x{final_image.shape[0]}")

    average_blanks = [average_coords(final_image, width, height, x, y) for y in range(1, height-1) for x in [0, width-1]]
    average_blank = sum(average_blanks) / len(average_blanks)

    print(f"average blank {average_blank}")

    blank_margin = 10

    marked = []

    for y in range(1, height):
        for x in range(1, width-1):
            average = average_coords(final_image, width, height, x, y)

            diff = average_blank - average

            print(f"field {x}x{y} diff {diff}")

            if diff > blank_margin:
                marked.append((x, y))

    print(f"Marked size {len(marked)}")

    for point in marked:
        x, y = point
        position = (int((x + 0.5) * field_width), int((y + 0.5) * field_height))

        cv2.circle(final, position, 30, (0, 255, 0), -1)

    show(final, "cropped", 0.5)
    show(frame_markers, "debug", 0.5)
