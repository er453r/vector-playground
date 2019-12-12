import cv2
import numpy as np


def find_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 11)

    height, width = gray.shape[:2]
    min_distance = min(width, height) / 20

    features = cv2.goodFeaturesToTrack(gray, 4, 0.01, min_distance)

    return order_points_clockwise(features[:, 0])


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


def order_points_clockwise(points):
    by_x = points[points[:, 0].argsort()]

    two_left = by_x[:2]
    two_left_from_the_top = two_left[two_left[:, 1].argsort()]

    two_right = by_x[-2:]
    two_right_from_the_top = two_right[two_right[:, 1].argsort()]

    # from top-left, clockwise
    return np.array([two_left_from_the_top[0], two_right_from_the_top[0], two_right_from_the_top[1], two_left_from_the_top[1]])


def debug_image(image, corners):
    color = (0, 0, 255)
    thickness = 8

    for corner in corners:
        cv2.circle(image, tuple(corner), thickness, color, -1)

    cv2.line(image, tuple(corners[0]), tuple(corners[1]), color, round(thickness / 2))
    cv2.line(image, tuple(corners[1]), tuple(corners[2]), color, round(thickness / 2))
    cv2.line(image, tuple(corners[2]), tuple(corners[3]), color, round(thickness / 2))
    cv2.line(image, tuple(corners[3]), tuple(corners[0]), color, round(thickness / 2))

    return image
