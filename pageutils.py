import cv2
import numpy as np

def corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 11)

    height, width = gray.shape[:2]
    min_distance = min(width, height) / 20

    features = cv2.goodFeaturesToTrack(gray, 4, 0.01, min_distance)

    return order_points_clockwise(features[:, 0])

def persp_transform(img, s_points):
    """Transform perspective from start points to target points."""
    # Euclidean distance - calculate maximum height and width
    height = max(np.linalg.norm(s_points[0] - s_points[1]),
                 np.linalg.norm(s_points[2] - s_points[3]))
    width = max(np.linalg.norm(s_points[1] - s_points[2]),
                np.linalg.norm(s_points[3] - s_points[0]))

    # Create target points
    t_points = np.array([[0, 0],
                         [0, height],
                         [width, height],
                         [width, 0]], np.float32)

    # getPerspectiveTransform() needs float32
    if s_points.dtype != np.float32:
        s_points = s_points.astype(np.float32)

    M = cv2.getPerspectiveTransform(s_points, t_points)
    return cv2.warpPerspective(img, M, (int(width), int(height)))

def order_points_clockwise(points):
    by_x = points[points[:, 0].argsort()]
    by_y = points[points[:, 1].argsort()]

    two_left = by_x[:2]
    two_left_from_the_top = two_left[two_left[:, 1].argsort()]

    two_right = by_x[-2:]
    two_right_from_the_top = two_right[two_right[:, 1].argsort()]

    sorted = np.array([two_left_from_the_top[0], two_right_from_the_top[1], two_left_from_the_top[1], two_right_from_the_top[0]])

    return sorted
