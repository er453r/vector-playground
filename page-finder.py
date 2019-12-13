import cv2

import utils

image = cv2.imread("images/aruco-test.jpg")

utils.debug_image(image)

cv2.waitKey(0)
cv2.destroyAllWindows()
