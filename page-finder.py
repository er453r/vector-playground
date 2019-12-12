import cv2
import pageutils

image = cv2.imread("images/test-page.jpg")
preview = image.copy()

corners = pageutils.find_corners(image)

preview = pageutils.debug_image(preview, corners)
image = pageutils.perspective_transform(image, corners)

cv2.imshow("preview", preview)
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
