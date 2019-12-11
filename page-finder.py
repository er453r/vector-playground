import cv2
import pageutils

image = cv2.imread("images/test-page.jpg")

corners = pageutils.corners(image)

for i in corners:
    x, y = i.ravel()
    cv2.circle(image, (x, y), 3, 255, -1)

warp = pageutils.persp_transform(image, corners)

cv2.imshow("Image", image)
cv2.imshow("Warp", warp)
cv2.waitKey(0)
cv2.destroyAllWindows()
