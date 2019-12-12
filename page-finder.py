import cv2
import pageutils

image = cv2.imread("images/test-page.jpg")

pageutils.preview_ui(image)

cv2.waitKey(0)
cv2.destroyAllWindows()
