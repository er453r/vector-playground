import cv2
import pageutils

# noinspection PyArgumentList
cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()

    pageutils.preview_ui(image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
