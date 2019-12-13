import matplotlib.pyplot as plt
import math
import matplotlib as mpl
import cv2.aruco as aruco


# noinspection PyUnresolvedReferences
def draw(width, height, marker_index):
    width += 2  # add technical columns
    markers = aruco.Dictionary_get(aruco.DICT_6X6_250)

    fig = plt.figure(figsize=(width*2, height*2))  # Notice the equal aspect ratio
    ax = [fig.add_subplot(height, width, i+1) for i in range(width * height)]

    for i in range(len(ax)):
        a = ax[i]
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.set_aspect('equal')
        a.set_xticks([], [])
        a.set_yticks([], [])

        x = i % width + 1
        y = math.floor(i / width) + 1

        if x == 1 or x == width:
            a.axis('off')

        if x == 1 and y == 1:
            img = aruco.drawMarker(markers, marker_index, 700)
            a.imshow(img, cmap=mpl.cm.gray, interpolation="nearest")

        if x == width and y == 1:
            img = aruco.drawMarker(markers, marker_index+1, 700)
            a.imshow(img, cmap=mpl.cm.gray, interpolation="nearest")

        if x == width and y == height:
            img = aruco.drawMarker(markers, marker_index+2, 700)
            a.imshow(img, cmap=mpl.cm.gray, interpolation="nearest")

        if x == 1 and y == height:
            img = aruco.drawMarker(markers, marker_index+3, 700)
            a.imshow(img, cmap=mpl.cm.gray, interpolation="nearest")

    fig.subplots_adjust(wspace=0, hspace=0)

    plt.savefig("markers.pdf")
    plt.show()


draw(6, 8, 1)
