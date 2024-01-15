import numpy as np
import cv2
import matplotlib
import operator as op
matplotlib.use('TkAgg')

image = cv2.imread('water10.jpeg')

mask = np.zeros(image.shape[:2], np.uint8)

backgroundModel = np.zeros((1, 65), np.float64)
foregroundModel = np.zeros((1, 65), np.float64)

rectangle = (200, 0, 110, 240)

cv2.grabCut(image, mask, rectangle, backgroundModel, foregroundModel, 3, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

image = image * mask2[:, :, np.newaxis]
edges = cv2.Canny(image, 100, 200)

max_w = 0
index = 0
for i in range(10, len(edges)-10):

    if op.countOf(edges[i], 255) > max_w:
        max_w = op.countOf(edges[i], 255)
        print(max_w)
        index = i
        print(index)

perc = (len(edges)-index) / (len(edges))
print("hh")
print(perc)
cv2.imwrite('houghlines.jpg', edges)

