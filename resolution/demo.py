import cv2 as cv
import sys
import os

print("Current working directory:", os.getcwd())

img = cv.imread("photos/IMG_1.jpeg")

if img is None:
    sys.exit("Could not read the image.")

print(img.shape)

cv.namedWindow("Display window", cv.WINDOW_NORMAL)
cv.resizeWindow("Display window", 800, 600)

cv.imshow("Display window", img)
k = cv.waitKey(0)
