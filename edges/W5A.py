# create image with edges
import cv2
import os

# input filename
filename = f"edges/art.png"
#filename = f"edges/frog.png"
#filename = f"edges/map.png"
#filename = f"edges/pokemon.png"
#filename = f"edges/sunset.png"

# output filename
base, _ = os.path.splitext(filename)
outfile = f"{base}.edges.jpg"

# load color image
color = cv2.imread(filename)

# convert to grayscale
gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

# create the output window
cv2.namedWindow('image')

# create trackbars
cv2.createTrackbar('blend', 'image', 0, 100, lambda x: None)
cv2.createTrackbar('thresh', 'image', 0, 255, lambda x: None)
cv2.createTrackbar('blur', 'image', 0, 31, lambda x: None)

# loop to show images
while True:
    # retrieve trackbar positions
    blend = cv2.getTrackbarPos('blend', 'image')
    thresh = cv2.getTrackbarPos('thresh', 'image')
    k = cv2.getTrackbarPos('blur', 'image')

    # ensure kernel k is an odd value
    k = max(1, k)
    if k % 2 == 0:
        k += 1
    k = min(31, k)
    cv2.setTrackbarPos('blur', 'image', k)

    # compute blending factors
    alpha = blend / 100.0
    beta = 1.0 - alpha

    # blur the image
    blur = cv2.GaussianBlur(gray, (k, k), 0)

    # compute X and Y derivatives and gradient magnitude
    dx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(dx, dy)
    grad = cv2.convertScaleAbs(mag)

    # threshold the gradient to get binary edges
    _, edges = cv2.threshold(grad, thresh, 255, cv2.THRESH_BINARY)

    # convert grayscale edges to color
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # blend color image and edge image
    blended = cv2.addWeighted(color, beta, edges_color, alpha, 0)

    # display the blended image
    cv2.imshow('image', blended)

    # wait for key press and exit on 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# close window
cv2.destroyAllWindows()
cv2.imwrite(outfile, blended)