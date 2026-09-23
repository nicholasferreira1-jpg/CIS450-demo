# CIS450-demo 


## Introduction

This repo illustrates best practice README file generation.


## Projects

Open-CV image processing demos.


## Resources

![OpenCV logo](OpenCV_logo_black_.png)
[Open-CV](https://opencv.org/)


## Edge Detection (W5A)

### How it works

This program finds the **edges** (outlines) in an image and blends them with the original picture.

1. **Grayscale** – the color image is converted to black and white, since edges are about brightness, not color.
2. **Blur** – the image is smoothed slightly to remove noise, so tiny specks don't get mistaken for edges.
3. **Find edges** – the program looks for spots where brightness changes quickly (like the outline of an object against its background). These spots become the "edges."
4. **Threshold** – a cutoff value decides which changes are strong enough to count as a real edge, versus which are too weak and get ignored.
5. **Blend** – the edge outline is combined with the original color image, so you can see both at once.

There are three settings you can adjust:
- **Blend** – how much of the edge outline shows up compared to the original image
- **Threshold** – how strong a brightness change needs to be to count as an edge
- **Blur** – how much smoothing happens before edges are detected (higher blur = fewer small/noisy edges)
