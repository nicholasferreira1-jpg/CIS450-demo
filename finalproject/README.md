# Final Project

## Description
My final project is a web application, deployed as a Docker image on Render, that turns photos into cartoons. The user uploads an image through the web page, and the app uses OpenCV to smooth the colors and outline the edges, then displays the cartoon version in the browser. It builds on the edge-detection template provided in class.

## Design
I am approaching this project by prototyping on top of the class template, which already has a working Flask web page, image upload, and Docker setup that runs on Render. 
My plan is to keep the template's upload-and-display flow and replace the edge-detection step with a cartoon effect. From my research, the common cartoonizer method has three steps: 
- convert the image to grayscale, blur it, and use adaptive thresholding to create bold black outlines;
- smooth the colors with a bilateral filter, which flattens colors while keeping edges sharp; 
- combine the outlines with the smoothed colors using a bitwise AND. 

I will first test this as a small Python script on the smiley image and a few photos, then move it into app.py. Later, I'd like to add sliders so the user can adjust outline thickness and color smoothing, similar to the threshold control in the template. One concern is Render's free-plan limit of 512 MB of RAM, because the bilateral filter is slow and memory-heavy on large photos, so I plan to resize large uploads before processing.

Websites I used to figure out how to code this:
- OpenCV Smoothing Images tutorial (bilateral filter): https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
- OpenCV Image Thresholding tutorial (adaptive threshold): https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
- GeeksforGeeks, Cartooning an Image using OpenCV – Python: https://www.geeksforgeeks.org/cartooning-an-image-using-opencv-python/
- Cartoonify an Image with OpenCV in Python (DataFlair): https://data-flair.training/blogs/cartoonify-image-opencv-python/
- Render free plan limits: https://render.com/docs/free

AI usage: I used Claude (Anthropic) during this phase. I asked it to suggest project ideas that fit the template. I also asked how a cartoon effect is made in OpenCV, which led me to the bilateral filter and adaptive threshold approach above.