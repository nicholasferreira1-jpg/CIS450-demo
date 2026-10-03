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


## Matplotlib Copilot Evaluation

### Setup
- Created a conda environment called `mpl-copilot` and installed matplotlib
- Cloned the matplotlib repo (ignored with .gitignore)

### Program
`plot.py` plots y = x² for x from 1 to 5 using `plt.plot(x, y)`.

### Code Trace of `plt.plot(x, y)`
1. **`pyplot.plot`** (pyplot.py): It plots data on the current axes and returns a list of Line2D objects representing the lines it created.
2. **`gca()`**: means “get the current axes.”
3. **`Axes.plot`**: You can pass data as x, y pairs, or just y values (in which case x-values are generated automatically). Optional arguments control axis scaling (scalex, scaley), provide named data (data), and set line properties such as color, marker, and style (**kwargs).
4. **`add_line()`**: adds a Line2D object to an axes so Matplotlib can draw it. It also associates the line with the axes and updates the data limits so the axes can autoscale to include it. It returns the line object. 
5. **`plt.show()`**:  uses Matplotlib to display a line plot:

Imports Matplotlib’s pyplot module as plt.
Sets x to [0, 1, 2, 3, 4] and y to [0, 1, 4, 9, 16].
Calls plt.plot(x, y) to plot the points and connect them with a line.
Calls plt.show() to display the plot.
The plotted values follow (y = x^2).

### What I learned / Copilot evaluation
- Copilot explained what each function actually does, and how it recals the functions that are inside as wwll. It's easier to understand because I can understand everystep of each function and know what is happening wtih the inputs. 
