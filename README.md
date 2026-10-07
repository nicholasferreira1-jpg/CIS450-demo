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


# Count Coins – Debugging AI

`findcoins.py` finds every coin in a photo, circles it by type
(Pennies = Red, Nickels = Green, Dimes = Blue, Quarters = Yellow), and writes
the counts and total value on the image.

```
python findcoins.py              # runs on coins1.png - coins4.png
python findcoins.py coins2.png   # runs on one image
```

Each `coinsN.png` produces `coinsN.annotated.png`.

## Results

| Image | Pennies | Nickels | Dimes | Quarters | Total |
|-------|--------:|--------:|------:|---------:|------:|
| coins1 | 4 | 0 | 1 | 0 | **$0.14** |
| coins2 | 1 | 1 | 1 | 1 | **$0.41** |
| coins3 | 4 | 1 | 1 | 0 | **$0.19** |
| coins4 | 19 | 7 | 7 | 17 | **$5.49** |

coins1–coins3 match the teacher's results. For coins4 I got $5.49 instead of
$5.09: the 19 pennies and 7 dimes match, but I count 7 nickels and 17
quarters instead of 9 and 15. I checked by cropping every silver coin: all 7
"nickels" show Jefferson or Monticello, and all 17 "quarters" show
Washington, an eagle, or a state design. Their sizes also form two separate
groups with the same size ratio as real nickels and quarters (1.144).

## Conversation with AI (Claude)

**Prompt 1:** *"Help me Create a Python Program called findcoins.py, to annotate the coins in the input image coinsN.png and save as
coinsN.annotated.png and compute to total value, which is displayed in the annotated image
somewhere. Label the coins Pennies = Red, Nickels = Green, Dimes = Blue, Quarters = Yellow."* (with the 4 coin images attached and the teachers result as well)

- The AI pulled the four coin images and the teacher's results so it could check its answers.
- **First try:** OpenCV's normal circle finder (`HoughCircles`) found
  hundreds of fake circles in the wood grain and the coin engravings.
- **Fix:** switched to `HOUGH_GRADIENT_ALT`, which only accepts near-perfect
  circles. Coin counts became almost right, but some coins got two circles
  (an inner ring and the outer edge).
- **Fix:** removed duplicate circles whose centers are inside a bigger circle.
- **Problem:** a fixed "redness" cutoff couldn't tell copper from silver in
  every photo, because the lighting is different (the silver coins in coins2
  look yellowish). **Fix:** split each photo's coins at the biggest gap in
  redness.
- **Problem:** in coins2 the nickel was called a dime. The photo is taken at
  an angle, so the nickel measured the same size as the penny. **Fix:**
  measured the coin's longest width with an ellipse fit (a tilted coin is
  squashed in one direction only), and found one scale (pixels per mm) that
  fits all coins together instead of comparing to the pennies alone.
- **Problem:** one dark penny in coins4 was missed. **Fix:** a second, looser
  search that only keeps new circles that don't overlap a found coin, are a
  normal coin size, and have a strong rim.
- **Problem:** labels were too big and the summary box covered coins.
  **Fix:** text sized to fit each coin, and the box placed in the corner
  that covers the fewest coins.
- **Problem:** results changed when the images were resized, so the Canvas
  images could give different answers than the slide images. **Fix:** resize
  every photo to a standard size before searching, then draw on the original.
- **Problem:** after that, one nickel in coins4 touching other coins grew onto
  its neighbor's edge and was called a quarter. **Fix:** an "outer edge" only
  counts if it goes most of the way around the coin.
- **Final check:** all four images give the same answers at 0.6×, 1×, 2× and
  3× size.

## How the code works

1. **Resize.** The photo is resized so its longest side is 800 pixels, so
   the settings work the same on any image size.
2. **Find circles.** The image is turned gray and blurred, then
   `cv2.HoughCircles` with `HOUGH_GRADIENT_ALT` finds round shapes.
   Duplicates and tiny circles are removed, and a second looser pass picks up
   coins the first pass missed.
3. **Refine each circle.** The program measures how strong the edge is all
   around a circle and nudges the center and radius until the circle sits
   exactly on the coin's rim.
4. **Copper or silver.** For each coin it measures how red the middle is,
   (R − B) / (R + G + B). The coins are sorted by redness and split at the
   biggest gap: the red group are pennies, the rest are silver.
5. **Size → coin type.** Each coin's true size is the longest width of an
   ellipse fitted to its rim. The program tries thousands of scales (pixels
   per mm) and keeps the one where every coin best matches a real diameter
   (dime 17.91 mm, penny 19.05 mm, nickel 21.21 mm, quarter 24.26 mm). Each
   silver coin then gets the closest of dime, nickel, or quarter.
6. **Draw.** Each coin is circled in its color and labeled. A box in the
   emptiest corner shows the counts and total, and the result is saved as
   `coinsN.annotated.png`.