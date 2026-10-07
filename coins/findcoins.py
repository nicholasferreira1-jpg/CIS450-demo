"""
findcoins.py - Find, identify, and count US coins in an image.

Usage:
    python findcoins.py coins1.png            # one image
    python findcoins.py coins1.png coins2.png # several images
    python findcoins.py                       # all coins*.png in this folder

For each input image coinsN.png it writes coinsN.annotated.png with a circle
around every coin (Pennies = Red, Nickels = Green, Dimes = Blue,
Quarters = Yellow) and the coin counts and total value printed on the image.
"""

import glob
import os
import sys

import cv2
import numpy as np

# Real US coin diameters in millimeters, and their values in cents.
COINS = {
    "penny":   {"mm": 19.05, "cents": 1},
    "nickel":  {"mm": 21.21, "cents": 5},
    "dime":    {"mm": 17.91, "cents": 10},
    "quarter": {"mm": 24.26, "cents": 25},
}

# Annotation colors (OpenCV uses BGR order, not RGB).
COLORS = {
    "penny":   (0, 0, 255),     # red
    "nickel":  (0, 200, 0),     # green
    "dime":    (255, 0, 0),     # blue
    "quarter": (0, 255, 255),   # yellow
}

# Every image is resized so its longest side is this many pixels before
# searching for coins (the results are drawn on the original image).
WORK_SIZE = 800

ANGLES = np.linspace(0, 2 * np.pi, 120, endpoint=False)


# ---------------------------------------------------------------------------
# Step 1: find the circles
# ---------------------------------------------------------------------------

def edge_profile(gx, gy, x, y, r):
    """Edge strength at 120 points around the circle (x, y, r)."""
    px = np.clip((x + r * np.cos(ANGLES)).astype(int), 0, gx.shape[1] - 1)
    py = np.clip((y + r * np.sin(ANGLES)).astype(int), 0, gx.shape[0] - 1)
    return np.abs(gx[py, px] * np.cos(ANGLES) + gy[py, px] * np.sin(ANGLES))


def edge_score(gx, gy, x, y, r):
    """How strongly the image has an edge exactly along the circle (x, y, r).

    For points around the circle, we measure the brightness change pointing
    straight out from the center. A circle that sits right on a coin's rim
    gets a high score.
    """
    px = np.clip((x + r * np.cos(ANGLES)).astype(int), 0, gx.shape[1] - 1)
    py = np.clip((y + r * np.sin(ANGLES)).astype(int), 0, gx.shape[0] - 1)
    radial = gx[py, px] * np.cos(ANGLES) + gy[py, px] * np.sin(ANGLES)
    return float(np.abs(radial).mean())


def best_circle(gx, gy, x, y, r, rmin, rmax):
    """Nudge a circle's center and radius to the position with the best edge
    score, searching coarse-to-fine. Radius stays within [rmin, rmax]."""
    best = (edge_score(gx, gy, x, y, r), x, y, r)
    for step in (0.06, 0.02, 0.007):
        offsets = np.linspace(-3, 3, 7) * step * r
        _, bx, by, br = best
        for dx in offsets:
            for dy in offsets:
                for dr in offsets:
                    rr = br + dr
                    if rr < rmin or rr > rmax:
                        continue
                    s = edge_score(gx, gy, bx + dx, by + dy, rr)
                    if s > best[0]:
                        best = (s, bx + dx, by + dy, rr)
    return best


def hough(gray, strictness):
    """Run OpenCV's circle finder. HOUGH_GRADIENT_ALT only accepts circles
    that are close to perfect (strictness is a 0-1 "roundness" score), so
    wood grain and the engravings on the coins are mostly ignored."""
    size = min(gray.shape)
    found = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT_ALT, dp=1.5, minDist=size * 0.03,
        param1=100, param2=strictness,
        minRadius=int(size * 0.02), maxRadius=int(size * 0.3))
    if found is None:
        return []
    return sorted((tuple(map(float, c)) for c in found[0]),
                  key=lambda c: -c[2])


def overlaps(circle, others, amount=0.6):
    x, y, r = circle
    return any(np.hypot(x - ox, y - oy) < amount * max(r, orr)
               for ox, oy, orr in others)


def find_coins(img):
    """Return a list of (x, y, radius) circles, one per coin."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.5)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)

    circles = hough(gray, 0.7)
    if not circles:
        return []

    # Remove duplicates: a smaller circle whose center is inside a bigger
    # one is usually the inner ring of the same coin.
    unique = []
    for x, y, r in circles:
        if all(np.hypot(x - ux, y - uy) > 0.6 * max(r, ur)
               for ux, uy, ur in unique):
            unique.append((x, y, r))

    # Remove tiny false detections (much smaller than a typical coin).
    median_r = np.median([r for _, _, r in unique])
    unique = [c for c in unique if c[2] > 0.6 * median_r]

    # Refine each circle so it sits exactly on the coin's outer edge.
    refined, scores = [], []
    for c in unique:
        x, y, r, score = refine(gx, gy, *c)
        refined.append((x, y, r))
        scores.append(score)

    # Second pass for coins the strict search missed (e.g. a dark penny
    # touching other coins). Search more loosely, but only keep a new circle
    # if it doesn't overlap a coin we already have, is a normal coin size for
    # this photo, and has a rim at least as strong as the weakest real coins.
    radii = [r for _, _, r in refined]
    lo, hi = 0.85 * min(radii), 1.15 * max(radii)
    min_score = 0.7 * np.median(scores)
    for c in hough(gray, 0.4):
        if overlaps(c, refined, 0.8):
            continue
        x, y, r, score = refine(gx, gy, *c)
        if (score >= min_score and lo <= r <= hi
                and not overlaps((x, y, r), refined, 0.8)):
            refined.append((x, y, r))
            scores.append(score)
    return refined


def refine(gx, gy, x, y, r):
    """Move a circle onto the coin's rim. Returns (x, y, r, edge score)."""
    score, x, y, r = best_circle(gx, gy, x, y, r, 0.8 * r, 1.2 * r)
    # Coins photographed at an angle show their thick rim. If there is a
    # reasonably strong edge slightly further out, use that outer edge.
    # The outer edge must go most of the way around the coin: where coins
    # touch, a neighbor's rim can look like an "outer edge" on one side only,
    # so we also compare the weakest 25% of points on each circle.
    outer = best_circle(gx, gy, x, y, 1.08 * r, 1.03 * r, 1.2 * r)
    weak_inner = np.percentile(edge_profile(gx, gy, x, y, r), 25)
    weak_outer = np.percentile(edge_profile(gx, gy, *outer[1:]), 25)
    if outer[0] >= 0.6 * score and weak_outer >= 0.5 * weak_inner:
        score, x, y, r = outer
    return x, y, r, score


def true_size(img, x, y, r):
    """Measure a coin's real size in pixels (half of its longest width).

    A coin photographed at an angle looks like an ellipse: it gets squashed
    in one direction, but its longest width still shows its true size. We
    find the rim at 180 angles around the coin and fit an ellipse to it.
    """
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 1.5)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    points = []
    radii = np.linspace(0.85 * r, 1.12 * r, 40)
    for a in np.linspace(0, 2 * np.pi, 180, endpoint=False):
        px = np.clip((x + radii * np.cos(a)).astype(int), 0, gray.shape[1] - 1)
        py = np.clip((y + radii * np.sin(a)).astype(int), 0, gray.shape[0] - 1)
        strength = np.abs(gx[py, px] * np.cos(a) + gy[py, px] * np.sin(a))
        best = radii[int(np.argmax(strength))]
        points.append((x + best * np.cos(a), y + best * np.sin(a)))
    _, (d1, d2), _ = cv2.fitEllipse(np.array(points, np.float32))
    return max(d1, d2) / 2


# ---------------------------------------------------------------------------
# Step 2: decide copper (penny) or silver (nickel, dime, quarter)
# ---------------------------------------------------------------------------

def redness(img, x, y, r):
    """Average (R - B) / (R + G + B) inside the central part of the coin.
    Copper pennies are much redder than silver coins."""
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(mask, (int(x), int(y)), int(r * 0.75), 255, -1)
    b, g, r_ = (img[..., i][mask > 0].astype(float).mean() for i in range(3))
    return (r_ - b) / (r_ + g + b + 1e-6)


def split_copper(values):
    """Return True/False (is copper) for each coin.

    Lighting changes from photo to photo (a silver coin under warm light can
    look as red as a penny under cool light), so instead of a fixed cutoff we
    sort the redness values and split at the biggest gap.
    """
    if len(values) == 1:
        return [values[0] > 0.12]
    order = sorted(values)
    gaps = [order[i + 1] - order[i] for i in range(len(order) - 1)]
    i = int(np.argmax(gaps))
    if gaps[i] < 0.05:                 # no clear gap: all coins are one kind
        cut = 0.12
        return [v > cut for v in values]
    cut = (order[i] + order[i + 1]) / 2
    return [v > cut for v in values]


# ---------------------------------------------------------------------------
# Step 3: tell silver coins apart by size
# ---------------------------------------------------------------------------

def classify(radii, is_copper):
    """Label every coin.

    Coins are compared to real diameters through one shared scale k
    (pixels per millimeter). We try many values of k and keep the one where
    every coin's measured size best matches a real coin of its color group
    (pennies must be pennies; silver coins can be dime, nickel, or quarter).
    """
    silver_types = ["dime", "nickel", "quarter"]
    log_r = np.log(radii)
    candidates = np.exp(np.linspace(np.log(min(radii) / 30),
                                    np.log(max(radii) / 5), 3000))
    best_k, best_cost = None, None
    for k in candidates:
        cost = 0.0
        for lr, copper in zip(log_r, is_copper):
            types = ["penny"] if copper else silver_types
            cost += min((lr - np.log(k * COINS[t]["mm"] / 2)) ** 2
                        for t in types)
        if best_cost is None or cost < best_cost:
            best_k, best_cost = k, cost

    labels = []
    for lr, copper in zip(log_r, is_copper):
        if copper:
            labels.append("penny")
        else:
            labels.append(min(silver_types,
                              key=lambda t: abs(lr - np.log(best_k * COINS[t]["mm"] / 2))))
    return labels


# ---------------------------------------------------------------------------
# Step 4: draw the results
# ---------------------------------------------------------------------------

def annotate(img, coins, labels):
    out = img.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    for (x, y, r), label in zip(coins, labels):
        color = COLORS[label]
        thick = max(2, int(r / 25))
        cv2.circle(out, (int(x), int(y)), int(r), color, thick)
        # Make the word fit inside the coin.
        scale = 1.0
        while scale > 0.3 and cv2.getTextSize(label, font, scale, 1)[0][0] > 1.5 * r:
            scale -= 0.05
        tthick = max(1, int(scale * 2))
        (tw, th), _ = cv2.getTextSize(label, font, scale, tthick)
        org = (int(x - tw / 2), int(y + th / 2))
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):   # dark outline
            cv2.putText(out, label, (org[0] + dx, org[1] + dy), font, scale,
                        (0, 0, 0), tthick, cv2.LINE_AA)
        cv2.putText(out, label, org, font, scale, color, tthick, cv2.LINE_AA)

    counts = {t: labels.count(t) for t in COINS}
    total = sum(counts[t] * COINS[t]["cents"] for t in COINS) / 100
    lines = [f"Pennies:  {counts['penny']}", f"Nickels:  {counts['nickel']}",
             f"Dimes:    {counts['dime']}", f"Quarters: {counts['quarter']}",
             f"Total: ${total:.2f}"]

    # Summary box: size it to the image, then put it in whichever corner
    # covers the fewest coins.
    scale = max(0.4, min(h, w) / 1000)
    tthick = max(1, int(scale * 2))
    line_h = int(34 * scale)
    box_w = max(cv2.getTextSize(t, font, scale, tthick)[0][0] for t in lines) + 20
    box_h = line_h * len(lines) + 14
    coin_mask = np.zeros((h, w), np.uint8)
    for x, y, r in coins:
        cv2.circle(coin_mask, (int(x), int(y)), int(r), 1, -1)
    corners = [(0, 0), (w - box_w, 0), (0, h - box_h), (w - box_w, h - box_h)]
    bx, by = min(corners, key=lambda c: coin_mask[c[1]:c[1] + box_h,
                                                  c[0]:c[0] + box_w].sum())
    overlay = out.copy()
    cv2.rectangle(overlay, (bx, by), (bx + box_w, by + box_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.7, out, 0.3, 0)
    for i, text in enumerate(lines):
        cv2.putText(out, text, (bx + 10, by + line_h * (i + 1)), font, scale,
                    (255, 255, 255), tthick, cv2.LINE_AA)
    return out, counts, total


# ---------------------------------------------------------------------------

def process(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Could not read {path}")
        return
    # Work on a standard-size copy so the blur and edge settings behave the
    # same no matter how big the original photo is.
    k = WORK_SIZE / max(img.shape[:2])
    work = cv2.resize(img, None, fx=k, fy=k,
                      interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
    coins = find_coins(work)
    if not coins:
        print(f"{path}: no coins found")
        return
    copper = split_copper([redness(work, *c) for c in coins])
    sizes = [true_size(work, *c) for c in coins]
    labels = classify(sizes, copper)

    # Scale the circles back up and draw on the original image.
    coins = [(x / k, y / k, r / k) for x, y, r in coins]
    out, counts, total = annotate(img, coins, labels)

    root, ext = os.path.splitext(path)
    out_path = f"{root}.annotated{ext}"
    cv2.imwrite(out_path, out)
    print(f"{os.path.basename(path)}: {counts['penny']} pennies, "
          f"{counts['nickel']} nickels, {counts['dime']} dimes, "
          f"{counts['quarter']} quarters = ${total:.2f}  ->  {out_path}")


if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        here = os.path.dirname(os.path.abspath(__file__))
        paths = sorted(p for p in glob.glob(os.path.join(here, "coins*.png"))
                       if ".annotated" not in p)
    for p in paths:
        process(p)