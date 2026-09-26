"""
ai-panorama.py
Based on the OpenCV stitching.py sample, modified to stitch more images.

Engines:
  detailed (default)  The stitching pipeline written out step by step, with
                  each image matched only against its neighbours.
  simple          cv.Stitcher, as in the original sample.

Methods:
  search (default) Try every run of neighbouring raw images, longest first,
                  and keep the longest run that stitches with every image used.
  grow  Find the strongest neighbouring pair of raw images, then add
                  one neighbour at a time (left or right), re-stitching the RAW
                  images each time so OpenCV can adjust all cameras together.
  hierarchical    Stitch pairs, then stitch the partial panoramas together.
                  (Earlier version. Feeding warped panoramas back into the
                  stitcher tends to produce distorted results.)
"""
from __future__ import print_function

import numpy as np
import cv2 as cv

import argparse
import glob
import os
import sys

# Run on the CPU. OpenCL (GPU) can crash with CL_OUT_OF_RESOURCES.
cv.ocl.setUseOpenCL(False)

modes = (cv.Stitcher_PANORAMA, cv.Stitcher_SCANS)

# Confidence thresholds to try, strict to loose (OpenCV default is 1.0).
# Below about 0.5 the stitcher starts accepting wrong matches and the
# result gets warped, so --min-conf stops the search there by default.
ALL_THRESHOLDS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

parser = argparse.ArgumentParser(prog='ai-panorama.py', description='Stitch as many images as possible.')
parser.add_argument('--mode',
    type = int, choices = modes, default = cv.Stitcher_PANORAMA,
    help = 'Determines configuration of stitcher. The default is `PANORAMA` (%d), '
        'mode suitable for creating photo panoramas. Option `SCANS` (%d) is suitable '
        'for stitching materials under affine transformation, such as scans.' % modes)
parser.add_argument('--output', default = 'ai-panorama.jpg',
    help = 'Resulting image. The default is `ai-panorama.jpg`.')
parser.add_argument('--method', choices = ('search', 'grow', 'hierarchical'), default = 'search',
    help = 'Stitching strategy. Default `search`.')
parser.add_argument('--engine', choices = ('detailed', 'simple'), default = 'detailed',
    help = '`detailed` builds the pipeline step by step and only matches '
        'neighbouring images; `simple` is cv.Stitcher. Default `detailed`.')
parser.add_argument('--range', type = int, default = 1,
    help = 'detailed engine: how many neighbours on each side an image may '
        'match with. Default 1 (next-door images only).')
parser.add_argument('--no-wave', action = 'store_true',
    help = 'Turn off wave correction (straightening of the horizon).')
parser.add_argument('--min-conf', type = float, default = 0.5,
    help = 'Lowest confidence threshold to try. Default 0.5.')
parser.add_argument('--resol', type = float, default = 0.8,
    help = 'Registration resolution in megapixels (OpenCV default 0.6).')
parser.add_argument('--max-width', type = int, default = 1200,
    help = 'Shrink input images to this width before stitching. Default 1200.')
parser.add_argument('--max-images', type = int, default = None,
    help = 'Only use the first N images (e.g. 7).')
parser.add_argument('--stages', default = 'stages',
    help = 'Folder to save intermediate results. Default `stages`.')
parser.add_argument('img', nargs='+', help = 'input images (wildcards like *.png are OK)')


def resize_to_width(img, max_width):
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / float(w)
    return cv.resize(img, (int(w * scale), int(h * scale)), interpolation = cv.INTER_AREA)


def crop_black_border(img):
    """Trim the solid black margins the stitcher leaves around a result."""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    coords = cv.findNonZero((gray > 0).astype(np.uint8))
    if coords is None:
        return img
    x, y, w, h = cv.boundingRect(coords)
    return img[y:y + h, x:x + w]


STATUS_TEXT = {
    1: "an image didn't match well enough, so it was dropped",
    2: "homography estimation failed",
    3: "camera parameter adjustment failed",
}


def detailed_stitch(imgs, conf, args):
    """The stitching pipeline built step by step (based on OpenCV's
    stitching_detailed.py sample). The key difference from cv.Stitcher:
    each image is only matched with its neighbours in the sequence, so a
    car or window in image 1 can't be mistaken for one in image 6.
    Returns (status, panorama, number_of_images_used)."""
    # 1. Find features (SIFT copes with the low-detail grass/sky shots)
    finder = cv.SIFT_create()
    features = [cv.detail.computeImageFeatures2(finder, img) for img in imgs]

    # 2. Match each image only with its neighbours
    matcher = cv.detail_BestOf2NearestRangeMatcher(args.range, False, 0.65)
    matches = matcher.apply2(features)
    matcher.collectGarbage()

    # 3. Keep the biggest group of images that are connected with confidence >= conf
    keep = cv.detail.leaveBiggestComponent(features, matches, conf)
    keep = [int(k) for k in np.array(keep).flatten()]
    if len(keep) < len(imgs):
        return 1, None, len(keep)

    # 4. Estimate a camera (rotation + focal length) for each image, then refine
    ok, cameras = cv.detail_HomographyBasedEstimator().apply(features, matches, None)
    if not ok:
        return 2, None, len(imgs)
    for cam in cameras:
        cam.R = cam.R.astype(np.float32)
    adjuster = cv.detail_BundleAdjusterRay()
    adjuster.setConfThresh(conf)
    adjuster.setRefinementMask(np.ones((3, 3), np.uint8))
    ok, cameras = adjuster.apply(features, matches, cameras)
    if not ok:
        return 3, None, len(imgs)

    # 5. Straighten the horizon
    if not args.no_wave:
        rmats = cv.detail.waveCorrect([np.copy(c.R) for c in cameras], cv.detail.WAVE_CORRECT_HORIZ)
        for cam, r in zip(cameras, rmats):
            cam.R = r

    # 6. Warp every image onto a sphere
    scale = float(np.median([c.focal for c in cameras]))
    warper = cv.PyRotationWarper('spherical', scale)
    corners, warped, masks = [], [], []
    for img, cam in zip(imgs, cameras):
        K = cam.K().astype(np.float32)
        corner, img_w = warper.warp(img, K, cam.R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
        _, mask_w = warper.warp(255 * np.ones(img.shape[:2], np.uint8), K, cam.R,
                                cv.INTER_NEAREST, cv.BORDER_CONSTANT)
        corners.append(corner)
        warped.append(img_w)
        masks.append(mask_w)

    # 7. Even out brightness, find good seams, and blend
    compensator = cv.detail.ExposureCompensator_createDefault(cv.detail.ExposureCompensator_GAIN_BLOCKS)
    compensator.feed(corners = corners, images = warped, masks = masks)
    for i in range(len(warped)):
        compensator.apply(i, corners[i], warped[i], masks[i])
    seam_finder = cv.detail_GraphCutSeamFinder('COST_COLOR')
    masks = seam_finder.find([w.astype(np.float32) for w in warped], corners, masks)

    sizes = [(w.shape[1], w.shape[0]) for w in warped]
    roi = cv.detail.resultRoi(corners = corners, sizes = sizes)
    blender = cv.detail_MultiBandBlender()
    bands = max(1, int(np.log2(max(1.0, np.sqrt(roi[2] * roi[3]) * 0.05)) - 1))
    blender.setNumBands(bands)
    blender.prepare(roi)
    for w, m, c in zip(warped, masks, corners):
        blender.feed(cv.UMat(w.astype(np.int16)), m, c)
    pano, _ = blender.blend(None, None)
    pano = cv.convertScaleAbs(pano)
    return cv.Stitcher_OK, pano, len(imgs)


def stitch_list(imgs, args, label, min_width = 0):
    """Stitch a list of images, loosening the confidence threshold until it
    works with EVERY image used. Returns (panorama, threshold) or (None, None)."""
    thresholds = [t for t in ALL_THRESHOLDS if t >= args.min_conf - 1e-9]
    reason = "unknown"
    for thresh in thresholds:
        if args.engine == 'detailed':
            try:
                status, pano, used = detailed_stitch(imgs, thresh, args)
            except cv.error as e:
                reason = str(e).strip().splitlines()[-1]
                continue
            if status != cv.Stitcher_OK:
                reason = STATUS_TEXT.get(status, "error code %d" % status)
                if status == 1:
                    reason = "only %d of %d images connected" % (used, len(imgs))
                continue
            if pano.shape[1] <= min_width:
                reason = "result did not get wider"
                continue
            print("  OK   %s  (confidence threshold %.1f)" % (label, thresh))
            return crop_black_border(pano), thresh

        stitcher = cv.Stitcher.create(args.mode)
        stitcher.setPanoConfidenceThresh(thresh)
        stitcher.setRegistrationResol(args.resol)
        if args.no_wave:
            stitcher.setWaveCorrection(False)
        try:
            status, pano = stitcher.stitch(imgs)
        except cv.error as e:
            reason = str(e).strip().splitlines()[-1]
            continue
        if status != cv.Stitcher_OK:
            reason = STATUS_TEXT.get(status, "error code %d" % status)
            continue
        # component() lists which input images made it into the result
        used = len(stitcher.component())
        if used < len(imgs):
            reason = "only %d of %d images were used" % (used, len(imgs))
            continue
        if pano.shape[1] <= min_width:
            reason = "result did not get wider"
            continue
        print("  OK   %s  (confidence threshold %.1f)" % (label, thresh))
        return crop_black_border(pano), thresh
    print("  FAIL %s  (%s)" % (label, reason))
    return None, None


def search_stitch(pieces, args):
    """Try runs of neighbouring images, longest first. The first length where
    any run succeeds wins; among those, keep the most confident one."""
    names = [n for n, _ in pieces]
    imgs = [im for _, im in pieces]
    n = len(imgs)
    for length in range(n, 1, -1):
        print("\nTrying runs of %d images:" % length)
        found = []
        for start in range(0, n - length + 1):
            end = start + length - 1
            label = "images %d-%d" % (start + 1, end + 1)
            pano, thresh = stitch_list(imgs[start:end + 1], args, label)
            if pano is not None:
                cv.imwrite(os.path.join(args.stages, "search_%d-%d.jpg" % (start + 1, end + 1)), pano)
                found.append((thresh, start, end, pano))
        if found:
            found.sort(key = lambda f: f[0], reverse = True)
            _, start, end, pano = found[0]
            return pano, names[start:end + 1]
    print("No run of images could be stitched.")
    return None, []


def grow_stitch(pieces, args):
    names = [n for n, _ in pieces]
    imgs = [im for _, im in pieces]
    n = len(imgs)
    one_w = max(im.shape[1] for im in imgs)

    # 1. Test every neighbouring pair; start from the most confident one
    print("\nTesting neighbouring pairs:")
    best = None
    for i in range(n - 1):
        pano, thresh = stitch_list([imgs[i], imgs[i + 1]], args,
                                   "[%s] + [%s]" % (names[i], names[i + 1]), 1.05 * one_w)
        if pano is not None and (best is None or thresh > best[2]):
            best = (i, i + 1, thresh, pano)
    if best is None:
        print("No pair of images could be stitched.")
        return None, []

    lo, hi, _, pano = best
    print("\nStarting from %s + %s" % (names[lo], names[hi]))
    cv.imwrite(os.path.join(args.stages, "grow_%d-%d.jpg" % (lo + 1, hi + 1)), pano)

    # 2. Add one neighbouring raw image at a time, re-stitching the whole run
    while True:
        candidates = []
        if lo > 0:
            candidates.append((lo - 1, hi))
        if hi < n - 1:
            candidates.append((lo, hi + 1))
        results = []
        for a, b in candidates:
            new_img = names[a] if a < lo else names[b]
            label = "images %d-%d (adding %s)" % (a + 1, b + 1, new_img)
            p, t = stitch_list(imgs[a:b + 1], args, label, 1.03 * pano.shape[1])
            if p is not None:
                results.append((t, a, b, p))
        if not results:
            break
        # Keep whichever side joined most confidently
        results.sort(key = lambda r: r[0], reverse = True)
        _, lo, hi, pano = results[0]
        pano = resize_to_width(pano, args.max_width * 4)
        cv.imwrite(os.path.join(args.stages, "grow_%d-%d.jpg" % (lo + 1, hi + 1)), pano)

    return pano, names[lo:hi + 1]


def hierarchical_stitch(pieces, args):
    level = 1
    while len(pieces) > 1:
        print("\nLevel %d: %d pieces" % (level, len(pieces)))
        merged, i, progress = [], 0, False
        while i < len(pieces):
            if i + 1 < len(pieces):
                (name_a, img_a), (name_b, img_b) = pieces[i], pieces[i + 1]
                pano, _ = stitch_list([img_a, img_b], args, "[%s] + [%s]" % (name_a, name_b),
                                      1.05 * max(img_a.shape[1], img_b.shape[1]))
                if pano is not None:
                    pano = resize_to_width(pano, args.max_width * 3)
                    new_name = name_a + "+" + name_b
                    cv.imwrite(os.path.join(args.stages, "L%d_%s.jpg" % (level, new_name)), pano)
                    merged.append((new_name, pano))
                    i += 2
                    progress = True
                    continue
            merged.append(pieces[i])
            i += 1
        pieces = merged
        if not progress:
            print("No more merges possible.")
            break
        level += 1
    best_name, pano = max(pieces, key = lambda p: p[1].shape[1])
    return pano, best_name.split("+")


def main():
    args = parser.parse_args()

    # Expand wildcards ourselves (Windows terminals don't do it)
    names = []
    for pattern in args.img:
        matches = sorted(glob.glob(pattern))
        names.extend(matches if matches else [pattern])
    if args.max_images:
        names = names[:args.max_images]

    pieces = []
    for img_name in names:
        img = cv.imread(img_name)
        if img is None:
            print("can't read image " + img_name)
            sys.exit(-1)
        name = os.path.splitext(os.path.basename(img_name))[0]
        pieces.append((name, resize_to_width(img, args.max_width)))
        print("loaded " + name)

    if len(pieces) < 2:
        print("Need at least 2 images.")
        sys.exit(-1)
    if not os.path.isdir(args.stages):
        os.makedirs(args.stages)

    if args.method == 'search':
        pano, used = search_stitch(pieces, args)
    elif args.method == 'grow':
        pano, used = grow_stitch(pieces, args)
    else:
        pano, used = hierarchical_stitch(pieces, args)

    if pano is None:
        sys.exit(-1)
    cv.imwrite(args.output, pano)
    print("\nstitching completed. %s saved!" % args.output)
    print("Panorama contains %d image(s): %s" % (len(used), ", ".join(used)))
    print('Done')

if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()