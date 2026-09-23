import cv2 as cv
import os

photos_dir = "photos"
output_dir = "resolution"
target_width = 640

for filename in os.listdir(photos_dir):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    filepath = os.path.join(photos_dir, filename)
    img = cv.imread(filepath)

    if img is None:
        print(f"Could not read {filepath}, skipping")
        continue

    original_height, original_width = img.shape[:2]

    scale = original_width / target_width
    new_height = round(original_height / scale)

    resized_img = cv.resize(img, (target_width, new_height), interpolation=cv.INTER_LINEAR)

    name_without_ext = os.path.splitext(filename)[0]
    new_filename = f"{name_without_ext}-640x{new_height}.png"
    new_filepath = os.path.join(output_dir, new_filename)

    cv.imwrite(new_filepath, resized_img)
    print(f"Saved {new_filepath}")