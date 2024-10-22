import random
import os
import cv2
from feature_extraction import compute_hog
from config import PATCH_WIDTH, PATCH_HEIGHT, NEGATIVE_SAMPLE_STEP

def generate_random_negative_samples(annotations, images_dir, num_samples):
    negative_feature_vectors = []
    for _ in range(num_samples):
        img_id = random.choice(range(1, 251))
        img_path = os.path.join(images_dir, f'{img_id}.png')
        if os.path.exists(img_path):
            image = cv2.imread(img_path)
            height, width = image.shape[:2]
            valid_patch = False
            while not valid_patch:
                x0 = random.randint(0, width - PATCH_WIDTH)
                y0 = random.randint(0, height - PATCH_HEIGHT)
                x1 = x0 + PATCH_WIDTH
                y1 = y0 + PATCH_HEIGHT
                valid_patch = not any(
                    y0 < ann[3] and y1 > ann[1] and x0 < ann[4] and x1 > ann[2]
                    for ann in annotations
                )
                if valid_patch:
                    patch = image[y0:y1, x0:x1]
                    if patch.shape[0] == PATCH_HEIGHT and patch.shape[1] == PATCH_WIDTH:
                        hog_features = compute_hog(patch)
                        negative_feature_vectors.append(hog_features)
    return negative_feature_vectors
