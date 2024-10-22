import cv2
import os
from skimage.feature import hog
from skimage import exposure
from config import PATCH_WIDTH, PATCH_HEIGHT

def compute_hog(image):
    hog_features, hog_image = hog(image, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), visualize=True,
                                  channel_axis=-1)
    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return hog_features

def extract_patches_and_compute_hog(annotations, images_dir):
    feature_vectors = []
    for img_id, x0, y0, x1, y1 in annotations:
        img_path = os.path.join(images_dir, f'{img_id}.png')
        if not os.path.exists(img_path):
            print(f'Image {img_path} not found!')
            continue
        image = cv2.imread(img_path)
        if image is None:
            print(f'Failed to read image {img_path}.')
            continue
        height, width = image.shape[:2]
        if y0 < 0 or x0 < 0 or y1 > height or x1 > width:
            print(f'Coordinates are out of bounds for image {img_id}.')
            continue
        patch = image[y0:y1, x0:x1]
        if patch.shape[0] != PATCH_HEIGHT or patch.shape[1] != PATCH_WIDTH:
            print(f'Incorrect patch size for image {img_id}: {patch.shape}')
            continue
        hog_features = compute_hog(patch)
        feature_vectors.append(hog_features)
    return feature_vectors
