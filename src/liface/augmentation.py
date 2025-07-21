import os
import random

import cv2
import numpy as np
from tqdm import tqdm


class FaceAugmenter:
    """Class for augmenting face images."""

    def __init__(self):
        self.augmentations = [
            self.random_flip,
            self.random_rotation,
            self.random_brightness,
            self.random_contrast,
            self.random_blur,
            self.random_crop,
        ]

    def random_flip(self, image):
        """Flipping images randomally."""
        if random.random() > 0.5:
            return cv2.flip(image, 1)
        return image

    def random_rotation(self, image):
        """Rotate images randomally."""
        angle = random.uniform(-15, 15)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC
        )

    def random_brightness(self, image):
        """Adjust brightness in image randomally."""
        value = random.uniform(-30, 30)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + value, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def random_contrast(self, image):
        """Adjust contrast in image randomally."""
        alpha = random.uniform(0.8, 1.2)
        return cv2.convertScaleAbs(image, alpha=alpha)

    def random_blur(self, image):
        """Adding blur effect in image randomally."""
        if random.random() > 0.7:
            ksize = random.choice([3, 5])
            return cv2.GaussianBlur(image, (ksize, ksize), 0)
        return image

    def random_crop(self, image):
        """Cropping an image randomally."""
        if random.random() > 0.5:
            h, w = image.shape[:2]
            scale = random.uniform(0.85, 0.95)
            new_h, new_w = int(h * scale), int(w * scale)
            y = random.randint(0, h - new_h)
            x = random.randint(0, w - new_w)
            cropped = image[y : y + new_h, x : x + new_w]
            return cv2.resize(cropped, (w, h))
        return image

    def augment_image(self, image, n_augmentations=3):
        """Apply random augmentations to an image"""
        augmentations = random.sample(self.augmentations, n_augmentations)
        for aug in augmentations:
            image = aug(image)
        return image

    def augment_dataset(self, input_dir, output_dir, n_augmentations=5):
        """Create augmented versions of a dataset"""
        os.makedirs(output_dir, exist_ok=True)

        for root, _, files in os.walk(input_dir):
            for file in tqdm(
                files, desc=f"Augmenting {os.path.basename(input_dir)}"
            ):
                if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                image = cv2.imread(os.path.join(root, file))
                if image is None:
                    continue

                # Create augmented versions
                for i in range(n_augmentations):
                    augmented = self.augment_image(image)

                    # Save augmented image
                    rel_path = os.path.relpath(root, input_dir)
                    output_subdir = os.path.join(output_dir, rel_path)
                    os.makedirs(output_subdir, exist_ok=True)

                    filename, ext = os.path.splitext(file)
                    output_path = os.path.join(
                        output_subdir, f"{filename}_aug{i}{ext}"
                    )
                    cv2.imwrite(output_path, augmented)
