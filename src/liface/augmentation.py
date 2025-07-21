import os
import random

import cv2
import numpy as np
from tqdm import tqdm

from liface import configurations as Config  # type: ignore


class FaceAugmenter:
    """Class for augmenting face images."""

    def __init__(
        self,
        seed=Config.RANDOM_SEED,
        enabled_augmentations=Config.ENABLED_AUGMENTATIONS,
    ):
        """
        Initialize the augmenter with a list of enabled augmentations.
        If None, all augmentations are used.

        Args:
            seed (int): the seed wanted to use for reproducibility.
                ({default: None})
            enabled_augmentations (list[str]): List of augmentation method names to enable.
                ({default: None})
        """
        if seed:
            random.seed(seed)

        self.all_augmentations = {
            "random_flip": self.random_flip,
            "random_rotation": self.random_rotation,
            "random_brightness": self.random_brightness,
            "random_contrast": self.random_contrast,
            "random_blur": self.random_blur,
            "add_noise": self.add_noise,
            "apply_clahe": self.apply_clahe,
        }

        if enabled_augmentations is None:
            self.augmentations = list(self.all_augmentations.values())
        else:
            self.augmentations = [
                self.all_augmentations[name]
                for name in enabled_augmentations
                if name in self.all_augmentations
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

    def add_noise(self, image):
        """Add random Gaussian noise to image."""
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        return cv2.add(image, noise)

    def apply_clahe(self, image):
        """Apply CLAHE to improve contrast in low-light images."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def augment_image(
        self, image, n_augmentations=Config.NUM_AUGMENTATIONS_PER_IMAGE
    ):
        """Apply random augmentations to an image"""
        augmentations = random.sample(self.augmentations, n_augmentations)
        for aug in augmentations:
            image = aug(image)
        return image

    def augment_dataset(
        self,
        input_dir,
        output_dir,
        n_augmentations=Config.NUM_AUGMENTATIONS_PER_DATASET,
    ):
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
