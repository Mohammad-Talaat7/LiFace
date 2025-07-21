import os

import cv2
from tqdm import tqdm

from .utils.logger import setup_logger
from .utils.preprocessing_utils import (
    align_haar_face,
    detect_faces_coordinates,
    return_detector,
)

logger = setup_logger(name="preprocessing", log_file="preprocessing_main.log")


class FacePreprocessor:
    """Class used for preprocessing images to get faces."""

    def __init__(
        self,
        detector_type="haar",
        device="cpu",
        cnn_model_path=None,
        cascade_model_path=None,
    ):
        logger.info(
            "Initializing FacePreprocessor Class with detector type: %s",
            detector_type,
        )

        self.detector_type = detector_type.lower()
        self.device = device

        if self.detector_type == "haar":
            self.detector, self.eye_cascade = return_detector( # type: ignore
                detector_type="haar",
                cascade_path=cascade_model_path,
                device=device,
            )
        else:
            self.detector = return_detector(
                detector_type, cnn_model_path=cnn_model_path, device=device
            )

    def detect_faces(self, image):
        """Detecting faces in an image."""
        logger.info("Detecting faces in image using %s", type(self.detector))
        faces = detect_faces_coordinates(
            detector_type=self.detector_type,
            detector=self.detector,
            image=image,
        )
        logger.info("Number of faces found in the image: %s", faces.size)
        return faces

    def align_face(self, image, face_rect):
        """Aligning faces"""
        x, y, w, h = face_rect
        face = image[y : y + h, x : x + w]

        if self.detector_type == "haar":
            tmp_face = align_haar_face(
                eye_cascade=self.eye_cascade,
                face=face,
                width=int(w),
                height=int(h),
            )

            if tmp_face is not None:
                face = tmp_face
                del tmp_face
        logger.info(
            "Returning aligned face with length %s with type %s.",
            face.size,
            type(face),
        )
        return face

    def process_directory(self, input_dir, output_dir, target_size=(112, 112)):
        """Process an input directory and store the outputs in an output directory."""
        os.makedirs(output_dir, exist_ok=True)
        logger.info(
            "Initialized output directory successfully, preprocessing the input directory."
        )

        for root, _, files in os.walk(input_dir):
            for file in tqdm(
                files, desc=f"Processing {os.path.basename(input_dir)}"
            ):
                if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                    logger.warning("File %s have non-valid extention.", file)
                    continue

                logger.info("Processing file: %s", file)
                img_path = os.path.join(root, file)
                image = cv2.imread(img_path)
                if image is None:
                    logger.warning("Can't read provided image within %s", file)
                    continue

                faces = self.detect_faces(image)
                if faces.size == 0:
                    logger.warning(
                        "Image provided within %s have zero faces in it.", file
                    )
                    continue

                aligned = self.align_face(image, faces[0])

                if aligned.size == 0:
                    logger.warning(
                        "Aligned Image recivied %s equal to zero.", aligned
                    )
                    continue

                resized = cv2.resize(aligned, target_size)

                rel_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, rel_path)
                os.makedirs(output_subdir, exist_ok=True)

                output_path = os.path.join(output_subdir, file)
                logger.info("Writing resized image to %s", output_path)
                cv2.imwrite(output_path, resized)


if __name__ == "__main__":
    processor = FacePreprocessor(detector_type="mtcnn")
    processor.process_directory(
        input_dir="/home/mohammad/Projects/LiFace/data/test",
        output_dir="/home/mohammad/Projects/LiFace/data/mtcnn",
    )
