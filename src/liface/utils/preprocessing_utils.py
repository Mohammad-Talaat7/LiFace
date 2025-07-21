import os
from pathlib import Path

import cv2
import dlib
import numpy as np

from .logger import setup_logger
from .mtcnn import MTCNN

logger = setup_logger("preprocessing_utils", "preprocessing_utils.log")
logger.info("Initialized preprocessor log file successfully!")
MODEL_DIR = Path(__file__).resolve().parents[3] / "models"


def return_detector(
    detector_type: str,
    cnn_model_path=None,
    cascade_path=None,
    device="cpu",
):
    """Returning detector object of the given type."""
    logger.info("Trying to initialize detector type: %s", detector_type)

    if detector_type == "haar":

        if cascade_path is not None:
            cascade_model_path = MODEL_DIR / cascade_path
        else:
            logger.warning(
                "Failed to initialize the detector HAAR with no path to the cascade model."
            )
            raise ValueError(
                "You must provide a valid cascade_path for HAAR detector."
            )

        # Check if the file exists
        if not os.path.exists(cascade_model_path):
            logger.warning("haar model cannot be found at given paths.")
            raise FileNotFoundError("haar model not found.")

        logger.info("Initializaing HAAR detector, using path: %s", cascade_model_path)

        detector = cv2.CascadeClassifier(cascade_model_path)
        eye_cascade = cv2.CascadeClassifier(cascade_model_path)
        logger.info("Returning the detector and the eye cascade")
        return detector, eye_cascade

    if detector_type == "mtcnn":
        logger.info("Initializaing MTCNN detector")
        detector = MTCNN(config={"keep_all": True}, device=device)
        logger.info("Returning the detector")
        return detector

    if detector_type == "dlib-hog":
        logger.info("Initializaing the dlib-hog detector")
        detector = dlib.get_frontal_face_detector()  # type: ignore # pylint: disable=no-member
        logger.info("Returning the detector")
        return detector

    if detector_type == "dlib-cnn":
        logger.info("Initializaing dlib-cnn detector")

        if cnn_model_path is not None:
            cnn_model_path = MODEL_DIR / cnn_model_path
            if not os.path.isfile(cnn_model_path):
                logger.error("Didn't receive a valid cnn_model_path")
                raise ValueError(
                    "You must provide a valid cnn_model_path for Dlib CNN detector."
                )
        else:
            logger.error("Didn't receive a valid cnn_model_path")
            raise ValueError(
                "You must provide a valid cnn_model_path for Dlib CNN detector."
            )

        detector = dlib.cnn_face_detection_model_v1(cnn_model_path)  # type: ignore # pylint: disable=no-member
        logger.info("Returning the detector")
        return detector

    logger.error("Unsupported detector type: %s", detector_type)
    raise ValueError(
        "Unsupported detector type. Choose from 'haar', 'mtcnn', 'dlib-hog', 'dlib-cnn'."
    )


def detect_faces_coordinates(detector_type, detector, image):
    """Detecting faces in image and returning its coordinates."""
    logger.info("using detector type: %s", type(detector))
    if detector_type == "haar":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5
        )

        # Ensure faces is always a numpy array
        if isinstance(faces, tuple):
            faces = np.array([]).reshape(0, 4)  # Empty array with shape (0, 4)

        logger.info(
            "Found (%s) faces in the provided image using HAAR detector.",
            len(faces),
        )
        logger.info("Returning in datatype: %s", type(faces))
        return faces

    if detector_type == "mtcnn":
        boxes,_,_ = detector.detect(image)

        faces = np.array(
            [
                [int(x), int(y), int(x2 - x), int(y2 - y)]
                for (x, y, x2, y2) in boxes
            ]
            if boxes is not None
            else []
        )

        logger.info(
            "Found (%s) faces in the provided image using MTCNN detector.",
            len(faces),
        )
        logger.info("Returning in datatype: %s", type(faces))
        return faces

    if detector_type == "dlib-hog":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        faces_edges = np.array(
            [[d.left(), d.top(), d.width(), d.height()] for d in faces]
        )

        logger.info(
            "Found (%s) faces in the provided image using Dlib-HOG detector.",
            len(faces),
        )
        logger.info("Returning in datatype: %s", type(faces_edges))
        return faces_edges

    if detector_type == "dlib-cnn":
        faces = detector(image, 1)

        faces_edges = np.array(
            [
                [d.rect.left(), d.rect.top(), d.rect.width(), d.rect.height()]
                for d in faces
            ]
        )
        logger.info(
            "Found (%s) faces in the provided image using Dlib-CNN detector.",
            len(faces),
        )
        logger.info("Returning in datatype: %s", type(faces_edges))
        return faces_edges

    logger.error("Unsupported detector type: %s", detector_type)
    raise ValueError(
        "Unsupported detector type. Choose from 'haar', 'mtcnn', 'dlib-hog', 'dlib-cnn'."
    )


def align_haar_face(eye_cascade, face, width, height):
    """Aligning faces using Eye Cascade in MultiScale."""
    if width <= 0 or height <= 0:
        logger.warning(
            "Invalid image dimensions: %s (width), %s (height)", width, height
        )
        raise ValueError(
            f"Invalid image dimensions: {width} (width), {height} (height)"
        )

    eyes = eye_cascade.detectMultiScale(face)
    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[0])
        centers = [((x + w // 2), (y + h // 2)) for (x, y, w, h) in eyes]
        angle = np.degrees(
            np.arctan2(
                centers[1][1] - centers[0][1], centers[1][0] - centers[0][0]
            )
        )
        rotation_matrix = cv2.getRotationMatrix2D(
            (width // 2, height // 2), angle, 1.0
        )

        return cv2.warpAffine(
            face, rotation_matrix, (width, height), flags=cv2.INTER_CUBIC
        )
    return None
