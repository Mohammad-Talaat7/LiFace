# pylint: disable=redefined-outer-name

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from liface.preprocessing import FacePreprocessor


@pytest.fixture(scope="module")
def test_face_image(tmp_path_factory):
    """Creates a test face image from Lena sample."""
    utils_path = Path(__file__).resolve().parent / "utils"
    lena = cv2.imread(str(utils_path / "lena.jpg"))
    if lena is None:
        pytest.skip("Sample image 'lena.jpg' not found.")
    tmp_dir = tmp_path_factory.mktemp("test_face_image")
    img_path = tmp_dir / "test_face.jpg"
    cv2.imwrite(str(img_path), lena)
    return str(img_path)


@pytest.mark.parametrize(
    "detector_type, cascade_model_path, cnn_model_path",
    [
        ("haar", "haarcascade_frontalface_default.xml", None),
        ("mtcnn", None, None),
        ("dlib-hog", None, None),
        ("dlib-cnn", None, "mmod_human_face_detector.dat"),
    ],
)
def test_face_preprocessor_with_all_detectors(
    detector_type,
    cascade_model_path,
    cnn_model_path,
    test_face_image,
    tmp_path,
):
    """Test full pipeline for Haar, MTCNN, Dlib-HOG, Dlib-CNN."""
    # Prepare model paths
    utils_path = Path(__file__).resolve().parent / "utils"
    cascade_path = (
        str(utils_path / cascade_model_path) if cascade_model_path else None
    )
    cnn_path = str(utils_path / cnn_model_path) if cnn_model_path else None

    # Initialize FacePreprocessor
    preprocessor = FacePreprocessor(
        detector_type=detector_type,
        cascade_model_path=cascade_path,
        cnn_model_path=cnn_path,
    )

    # Load image
    image = cv2.imread(test_face_image)
    assert image is not None

    # Detect faces
    faces = preprocessor.detect_faces(image)
    assert isinstance(faces, np.ndarray)

    if faces.size == 0:
        pytest.skip(f"No face detected with detector {detector_type}")

    # Align (only for Haar)
    if detector_type == "haar":
        aligned_face = preprocessor.align_face(image, faces[0])
        assert isinstance(aligned_face, np.ndarray)
        assert aligned_face.size > 0
    else:
        aligned_face = image  # No alignment for others

    # Process directory
    input_dir = tmp_path / "input_images"
    output_dir = tmp_path / "output_images"
    os.makedirs(input_dir, exist_ok=True)
    cv2.imwrite(str(input_dir / "face.jpg"), image)

    preprocessor.process_directory(str(input_dir), str(output_dir))

    # Check output image
    assert (output_dir / "face.jpg").exists(), "Output image not saved!"
    output_img = cv2.imread(str(output_dir / "face.jpg"))
    assert output_img is not None
    assert output_img.shape[:2] == (112, 112)
