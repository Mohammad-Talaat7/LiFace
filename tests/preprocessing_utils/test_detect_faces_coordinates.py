# pylint: disable=redefined-outer-name

from pathlib import Path

import cv2
import dlib
import numpy as np
import pytest

from facenet_pytorch import MTCNN  # type: ignore
from liface.utils.preprocessing_utils import detect_faces_coordinates


@pytest.fixture(scope="module")
def utils_path():
    """Returns the utils directory path."""
    return Path(__file__).resolve().parent.parent / "utils"


@pytest.fixture(scope="module")
def sample_image(utils_path):
    """Loads sample test image (lena.jpg) for detection tests."""
    lena = cv2.imread(str(utils_path / "lena.jpg"))
    if lena is None:
        pytest.skip("Sample image 'lena.jpg' not found.")
    return lena


@pytest.fixture(scope="module")
def haar_detector(utils_path):
    """Fixture for Haar cascade detector."""
    haar_model_path = utils_path / "haarcascade_frontalface_default.xml"
    assert haar_model_path.exists(), "Haar cascade model not found!"
    return cv2.CascadeClassifier(str(haar_model_path))


@pytest.fixture(scope="module")
def mtcnn_detector():
    """Fixture for MTCNN detector."""
    return MTCNN(keep_all=True, device="cpu")


@pytest.fixture(scope="module")
def dlib_hog_detector():
    """Fixture for Dlib HOG detector."""
    return dlib.get_frontal_face_detector()  # type: ignore # pylint: disable=no-member


@pytest.fixture(scope="module")
def dlib_cnn_detector(utils_path):
    """Fixture for Dlib CNN detector."""
    cnn_model_path = utils_path / "mmod_human_face_detector.dat"
    assert cnn_model_path.exists(), "Dlib CNN model not found!"
    return dlib.cnn_face_detection_model_v1(str(cnn_model_path))  # type: ignore # pylint: disable=no-member


class TestDetectFacesCoordinates:
    """Test suite for detect_faces_coordinates function."""

    @pytest.mark.parametrize(
        "detector_type, detector_fixture",
        [
            ("haar", "haar_detector"),
            ("mtcnn", "mtcnn_detector"),
            ("dlib-hog", "dlib_hog_detector"),
            ("dlib-cnn", "dlib_cnn_detector"),
        ],
    )
    def test_face_detection_with_all_detectors(
        self, detector_type, detector_fixture, sample_image, request
    ):
        """Test face detection for all supported detectors."""
        detector = request.getfixturevalue(detector_fixture)
        result = detect_faces_coordinates(
            detector_type, detector, sample_image
        )
        assert isinstance(result, np.ndarray)
        if result.size > 0:
            assert result.shape[1] == 4  # 4 coordinates

    def test_invalid_detector_type(self, sample_image, haar_detector):
        """Test invalid detector type raises ValueError."""
        with pytest.raises(ValueError):
            detect_faces_coordinates(
                "invalid_detector", haar_detector, sample_image
            )

    @pytest.mark.parametrize("invalid_detector_type", [None, "", 123, 3.14])
    def test_invalid_detector_type_values(
        self, invalid_detector_type, sample_image, haar_detector
    ):
        """Test with various invalid detector type values."""
        with pytest.raises(ValueError):
            detect_faces_coordinates(
                invalid_detector_type, haar_detector, sample_image
            )

    def test_mismatched_detector(self, sample_image, haar_detector):
        """Test mismatched detector type and detector object raises error."""
        with pytest.raises(Exception):
            detect_faces_coordinates("mtcnn", haar_detector, sample_image)

    def test_empty_image(self, haar_detector):
        """Test with empty image raises error."""
        with pytest.raises(Exception):
            detect_faces_coordinates("haar", haar_detector, np.array([]))

    def test_invalid_image_dimensions(self, haar_detector):
        """Test with invalid image dimensions raises error."""
        invalid_image = np.random.randint(
            0, 255, (100, 100), dtype=np.uint8
        )  # 2D array
        with pytest.raises(Exception):
            detect_faces_coordinates("haar", haar_detector, invalid_image)
