# pylint: disable=redefined-outer-name

from pathlib import Path

import cv2
import dlib
import pytest

from liface.utils.mtcnn import MTCNN
from liface.utils.preprocessing_utils import return_detector


@pytest.fixture(scope="module")
def utils_path():
    """Returns the utils directory path."""
    return Path(__file__).resolve().parent.parent / "utils"


def test_haar_detector(utils_path):
    """Test HAAR cascade detector initialization with real model."""
    haar_model_path = utils_path / "haarcascade_frontalface_default.xml"
    assert haar_model_path.exists(), "Haar cascade model not found!"

    detector, eye_cascade = return_detector(
        "haar", cascade_path=str(haar_model_path)
    )
    assert isinstance(detector, cv2.CascadeClassifier)
    assert isinstance(eye_cascade, cv2.CascadeClassifier)


def test_mtcnn_detector():
    """Test MTCNN detector initialization."""
    detector = return_detector("mtcnn", device="cpu")
    assert isinstance(detector, MTCNN)


def test_dlib_hog_detector():
    """Test dlib HOG detector initialization."""
    detector = return_detector("dlib-hog")
    assert isinstance(detector, dlib.fhog_object_detector)  # type: ignore # pylint: disable=no-member


def test_dlib_cnn_detector_valid_model(utils_path):
    """Test dlib CNN detector initialization with valid model."""
    cnn_model_path = utils_path / "mmod_human_face_detector.dat"
    assert cnn_model_path.exists(), "Dlib CNN model not found!"

    detector = return_detector("dlib-cnn", cnn_model_path=str(cnn_model_path))
    assert isinstance(detector, dlib.cnn_face_detection_model_v1)  # type: ignore # pylint: disable=no-member


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_mtcnn_different_devices(device):
    """Test MTCNN initialization with different devices."""
    try:
        detector = return_detector("mtcnn", device=device)
        assert isinstance(detector, MTCNN)
    except RuntimeError:
        pytest.skip(f"Device {device} not available")


def test_invalid_detector_type():
    """Test that invalid detector type raises ValueError."""
    with pytest.raises(ValueError):
        return_detector("invalid_detector")


def test_haar_invalid_cascade_path():
    """Test that HAAR detector raises FileNotFoundError with invalid cascade path."""
    with pytest.raises(FileNotFoundError):
        return_detector("haar", cascade_path="invalid_path.xml")


def test_dlib_cnn_missing_model():
    """Test that dlib CNN detector raises ValueError when model path is missing."""
    with pytest.raises(ValueError):
        return_detector("dlib-cnn")


def test_dlib_cnn_invalid_model_path(tmp_path):
    """Test that dlib CNN detector raises ValueError with invalid model path."""
    invalid_path = tmp_path / "non_existent_model.dat"
    with pytest.raises(ValueError):
        return_detector("dlib-cnn", cnn_model_path=str(invalid_path))
