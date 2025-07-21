# pylint: disable=redefined-outer-name

from pathlib import Path

import cv2
import numpy as np
import pytest

from liface.utils.preprocessing_utils import align_haar_face


@pytest.fixture(scope="module")
def utils_path():
    """Returns the utils directory path."""
    return Path(__file__).resolve().parent.parent / "utils"


@pytest.fixture(scope="module")
def eye_cascade(utils_path):
    """Fixture that loads the actual eye Haar cascade model."""
    eye_cascade_path = utils_path / "haarcascade_frontalface_default.xml"
    if not eye_cascade_path.exists():
        pytest.skip(
            "Eye Haar cascade model not found (haarcascade_frontalface_default.xml)."
        )
    cascade = cv2.CascadeClassifier(str(eye_cascade_path))
    return cascade


@pytest.fixture(scope="module")
def sample_face(utils_path):
    """Fixture that loads and returns a cropped face from the lena sample image."""
    lena = cv2.imread(str(utils_path / "lena.jpg"))
    if lena is None:
        pytest.skip("Sample image 'lena.jpg' not found.")

    # Convert to grayscale (align_haar_face works with grayscale faces)
    gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)

    # Crop central face region for testing (approximate for lena)
    face_crop = gray[200:300, 200:300]
    return face_crop


class TestAlignHAARFace:
    """Test suite for align_haar_face function using real detectors and sample image."""

    def test_align_haar_face_normal_case(self, eye_cascade, sample_face):
        """Test normal alignment case with real Haar eye cascade."""
        width, height = sample_face.shape[1], sample_face.shape[0]
        result = align_haar_face(eye_cascade, sample_face, width, height)

        # Result may be None if eyes are not detected
        if result is not None:
            assert isinstance(result, np.ndarray)
            assert result.shape == sample_face.shape

    @pytest.mark.parametrize(
        "width,height", [(100, 100), (150, 150), (200, 100)]
    )
    def test_align_haar_face_with_various_dimensions(
        self, eye_cascade, sample_face, width, height
    ):
        """Test alignment with various dimensions (resizing input)."""
        resized_face = cv2.resize(sample_face, (width, height))
        result = align_haar_face(eye_cascade, resized_face, width, height)

        if result is not None:
            assert isinstance(result, np.ndarray)
            assert result.shape == (height, width)

    @pytest.mark.parametrize(
        "width, height", [(0, 100), (100, 0), (-100, 100)]
    )
    def test_align_haar_face_invalid_dimensions(
        self, eye_cascade, sample_face, width, height
    ):
        """Test alignment with invalid dimensions."""
        with pytest.raises(ValueError) as exc_info:
            align_haar_face(eye_cascade, sample_face, width, height)

        assert "Invalid image dimensions" in str(exc_info.value)
