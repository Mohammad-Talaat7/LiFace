from pathlib import Path

# Base paths
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

# Model paths
PNET_PATH = MODEL_DIR / "pnet.pt"
RNET_PATH = MODEL_DIR / "rnet.pt"
ONET_PATH = MODEL_DIR / "onet.pt"
HAAR_CASCADE_PATH = MODEL_DIR / "haarcascade_frontalface_default.xml"
DLIB_CNN_PATH = MODEL_DIR / "mmod_human_face_detector.dat"

# Global Configuration
OUTPUT_SIZE = (112, 112)
DETECTOR_TYPE = "haar"
DEVICE = "cpu"
RANDOM_SEED = 42
ENABLED_AUGMENTATIONS = None
NUM_AUGMENTATIONS_PER_IMAGE = 3
NUM_AUGMENTATIONS_PER_DATASET = 5

# MTCNN Configuration
MTCNN_IMAGE_MARGIN = 0
MTCNN_MIN_FACE_SIZE = 20
MTCNN_THRESHOLDS = {"pnet": 0.6, "rnet": 0.7, "onet": 0.7}
MTCNN_PYRAMID_SCALING_FACTOR = 0.709
MTCNN_POST_PROCESS = True
MTCNN_KEEP_ALL = True
MTCNN_SELECT_LARGEST = True
MTCNN_PRETRAINED = True
MTCNN_SAVE_PATH = None
MTCNN_RETURN_PROB = False
MTCNN_BATCH_SIZE = 512
