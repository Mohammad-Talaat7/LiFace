import os

import numpy as np
import torch
from torch import nn

from liface import configurations as Config  # type: ignore

from .mtcnn_face_detection import detect_face, extract_face


class PNet(nn.Module):
    """MTCNN PNet.

    Keyword Arguments:
        pretrained {bool} -- Whether or not to load saved pretrained weights (default: {True})
    """

    def __init__(self, pretrained=True):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3),
            nn.PReLU(10),
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(10, 16, kernel_size=3),
            nn.PReLU(16),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.PReLU(32),
        )
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1)

        self.training = False

        if pretrained:
            state_dict_path = Config.PNET_PATH
            state_dict = torch.load(state_dict_path)
            self.load_state_dict(state_dict)

    def forward(self, x):
        """
        Forward pass of the PNet (Proposal Network) in MTCNN.

        Args:
            x (torch.Tensor): Input image tensor of shape (N, 3, H, W), where
                N is the batch size, 3 is the number of channels (RGB), and
                H, W are the height and width of the input images.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - b (torch.Tensor): Bounding box regression output of shape (N, 4, H_out, W_out),
                  representing adjustments to bounding boxes at each spatial location.
                - a (torch.Tensor): Face classification probabilities of shape (N, 2, H_out, W_out),
                  where the second dimension represents background vs face class probabilities.

        Notes:
            - This is the first stage in the MTCNN pipeline.
            - It operates on the image pyramid to propose candidate face regions.
            - The output feature map has reduced spatial dimensions due to the MaxPooling layer.
        """
        x = self.feature_extractor(x)
        a = torch.softmax(self.conv4_1(x), dim=1)
        b = self.conv4_2(x)
        return b, a


class RNet(nn.Module):
    """MTCNN RNet.

    Keyword Arguments:
        pretrained {bool} -- Whether or not to load saved pretrained weights (default: {True})
    """

    def __init__(self, pretrained=True):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3),
            nn.PReLU(28),
            nn.MaxPool2d(3, 2, ceil_mode=True),
            nn.Conv2d(28, 48, kernel_size=3),
            nn.PReLU(48),
            nn.MaxPool2d(3, 2, ceil_mode=True),
            nn.Conv2d(48, 64, kernel_size=2),
            nn.PReLU(64),
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(576, 128),
            nn.PReLU(128),
        )

        self.classifier = nn.Linear(128, 2)
        self.bbox_regressor = nn.Linear(128, 4)

        self.training = False

        if pretrained:
            state_dict_path = Config.RNET_PATH
            state_dict = torch.load(state_dict_path)
            self.load_state_dict(state_dict)

    def forward(self, x):
        """
        Forward pass of the RNet (Refine Network) in MTCNN.

        Args:
            x (torch.Tensor): Input image tensor of shape (N, 3, 24, 24), where
                N is the batch size, and 3 is the number of color channels (RGB).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - b (torch.Tensor): Bounding box regression output of shape (N, 4),
                  representing predicted offsets for the bounding boxes.
                - a (torch.Tensor): Face classification probabilities of shape (N, 2),
                  representing confidence scores for [non-face, face] classes.

        Notes:
            - This is the second stage of the MTCNN pipeline.
            - It refines the bounding boxes proposed by PNet and filters out false positives.
            - Input images should be tightly cropped candidate face patches resized to 24×24.
            - The permutation `x.permute(0, 3, 2, 1)` is a non-standard transformation that
              may relate to a specific weight format or compatibility adjustment.
        """
        x = self.features(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.flatten(x)
        x = self.fc(x)
        a = torch.softmax(self.classifier(x), dim=1)
        b = self.bbox_regressor(x)
        return b, a


class ONet(nn.Module):
    """MTCNN ONet.

    Keyword Arguments:
        pretrained {bool} -- Whether or not to load saved pretrained weights (default: {True})
    """

    def __init__(self, pretrained=True):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.PReLU(32),
            nn.MaxPool2d(3, 2, ceil_mode=True),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.PReLU(64),
            nn.MaxPool2d(3, 2, ceil_mode=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.PReLU(64),
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(64, 128, kernel_size=2),
            nn.PReLU(128),
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(nn.Linear(1152, 256), nn.PReLU(256))

        self.classifier = nn.Linear(256, 2)
        self.bbox_regressor = nn.Linear(256, 4)
        self.landmark_regressor = nn.Linear(256, 10)

        self.training = False

        if pretrained:
            state_dict_path = Config.ONET_PATH
            state_dict = torch.load(state_dict_path)
            self.load_state_dict(state_dict)

    def forward(self, x):
        """
        Forward pass of the ONet (Output Network) in MTCNN.

        Args:
            x (torch.Tensor): Input image tensor of shape (N, 3, 48, 48), where
                N is the batch size, and 3 is the number of color channels (RGB).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - b (torch.Tensor): Bounding box regression output of shape (N, 4),
                  representing predicted bounding box offsets [dx1, dy1, dx2, dy2].
                - c (torch.Tensor): Facial landmark localization output of shape (N, 10),
                  representing 5 landmark coordinates (x1, y1, ..., x5, y5).
                - a (torch.Tensor): Face classification probabilities of shape (N, 2),
                  representing confidence scores for [non-face, face] classes.

        Notes:
            - This is the final stage of the MTCNN pipeline.
            - It refines the bounding boxes and additionally predicts facial landmarks.
            - Input images are the candidate face regions resized to 48×48.
            - The `x.permute(0, 3, 2, 1)` step may be required with pretrained weights.
        """
        x = self.features(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.flatten(x)
        x = self.fc(x)
        a = torch.softmax(self.classifier(x), dim=1)
        b = self.bbox_regressor(x)
        c = self.landmark_regressor(x)
        return b, c, a


class MTCNN(nn.Module):
    """MTCNN face detection module.

    This class loads pretrained P-, R-, and O-nets and returns images cropped to include the face
    only, given raw input images of one of the following types:
        - PIL image or list of PIL images
        - numpy.ndarray (uint8) representing either a single image (3D) or a batch of images (4D).
    Cropped faces can optionally be saved to file
    also.

    Keyword Arguments:
        image_size {int} -- Output image size in pixels. The image will be square. (default: {160})
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image.
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size (this is a bug in davidsandberg/facenet).
            (default: {0})
        config (dict) -- Optional dictionary of advanced configuration options. (default: {None})
        device {torch.device} -- The device on which to run neural net passes. Image tensors and
            models are copied to this device before running forward passes. (default: {None})
    """

    def __init__(
        self,
        image_size=Config.OUTPUT_SIZE[0],
        margin=Config.MTCNN_IMAGE_MARGIN,
        config=None,
        device=None,
    ):
        super().__init__()

        config = config or {}

        self.image_size = image_size
        self.margin = margin

        self.config = {
            "min_face_size": config.get(
                "min_face_size", Config.MTCNN_MIN_FACE_SIZE
            ),
            "thresholds": config.get("thresholds", Config.MTCNN_THRESHOLDS),
            "factor": config.get(
                "factor", Config.MTCNN_PYRAMID_SCALING_FACTOR
            ),
            "post_process": config.get(
                "post_process", Config.MTCNN_POST_PROCESS
            ),
            "keep_all": config.get("keep_all", Config.MTCNN_KEEP_ALL),
            "select_largest": config.get(
                "select_largest", Config.MTCNN_SELECT_LARGEST
            ),
        }
        self.config["selection_method"] = config.get(
            "selection_method",
            "largest" if self.config["select_largest"] else "probability",
        )

        self.nets = {
            "pnet": PNet(
                pretrained=config.get("pretrained", Config.MTCNN_PRETRAINED)
            ),
            "rnet": RNet(
                pretrained=config.get("pretrained", Config.MTCNN_PRETRAINED)
            ),
            "onet": ONet(
                pretrained=config.get("pretrained", Config.MTCNN_PRETRAINED)
            ),
        }

        self.device = torch.device(device) if device else torch.device("cpu")
        self.to(self.device)

    def forward(
        self,
        img,
        save_path=Config.MTCNN_SAVE_PATH,
        return_prob=Config.MTCNN_RETURN_PROB,
    ):
        """Run MTCNN face detection on a PIL image or numpy array. This method performs both
        detection and extraction of faces, returning tensors representing detected faces rather
        than the bounding boxes. To access bounding boxes, see the MTCNN.detect() method below.

        Arguments:
            img {PIL.Image, np.ndarray, or list} -- A PIL image, np.ndarray, torch.Tensor, or list.

        Keyword Arguments:
            save_path {str} -- An optional save path for the cropped image. Note that when
                self.post_process=True, although the returned tensor is post processed, the saved
                face image is not, so it is a true representation of the face in the input image.
                If `img` is a list of images, `save_path` should be a list of equal length.
                (default: {None})
            return_prob {bool} -- Whether or not to return the detection probability.
                (default: {False})

        Returns:
            Union[torch.Tensor, tuple(torch.tensor, float)] -- If detected, cropped image of a face
                with dimensions 3 x image_size x image_size. Optionally, the probability that a
                face was detected. If self.keep_all is True, n detected faces are returned in an
                n x 3 x image_size x image_size tensor with an optional list of detection
                probabilities. If `img` is a list of images, the item(s) returned have an extra
                dimension (batch) as the first dimension.
        """

        # Detect faces
        batch_boxes, batch_probs, batch_points = self.detect(
            img, landmarks=True
        )
        # Select faces
        if not self.config["keep_all"]:
            batch_boxes, batch_probs, batch_points = self.select_boxes(
                (batch_boxes, batch_probs, batch_points),
                img,
                method=self.config["selection_method"],
            )
        # Extract faces
        faces = self.extract(img, batch_boxes, save_path)

        if return_prob:
            return faces, batch_probs
        return faces

    def detect(self, img, landmarks=False):
        """Detect all faces in PIL image and return bounding boxes and optional facial landmarks.

        This method is used by the forward method and is also useful for face detection tasks
        that require lower-level handling of bounding boxes and facial landmarks (e.g., face
        tracking). The functionality of the forward function can be emulated by using this method
        followed by the extract_face() function.

        Arguments:
            img {PIL.Image, np.ndarray, or list} -- A PIL image, np.ndarray, torch.Tensor, or list.

        Keyword Arguments:
            landmarks {bool} -- Whether to return facial landmarks in addition to bounding boxes.
                (default: {False})

        Returns:
            tuple(numpy.ndarray, list) -- For N detected faces, a tuple containing an
                Nx4 array of bounding boxes and a length N list of detection probabilities.
                Returned boxes will be sorted in descending order by detection probability if
                self.select_largest=False, otherwise the largest face will be returned first.
                If `img` is a list of images, the items returned have an extra dimension
                (batch) as the first dimension. Optionally, a third item, the facial landmarks,
                are returned if `landmarks=True`.

        Example:
        >>> from PIL import Image, ImageDraw
        >>> from facenet_pytorch import MTCNN, extract_face
        >>> mtcnn = MTCNN(keep_all=True)
        >>> boxes, probs, points = mtcnn.detect(img, landmarks=True)
        >>> # Draw boxes and save faces
        >>> img_draw = img.copy()
        >>> draw = ImageDraw.Draw(img_draw)
        >>> for i, (box, point) in enumerate(zip(boxes, points)):
        ...     draw.rectangle(box.tolist(), width=5)
        ...     for p in point:
        ...         draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=10)
        ...     extract_face(img, box, save_path='detected_face_{}.png'.format(i))
        >>> img_draw.save('annotated_faces.png')
        """

        with torch.no_grad():
            config = {
                "minsize": self.config["min_face_size"],
                "factor": self.config["factor"],
                "threshold": self.config["thresholds"],
            }
            batch_boxes, batch_points = detect_face(
                imgs=img, config=config, nets=self.nets, device=self.device
            )

        boxes, probs, points = [], [], []
        for box, point in zip(batch_boxes, batch_points):
            box = np.array(box)
            point = np.array(point)
            if len(box) == 0:
                boxes.append(None)
                probs.append([None])
                points.append(None)
            elif self.config["select_largest"]:
                box_order = np.argsort(
                    (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
                )[::-1]
                box = box[box_order]
                point = point[box_order]
                boxes.append(box[:, :4])
                probs.append(box[:, 4])
                points.append(point)
            else:
                boxes.append(box[:, :4])
                probs.append(box[:, 4])
                points.append(point)
        boxes = np.array(boxes, dtype=object)
        probs = np.array(probs, dtype=object)
        points = np.array(points, dtype=object)

        if (
            not isinstance(img, (list, tuple))
            and not (isinstance(img, np.ndarray) and len(img.shape) == 4)
            and not (isinstance(img, torch.Tensor) and len(img.shape) == 4)
        ):
            boxes = boxes[0]
            probs = probs[0]
            points = points[0]

        if not landmarks:
            points = None
        return boxes, probs, points

    def select_boxes(
        self,
        detections,
        imgs,
        method="probability",
    ):
        """Selects a single box from multiple for a given image using one of multiple heuristics.

        Arguments:
                detections {Tuple}: Tuple of (boxes, probs, points):
                    boxes {np.ndarray} -- Ix0 ndarray where each element is a Nx4 ndarry of
                        bounding boxes for N detected faces in I images (output from self.detect).
                    probs {np.ndarray} -- Ix0 ndarray where each element is a Nx0 ndarry of
                        probabilities for N detected faces in I images (output from self.detect).
                    points {np.ndarray} -- Ix0 ndarray where each element is a Nx5x2 array of
                        points for N detected faces. (output from self.detect).
                imgs {PIL.Image, np.ndarray, or list} -- A PIL image, np.ndarray, torch.Tensor, list

        Keyword Arguments:
                method {str} -- Which heuristic to use for selection:
                    "probability": highest probability selected
                    "largest": largest box selected
                    "largest_over_theshold@<threshold>": largest box over probability selected
                    "center_weighted_size@<weight>": boxsize - weighted squared offset from center
                    (default: {'probability'})
                threshold {float} -- theshold for "largest_over_threshold" method. (default: {0.9})
                center_weight {float} -- weight for squared offset in center weighted size method.
                    (default: {2.0})

        Returns:
                tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray) -- nx4 ndarray of bounding boxes
                    for n images. Ix0 array of probabilities for each box, array of landmark points.
        """

        boxes_list, probs_list, points_list = detections
        is_batch = isinstance(imgs, (list, tuple)) or (
            hasattr(imgs, "shape") and len(imgs.shape) == 4
        )

        if not is_batch:
            imgs = [imgs]
            boxes_list, probs_list, points_list = (
                [boxes_list],
                [probs_list],
                [points_list],
            )

        method_type, method_val = self._parse_method(method)

        results = [
            self._select_from_image((b, p, pts), img, method_type, method_val)
            for b, p, pts, img in zip(
                boxes_list, probs_list, points_list, imgs
            )
        ]
        boxes, probs, points = zip(*results)

        if is_batch:
            return np.array(boxes), np.array(probs), np.array(points)
        return boxes[0], probs[0], points[0]

    def _parse_method(self, method):
        if "@" in method:
            base, val = method.split("@")
            return base, float(val)
        return method, None

    def _select_from_image(self, detections, img, method, val):
        if detections[0] is None:
            return None, [None], None

        boxes = np.array(detections[0])
        probs = np.array(detections[1])
        points = np.array(detections[2])

        if method == "probability":
            idx = np.argmax(probs)

        elif method == "largest":
            idx = np.argmax(
                (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            )

        elif method == "largest_over_threshold":
            threshold = val if val is not None else 0.9
            if not np.any(probs > threshold):
                return None, [None], None
            boxes, probs, points = (
                boxes[probs > threshold],
                probs[probs > threshold],
                points[probs > threshold],
            )
            idx = np.argmax(
                (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            )

        elif method == "center_weighted_size":
            weight = val if val is not None else 2.0
            center = (img.width / 2, img.height / 2)
            centers = np.stack(
                [
                    (boxes[:, 0] + boxes[:, 2]) / 2,
                    (boxes[:, 1] + boxes[:, 3]) / 2,
                ],
                axis=1,
            )
            offsets = np.sum((centers - center) ** 2, axis=1)
            scores = (boxes[:, 2] - boxes[:, 0]) * (
                boxes[:, 3] - boxes[:, 1]
            ) - weight * offsets
            idx = np.argmax(scores)

        else:
            raise ValueError(f"Unknown selection method: {method}")

        return boxes[[idx]], probs[[idx]], points[[idx]]

    def extract(self, img, batch_boxes, save_path):
        """
        Extracts and optionally saves aligned face crops from input image(s)
        using bounding box coordinates.

        Args:
            img (Union[np.ndarray, PIL.Image.Image, torch.Tensor, List]):
                A single image or a batch of images from which faces will be extracted.
                Each image can be a NumPy array, PIL image, or Torch tensor.

            batch_boxes (Union[np.ndarray, List[np.ndarray], None]):
                Detected bounding boxes for faces in each image.
                Should be a list of arrays (one per image), where each array has shape (N, 4).
                If `None` for a particular image, face extraction will be skipped for that image.

            save_path (Union[str, List[str]]):
                Path or list of paths to save the extracted faces.
                If a single image is passed, this can be a string.
                If a batch is passed, should be a list of the same length as `img`.

        Returns:
            Union[torch.Tensor, List[torch.Tensor], None, List[None]]:
                - If a single image is passed and`keep_all=False`,returns single tensor for one face
                - If a single image is passed and`keep_all=True`,returns tensor stack of all faces.
                - If a batch is passed, returns a list where each item is either a tensor or `None`.

        Notes:
            - The method supports post-processing of faces (e.g., resizing and normalization)
              if enabled in `self.config["post_process"]`.
            - The `keep_all` flag in `self.config` determines whether to extract all detected faces
              or just the most confident one.
            - If no boxes are provided for an image, `None` is returned for that image.
            - This function relies on internal helper methods such as `_is_batch_input`,
              `_ensure_list`, `_normalize_save_paths`, and `_extract_faces`.

        Raises:
            ValueError: If `img` and `save_path` lengths mismatch when given as lists.
        """
        is_batch = self._is_batch_input(img)
        img, batch_boxes = self._ensure_list(img, batch_boxes, is_batch)
        save_path = self._normalize_save_paths(save_path, len(img))

        faces = []
        keep_all = self.config["keep_all"]
        post_process = self.config["post_process"]

        for im, box_im, path_im in zip(img, batch_boxes, save_path):
            if box_im is None:
                faces.append(None)
                continue

            boxes_to_process = box_im if keep_all else box_im[[0]]
            faces_im = self._extract_faces(
                im, boxes_to_process, path_im, post_process
            )

            if keep_all:
                faces_im = torch.stack(faces_im)
            else:
                faces_im = faces_im[0]

            faces.append(faces_im)

        return faces if is_batch else faces[0]

    def _is_batch_input(self, img):
        if isinstance(img, (list, tuple)):
            return True
        if isinstance(img, np.ndarray) and len(img.shape) == 4:
            return True
        if isinstance(img, torch.Tensor) and len(img.shape) == 4:
            return True
        return False

    def _ensure_list(self, img, batch_boxes, is_batch):
        if not is_batch:
            return [img], [batch_boxes]
        return img, batch_boxes

    def _normalize_save_paths(self, save_path, length):
        if isinstance(save_path, str):
            return [save_path]
        if save_path is None:
            return [None] * length
        return save_path

    def _extract_faces(self, img, boxes, path, post_process):
        faces = []
        for i, box in enumerate(boxes):
            face_path = self._get_face_path(path, i)
            face = extract_face(
                img, box, self.image_size, self.margin, face_path
            )
            if post_process:
                face = (face - 127.5) / 128.0
            faces.append(face)
        return faces

    def _get_face_path(self, base_path, index):
        if base_path is None or index == 0:
            return base_path
        name, ext = os.path.splitext(base_path)
        return f"{name}_{index + 1}{ext}"
