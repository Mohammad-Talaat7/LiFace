import os

import numpy as np
import torch
from PIL import Image
from torchvision.ops.boxes import batched_nms
from torchvision.transforms import functional as F

from .mtcnn_utils import (
    crop_resize,
    empty_results,
    fixed_batch_process,
    get_size,
    imresample,
    pad,
    rerec,
    save_img,
)


def preprocess_images(imgs, device):
    """
    Preprocess input images for model inference.

    Converts input images into a 4D PyTorch tensor with shape (N, C, H, W)
    and moves it to the specified device. Accepts various input types including
    NumPy arrays, PyTorch tensors, and lists/tuples of PIL Images.

    Args:
        imgs (np.ndarray | torch.Tensor | list | tuple): The input image(s).
            Supported formats:
            - Single NumPy array of shape (H, W, C) or (N, H, W, C)
            - Single PyTorch tensor of shape (H, W, C) or (N, H, W, C)
            - List or tuple of PIL Images with the same dimensions
        device (torch.device): The device to move the resulting tensor to.

    Returns:
        torch.Tensor: A 4D tensor of shape (N, C, H, W) with dtype float32
        and located on the specified device.

    Raises:
        ValueError: If a list/tuple of images contains images with
        inconsistent dimensions.
    """
    if isinstance(imgs, (np.ndarray, torch.Tensor)):
        imgs = (
            torch.as_tensor(np.copy(imgs), device=device)
            if isinstance(imgs, np.ndarray)
            else imgs.to(device)
        )
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(0)
    else:
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]
        if any(img.size != imgs[0].size for img in imgs):
            raise ValueError(
                "MTCNN batch processing only supports images with the same dimensions."
            )
        imgs = torch.as_tensor(
            np.stack([np.uint8(img) for img in imgs]), device=device
        )

    imgs = imgs.permute(0, 3, 1, 2).type(torch.float32)
    return imgs


def build_pyramid(h, w, minsize, factor):
    """
    Build an image scale pyramid for multi-scale processing.

    Computes a list of scaling factors used to resize the input image to
    progressively smaller resolutions. Commonly used in object detection
    algorithms like MTCNN to detect objects at multiple scales.

    Args:
        h (int): Height of the original image.
        w (int): Width of the original image.
        minsize (int): Minimum object size to detect.
        factor (float): Scaling factor to reduce the image size at each level
            (e.g., 0.709).

    Returns:
        list of float: A list of scale factors, each corresponding to a level
        in the image pyramid.

    Notes:
        - The pyramid generation stops when the smallest side of the image
          becomes smaller than 12 pixels.
        - The constant `12.0` corresponds to the minimum size the detector
          can handle at the base level.
    """
    m = 12.0 / minsize
    minl = min(h, w) * m
    scales = []

    scale = m
    while minl >= 12:
        scales.append(scale)
        scale *= factor
        minl *= factor

    return scales


def bbreg(boundingbox, reg):
    """
    Apply bounding box regression to refine detected boxes.

    Adjusts the positions of bounding boxes using the provided regression
    offsets. Typically used to improve localization accuracy in object
    detection pipelines such as MTCNN.

    Args:
        boundingbox (torch.Tensor): A tensor of shape (N, 5) or (N, 4)
            containing bounding box coordinates in the format (x1, y1, x2, y2, ...).
        reg (torch.Tensor): A tensor of shape (N, 4) or (1, 4, N, 1) containing
            the regression offsets for the bounding boxes.

    Returns:
        torch.Tensor: The refined bounding boxes, with the first 4 columns
        updated based on the regression offsets.

    Raises:
        ValueError: If `reg` has an unexpected shape that prevents reshaping.
    """
    if reg.shape[1] == 1:
        reg = torch.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b1 = boundingbox[:, 0] + reg[:, 0] * w
    b2 = boundingbox[:, 1] + reg[:, 1] * h
    b3 = boundingbox[:, 2] + reg[:, 2] * w
    b4 = boundingbox[:, 3] + reg[:, 3] * h
    boundingbox[:, :4] = torch.stack([b1, b2, b3, b4]).permute(1, 0)

    return boundingbox


def _process_scale(imgs, scale, pnet, threshold, offset):
    """
    Run P-Net on a single scale of the image pyramid.

    Resizes the input images according to the given scale, applies the
    P-Net to detect candidate bounding boxes, filters them based on
    threshold and non-maximum suppression (NMS), and returns the results
    with updated indices.

    Args:
        imgs (torch.Tensor): A 4D tensor of shape (N, 3, H, W), where N is the
            number of images in the batch.
        scale (float): The scaling factor to resize the images.
        pnet (Callable): The Proposal Network that returns (reg, probs) for the input.
        threshold (float): Confidence score threshold for keeping boxes.
        offset (int): Offset index to correctly label boxes across scales.

    Returns:
        tuple:
            - b (torch.Tensor): Bounding boxes of shape (K, 5) where each row is
              (x1, y1, x2, y2, score).
            - inds (torch.Tensor): Image indices of shape (K,) corresponding to `b`.
            - scale_pick (torch.Tensor): Indices of selected boxes after NMS,
              offset to match global indexing.
    """
    h, w = imgs.shape[2:4]
    im_data = imresample(imgs, (int(h * scale + 1), int(w * scale + 1)))
    im_data = (im_data - 127.5) * 0.0078125
    reg, probs = pnet(im_data)

    b, inds = generate_bounding_box(reg, probs[:, 1], scale, threshold)
    pick = batched_nms(b[:, :4], b[:, 4], inds, 0.5)
    scale_pick = pick + offset
    return b, inds, scale_pick


def run_pnet(imgs, scales, pnet, threshold):
    """
    Run Proposal Network (P-Net) across an image pyramid to generate candidate boxes.

    Applies the P-Net to input images at multiple scales, collects and refines
    candidate bounding boxes using non-maximum suppression (NMS), and returns
    the final filtered boxes along with their corresponding image indices.

    Args:
        imgs (torch.Tensor): A 4D tensor of shape (N, 3, H, W) representing a
            batch of input images.
        scales (list of float): A list of scale factors used to resize the input
            images for multi-scale detection.
        pnet (Callable): The Proposal Network (P-Net) that outputs (reg, probs)
            for an input tensor.
        threshold (float): Minimum confidence score required to keep a box.

    Returns:
        tuple:
            - boxes (torch.Tensor): A tensor of shape (M, 5), where each row is
              (x1, y1, x2, y2, score) for the retained proposal.
            - image_inds (torch.Tensor): A 1D tensor of shape (M,) indicating the
              image index for each retained box.

    Notes:
        - Uses `_process_scale` internally to process each scale.
        - Performs NMS twice: once per scale and again after combining all scales.
        - Returns empty tensors if no boxes are detected.
    """
    boxes, image_inds, scale_picks = [], [], []
    offset = 0

    for scale in scales:
        b, inds, scale_pick = _process_scale(
            imgs, scale, pnet, threshold, offset
        )
        boxes.append(b)
        image_inds.append(inds)
        scale_picks.append(scale_pick)
        offset += b.shape[0]

    if not boxes:
        return torch.empty(0), torch.empty(0, dtype=torch.int)

    boxes = torch.cat(boxes, dim=0)
    image_inds = torch.cat(image_inds, dim=0)
    scale_picks = torch.cat(scale_picks, dim=0)

    boxes = boxes[scale_picks]
    image_inds = image_inds[scale_picks]

    pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
    return boxes[pick], image_inds[pick]


def calibrate_and_pad_boxes(boxes):
    """
    Apply bounding box regression and convert to square boxes.

    This function adjusts bounding box coordinates using regression offsets
    (typically output by a network), and then pads or crops them into square
    shapes using the `rerec` function.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 9), where each row is
            (x1, y1, x2, y2, score, dx1, dy1, dx2, dy2). The last four elements
            are the bounding box regression offsets.

    Returns:
        torch.Tensor: A tensor of shape (N, 5) containing square bounding boxes
        in the format (x1, y1, x2, y2, score) after calibration and padding.

    Notes:
        - Bounding box regression is performed using the width and height
          of each box and the provided offsets.
        - The resulting boxes are converted to square format via `rerec`.
        - The function assumes the input tensor includes regression values
          starting at index 5.
    """
    regw = boxes[:, 2] - boxes[:, 0]
    regh = boxes[:, 3] - boxes[:, 1]
    qq1 = boxes[:, 0] + boxes[:, 5] * regw
    qq2 = boxes[:, 1] + boxes[:, 6] * regh
    qq3 = boxes[:, 2] + boxes[:, 7] * regw
    qq4 = boxes[:, 3] + boxes[:, 8] * regh
    boxes = torch.stack([qq1, qq2, qq3, qq4, boxes[:, 4]]).permute(1, 0)
    boxes = rerec(boxes)
    return boxes


def _extract_rnet_crops(imgs, boxes, image_inds):
    """
    Extract and resize image crops for R-Net input.

    Pads and crops regions from the original images based on bounding boxes,
    then resamples each crop to 24×24 resolution for input into R-Net
    (second stage of MTCNN).

    Args:
        imgs (torch.Tensor): A 4D tensor of shape (N, 3, H, W) representing
            the batch of input images.
        boxes (torch.Tensor): A tensor of shape (M, 5) representing bounding
            boxes in the format (x1, y1, x2, y2, score).
        image_inds (torch.Tensor): A 1D tensor of length M, indicating which
            image each box belongs to.

    Returns:
        list of torch.Tensor: A list of tensors, each of shape (1, 3, 24, 24),
        representing the cropped and resized regions ready for R-Net inference.

    Notes:
        - Uses `pad` to compute cropping boundaries and handle edge cases.
        - Resizes each region to 24×24 using `imresample`.
        - Only extracts crops where the box lies within valid bounds.
    """
    y, ey, x, ex = pad(boxes, imgs.shape[3], imgs.shape[2])
    crops = []

    for k, _ in enumerate(y):
        if ey[k] > y[k] - 1 and ex[k] > x[k] - 1:
            img_k = imgs[image_inds[k], :, y[k] - 1 : ey[k], x[k] - 1 : ex[k]]
            img_k = img_k.unsqueeze(0)
            crops.append(imresample(img_k, (24, 24)))

    return crops


def _filter_rnet_outputs(im_data, boxes, image_inds, rnet, threshold):
    """
    Run R-Net on cropped image regions and filter outputs based on confidence and NMS.

    Normalizes the input image crops, applies R-Net to generate confidence scores
    and bounding box regression offsets, filters out low-confidence detections,
    applies bounding box regression and squaring, and performs non-maximum
    suppression (NMS) to reduce duplicate detections.

    Args:
        im_data (torch.Tensor): A tensor of shape (N, 3, 24, 24) representing
            normalized image crops to be processed by R-Net.
        boxes (torch.Tensor): A tensor of shape (N, 5) containing candidate
            bounding boxes (x1, y1, x2, y2, score).
        image_inds (torch.Tensor): A tensor of shape (N,) indicating the
            original image index for each box.
        rnet (Callable): The R-Net model which returns (regression, confidence)
            outputs when called.
        threshold (float): The minimum confidence score to retain a box.

    Returns:
        tuple:
            - torch.Tensor: Refined and squared bounding boxes of shape (M, 5),
              where M is the number of boxes remaining after filtering and NMS.
            - torch.Tensor: Corresponding image indices of shape (M,).

    Notes:
        - Bounding box regression is applied using outputs from R-Net.
        - NMS is applied with an IoU threshold of 0.7 to reduce duplicate detections.
        - Input `im_data` is assumed to be already padded/resampled to 24×24 crops.
    """
    im_data = (im_data - 127.5) * 0.0078125
    out0, out1 = fixed_batch_process(im_data, rnet)

    score = out1.permute(1, 0)[1, :]
    ipass = score > threshold

    filtered_boxes = torch.cat(
        (boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1
    )
    filtered_inds = image_inds[ipass]
    mv = out0.permute(1, 0)[:, ipass].permute(1, 0)

    pick = batched_nms(
        filtered_boxes[:, :4], filtered_boxes[:, 4], filtered_inds, 0.7
    )

    return rerec(bbreg(filtered_boxes[pick], mv[pick])), filtered_inds[pick]


def run_rnet(imgs, boxes, image_inds, rnet, threshold):
    """
    Run R-Net on image crops to refine and filter bounding boxes.

    Extracts 24×24 crops from the original images using the given bounding boxes,
    feeds them through R-Net to get confidence scores and bounding box regressions,
    then filters and refines the boxes based on a confidence threshold and NMS.

    Args:
        imgs (torch.Tensor): A 4D tensor of shape (N, 3, H, W) representing the
            batch of input images.
        boxes (torch.Tensor): A tensor of shape (M, 5) containing candidate bounding
            boxes in the format (x1, y1, x2, y2, score).
        image_inds (torch.Tensor): A 1D tensor of shape (M,) indicating which image
            each box corresponds to in `imgs`.
        rnet (Callable): The R-Net model that outputs bounding box regressions
            and confidence scores when applied to image crops.
        threshold (float): Minimum confidence score for keeping a detection.

    Returns:
        tuple:
            - torch.Tensor: A tensor of shape (K, 5) with refined and squared
              bounding boxes that passed the threshold and NMS.
            - torch.Tensor: A tensor of shape (K,) indicating the image index
              each final box belongs to.

    Notes:
        - Returns empty tensors if no crops are extracted.
        - Internally uses `_extract_rnet_crops` and `_filter_rnet_outputs`.
        - This is the second stage in the MTCNN pipeline.
    """
    crops = _extract_rnet_crops(imgs, boxes, image_inds)

    if not crops:
        return torch.empty(0), torch.empty(0, dtype=torch.int)

    im_data = torch.cat(crops, dim=0)
    return _filter_rnet_outputs(im_data, boxes, image_inds, rnet, threshold)


def _extract_onet_crops(imgs, boxes, image_inds):
    """
    Extract and resize image crops for O-Net input.

    Pads and crops regions from the original images based on bounding boxes,
    then resamples each crop to 48×48 resolution for input into O-Net
    (third stage of MTCNN).

    Args:
        imgs (torch.Tensor): A 4D tensor of shape (N, 3, H, W) representing
            the batch of input images.
        boxes (torch.Tensor): A tensor of shape (M, 5) representing bounding
            boxes in the format (x1, y1, x2, y2, score).
        image_inds (torch.Tensor): A 1D tensor of length M, indicating which
            image each box belongs to.

    Returns:
        list of torch.Tensor: A list of tensors, each of shape (1, 3, 48, 48),
        representing the cropped and resized regions ready for O-Net inference.

    Notes:
        - Uses `pad` to compute safe crop boundaries.
        - Skips invalid crops that would result in negative dimensions.
        - Uses `imresample` to resize each crop to the expected O-Net input size.
    """
    y, ey, x, ex = pad(boxes, imgs.shape[3], imgs.shape[2])
    crops = []

    for k, _ in enumerate(y):
        if ey[k] > y[k] - 1 and ex[k] > x[k] - 1:
            img_k = imgs[image_inds[k], :, y[k] - 1 : ey[k], x[k] - 1 : ex[k]]
            crops.append(imresample(img_k.unsqueeze(0), (48, 48)))

    return crops


def _run_onet_model(im_data, onet):
    """
    Run O-Net on preprocessed image crops.

    Normalizes the input image crops and applies the O-Net model to obtain
    bounding box regression outputs and classification scores.

    Args:
        im_data (torch.Tensor): A tensor of shape (N, 3, 48, 48) representing
            the cropped and resized face regions, ready for O-Net input.
        onet (Callable): The O-Net model that returns (regression, confidence)
            outputs for the given input tensor.

    Returns:
        tuple:
            - torch.Tensor: Bounding box regression offsets of shape (N, 4).
            - torch.Tensor: Classification scores of shape (N, 2), where the
              second column typically contains face confidence probabilities.

    Notes:
        - Input tensor must be normalized to the [-1, 1] range as expected by O-Net.
        - Uses `fixed_batch_process` for efficient batch inference.
    """
    im_data = (im_data - 127.5) * 0.0078125
    return fixed_batch_process(im_data, onet)


def _postprocess_onet_outputs(
    boxes: torch.Tensor,
    image_inds: torch.Tensor,
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    threshold: float,
):
    """
    Postprocess O-Net outputs to produce final face detections and landmarks.

    Filters detections by confidence score, applies bounding box regression
    and landmark transformation, then performs non-maximum suppression (NMS)
    to finalize the detections.

    Args:
        boxes (torch.Tensor): Tensor of shape (N, 5) with bounding boxes
            (x1, y1, x2, y2, score) before refinement.
        image_inds (torch.Tensor): Tensor of shape (N,) indicating the source
            image index for each box.
        out0 (torch.Tensor): Bounding box regression output of shape (4, N).
        out1 (torch.Tensor): Landmark localization output of shape (10, N),
            with 5 x-points followed by 5 y-points.
        out2 (torch.Tensor): Classification scores of shape (2, N).
        threshold (float): Minimum confidence score to retain a detection.

    Returns:
        tuple:
            - boxes (torch.Tensor): Refined bounding boxes of shape (M, 5),
              where M is the number of final detections.
            - image_inds (torch.Tensor): Tensor of shape (M,) indicating the
              image index for each final detection.
            - points (torch.Tensor): Landmark coordinates of shape (M, 5, 2),
              where each point is (x, y) for left eye, right eye, nose, left
              mouth corner, right mouth corner.

    Notes:
        - Applies bounding box regression using `bbreg`.
        - Converts landmark coordinates from relative to absolute positions.
        - Applies `batched_nms_numpy` with "Min" strategy to suppress duplicates.
        - Returns empty tensors if no boxes pass the threshold.
    """
    out0, out1, out2 = outputs
    ipass = (out2.permute(1, 0)[1, :]) > threshold

    if not ipass.any():
        return (
            torch.empty(0),
            torch.empty(0, dtype=torch.int),
            torch.zeros(0, 5, 2),
        )

    boxes = torch.cat(
        (boxes[ipass, :4], (out2.permute(1, 0)[1, :])[ipass].unsqueeze(1)),
        dim=1,
    )
    image_inds = image_inds[ipass]
    mv = out0.permute(1, 0)[:, ipass].permute(1, 0)
    points = out1[ipass]

    w_i = boxes[:, 2] - boxes[:, 0] + 1
    h_i = boxes[:, 3] - boxes[:, 1] + 1

    points_x = points[:, 0:5] * w_i.unsqueeze(1) + boxes[:, 0].unsqueeze(1) - 1
    points_y = (
        points[:, 5:10] * h_i.unsqueeze(1) + boxes[:, 1].unsqueeze(1) - 1
    )

    points = torch.stack((points_x, points_y), dim=2)

    boxes = bbreg(boxes, mv)

    pick = batched_nms_numpy(boxes[:, :4], boxes[:, 4], image_inds, 0.7, "Min")

    return boxes[pick], image_inds[pick], points[pick]


def run_onet(imgs, boxes, image_inds, onet, threshold):
    """
    Run O-Net on image crops to produce final face detections and landmarks.

    Extracts 48×48 face crops from the original images based on candidate
    bounding boxes, processes them through O-Net to refine box locations and
    predict facial landmarks, and postprocesses the outputs with NMS and box
    calibration.

    Args:
        imgs (torch.Tensor): A 4D tensor of shape (N, 3, H, W), representing
            the batch of input images.
        boxes (torch.Tensor): A tensor of shape (M, 5), containing candidate
            bounding boxes in the format (x1, y1, x2, y2, score).
        image_inds (torch.Tensor): A 1D tensor of shape (M,) indicating which
            image each box belongs to.
        onet (Callable): The O-Net model, which outputs:
            - out0: bounding box regression (4, M),
            - out1: landmark localization (10, M),
            - out2: classification scores (2, M).
        threshold (float): Minimum confidence score to retain a detection.

    Returns:
        tuple:
            - torch.Tensor: Final refined bounding boxes of shape (K, 5),
              in the format (x1, y1, x2, y2, score).
            - torch.Tensor: Image indices of shape (K,) corresponding to the
              selected boxes.
            - torch.Tensor: Landmark coordinates of shape (K, 5, 2), where each
              row contains (x, y) coordinates of the 5 predicted facial landmarks.

    Notes:
        - Returns empty tensors if no valid crops can be extracted.
        - Internally uses `_extract_onet_crops`, `_run_onet_model`, and
          `_postprocess_onet_outputs`.
        - This is the third and final stage in the MTCNN pipeline.
    """
    crops = _extract_onet_crops(imgs, boxes, image_inds)

    if not crops:
        return (
            torch.empty(0),
            torch.empty(0, dtype=torch.int),
            torch.zeros(0, 5, 2),
        )

    im_data = torch.cat(crops, dim=0)
    out0, out1, out2 = _run_onet_model(im_data, onet)

    return _postprocess_onet_outputs(
        boxes, image_inds, (out0, out1, out2), threshold
    )


def group_results_by_image(batch_size, boxes, points, image_inds):
    """
    Group detection results by image in the batch.

    Converts batched detection outputs (boxes and landmarks) from flat tensors
    into per-image grouped arrays based on their image indices. This is useful
    when detections across all batch images are concatenated and need to be
    reorganized into their respective image groups.

    Args:
        batch_size (int): The number of images in the batch.
        boxes (torch.Tensor): Tensor of shape (N, 5) with bounding boxes
            (x1, y1, x2, y2, score).
        points (torch.Tensor): Tensor of shape (N, 5, 2) with facial landmarks
            for each detected face.
        image_inds (torch.Tensor): Tensor of shape (N,) containing image index
            (in the range [0, batch_size-1]) for each detection.

    Returns:
        tuple:
            - np.ndarray: Array of length `batch_size`, where each element is
              an array of bounding boxes for that image.
            - np.ndarray: Array of length `batch_size`, where each element is
              an array of facial landmarks for that image.

    Notes:
        - The outputs are returned as `dtype=object` arrays, where each element
          is a NumPy array corresponding to one image.
        - This function assumes all inputs are on the same device and valid.
    """
    boxes, points, image_inds = (
        boxes.cpu().numpy(),
        points.cpu().numpy(),
        image_inds.cpu().numpy(),
    )
    batch_boxes, batch_points = [], []
    for b in range(batch_size):
        idx = np.where(image_inds == b)
        batch_boxes.append(boxes[idx].copy())
        batch_points.append(points[idx].copy())
    return np.array(batch_boxes, dtype=object), np.array(
        batch_points, dtype=object
    )


def generate_bounding_box(reg, probs, scale, thresh):
    """
    Generate bounding boxes from the feature map of the Proposal Network (P-Net).

    This function identifies locations on the feature map where the face
    classification score exceeds a threshold, computes the corresponding bounding
    boxes in the original image space, and attaches the regression offsets for later
    refinement.

    Args:
        reg (torch.Tensor): Regression output from P-Net of shape (4, H, W).
        probs (torch.Tensor): Face classification scores of shape (H, W), typically
            the second channel from P-Net output (face probability map).
        scale (float): The scaling factor applied to the original image to reach
            the current pyramid level.
        thresh (float): Confidence threshold to filter out low-scoring boxes.

    Returns:
        tuple:
            - boundingbox (torch.Tensor): A tensor of shape (N, 9) with bounding box
              coordinates (x1, y1, x2, y2), score, and 4 regression offsets.
            - image_inds (torch.Tensor): A tensor of shape (N,) with the image index
              for each bounding box (used when processing batches).

    Notes:
        - Uses a sliding window approach with stride 2 and cell size 12.
        - Bounding boxes are mapped from the feature map to the original image scale.
        - Regression values are used later to refine the box coordinates.
    """
    stride = 2
    cellsize = 12

    reg = reg.permute(1, 0, 2, 3)

    mask = probs >= thresh
    mask_inds = mask.nonzero()
    image_inds = mask_inds[:, 0]
    score = probs[mask]
    reg = reg[:, mask].permute(1, 0)
    bb = mask_inds[:, 1:].type(reg.dtype).flip(1)
    q1 = ((stride * bb + 1) / scale).floor()
    q2 = ((stride * bb + cellsize - 1 + 1) / scale).floor()
    boundingbox = torch.cat([q1, q2, score.unsqueeze(1), reg], dim=1)
    return boundingbox, image_inds


def nms_numpy(boxes, scores, threshold, method):
    """
    Perform Non-Maximum Suppression (NMS) on bounding boxes using NumPy.

    Suppresses overlapping bounding boxes based on Intersection-over-Union (IoU)
    or Minimum-overlap criteria. Keeps the highest-scoring boxes while discarding
    others with high overlap.

    Args:
        boxes (np.ndarray): Array of shape (N, 4) or (N, ≥4) containing bounding
            boxes in the format (x1, y1, x2, y2[, ...]).
        scores (np.ndarray): Array of shape (N,) with confidence scores for each box.
        threshold (float): Overlap threshold for suppression. Boxes with IoU/Min-overlap
            greater than this will be discarded.
        method (str): Suppression method. Either:
            - "Min": Uses min(area_i, area_j) as denominator.
            - "Union": Uses area_i + area_j - intersection as denominator (standard IoU).

    Returns:
        np.ndarray: Array of selected box indices (int32 dtype), sorted by descending score.

    Notes:
        - This function assumes the boxes are not batched by image. Use image-level NMS
          for batched processing.
        - If `boxes` is empty, returns an empty array of shape (0, 3).
        - The "Min" method is typically used in MTCNN for tighter suppression.
    """
    if boxes.size == 0:
        return np.empty((0, 3))

    coords = boxes[:, :4]
    area = (coords[:, 2] - coords[:, 0] + 1) * (
        coords[:, 3] - coords[:, 1] + 1
    )

    order = np.argsort(scores)
    keep = []

    while order.size > 0:
        keep.append(order[-1])

        xx1 = np.maximum(coords[order[-1]][0], coords[order[:-1]][:, 0])
        yy1 = np.maximum(coords[order[-1]][1], coords[order[:-1]][:, 1])
        xx2 = np.minimum(coords[order[-1]][2], coords[order[:-1]][:, 2])
        yy2 = np.minimum(coords[order[-1]][3], coords[order[:-1]][:, 3])

        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)

        denom = (
            np.minimum(area[order[-1]], area[order[:-1]])
            if method == "Min"
            else (area[order[-1]] + area[order[:-1]] - inter)
        )
        overlap = inter / denom

        order = order[np.where(overlap <= threshold)]

    return np.array(keep, dtype=np.int32)


def batched_nms_numpy(boxes, scores, idxs, threshold, method):
    """
    Perform batched Non-Maximum Suppression (NMS) using NumPy for grouped indices.

    Applies NMS separately per group (e.g., per image or class) by offsetting
    boxes with a large constant to prevent cross-group suppression. This is useful
    when applying NMS on concatenated boxes from different images or classes.

    Args:
        boxes (torch.Tensor): Tensor of shape (N, 4) representing bounding boxes.
        scores (torch.Tensor): Tensor of shape (N,) with confidence scores.
        idxs (torch.Tensor): Tensor of shape (N,) indicating the group index
            (e.g., image ID or class ID) for each box.
        threshold (float): Overlap threshold for suppression.
        method (str): NMS method to use, either:
            - "Min": Minimum-overlap denominator (used in MTCNN).
            - "Union": Standard IoU denominator.

    Returns:
        torch.Tensor: Indices of boxes that are kept after NMS, as a 1D LongTensor.

    Notes:
        - Internally adds large coordinate offsets to each group to prevent
          suppression across groups.
        - Relies on `nms_numpy` to do the actual suppression.
        - Returns indices on the same device as input tensors.
        - Assumes boxes are in the format (x1, y1, x2, y2).
    """
    device = boxes.device
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=device)
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    boxes_for_nms = boxes_for_nms.cpu().numpy()
    scores = scores.cpu().numpy()
    keep = nms_numpy(boxes_for_nms, scores, threshold, method)
    return torch.as_tensor(keep, dtype=torch.long, device=device)


def extract_face(img, box, image_size, margin_value, save_path=None):
    """
    Extract a face region (optionally with margin) from a PIL image and return it as a tensor.

    Crops the region specified by the bounding box, adds an optional margin, resizes the
    result to a square image of the specified size, and converts it to a float tensor.
    Optionally saves the extracted face to disk.

    Args:
        img (PIL.Image.Image): Input image from which to extract the face.
        box (tuple[float]): A 4-element array specifying the bounding box (x1, y1, x2, y2).
        image_size (int, optional): Target output size (height and width). Default is 160.
        margin_value (int, optional): Number of margin pixels to add around the bounding box
            in the output image space. Default is 0.
        save_path (str, optional): File path to save the extracted face image. Default is None.

    Returns:
        torch.Tensor: A float tensor of shape (3, image_size, image_size) representing the face.

    Notes:
        - The margin is computed relative to the resized image size, not the original image size.
        - If `save_path` is provided, the extracted face will be saved using `save_img()`.
        - Supports PIL input and returns a PyTorch tensor ready for model input.
    """
    margin = [
        margin_value * (box[2] - box[0]) / (image_size - margin_value),
        margin_value * (box[3] - box[1]) / (image_size - margin_value),
    ]
    raw_image_size = get_size(img)
    box = (
        float(max(box[0] - margin[0] / 2, 0)),
        float(max(box[1] - margin[1] / 2, 0)),
        float(min(box[2] + margin[0] / 2, raw_image_size[0])),
        float(min(box[3] + margin[1] / 2, raw_image_size[1])),
    )

    face = crop_resize(img, box, image_size)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) + "/", exist_ok=True)
        save_img(face, save_path)

    if isinstance(face, np.ndarray):
        face = F.to_tensor(face.astype(np.float32))
    elif isinstance(face, Image.Image):
        face = F.to_tensor(face)
    elif isinstance(face, torch.Tensor):
        face = face.float()
    else:
        raise NotImplementedError
    return face


def detect_face(imgs, config, nets, device):
    """
    Perform face detection on a batch of images using MTCNN (PNet, RNet, ONet).

    This function processes input images through a three-stage cascade of
    convolutional neural networks to detect faces and facial landmarks.

    Args:
        imgs (Union[torch.Tensor, np.ndarray, list]): A batch of input images. Each image
            must be of the same size and can be a NumPy array, Torch tensor, or list of PIL images.
        config (dict): A dictionary with detection parameters:
            - "minsize" (int): Minimum face size.
            - "factor" (float): Scale factor for the image pyramid.
            - "threshold" (dict): Thresholds for each stage [PNet, RNet, ONet].
        nets (dict): A dictionary containing the trained models for each MTCNN stage:
            - "pnet": Proposal Network (PNet)
            - "rnet": Refinement Network (RNet)
            - "onet": Output Network (ONet)
        device (torch.device): The device (CPU/GPU) to run the models on.

    Returns:
        tuple:
            - batch_boxes (np.ndarray): Array of shape (batch_size,) where each element is
              an array of detected face bounding boxes for the corresponding image.
            - batch_points (np.ndarray): Array of shape (batch_size,) where each element is
              an array of detected facial landmarks of shape (num_faces, 5, 2).

    Notes:
        - The function handles batch processing, but assumes all images are the same size.
        - Returns empty arrays if no faces are detected in any of the stages.
        - Output bounding boxes are in (x1, y1, x2, y2, confidence) format.
        - Facial landmarks are in the format (5 points × 2 coordinates).
    """
    imgs = preprocess_images(imgs, device)
    batch_size, _, h, w = imgs.shape

    scales = build_pyramid(
        h=h, w=w, minsize=config["minsize"], factor=config["factor"]
    )

    # First stage: PNet
    boxes, image_inds = run_pnet(
        imgs, scales, nets["pnet"], config["threshold"]["pnet"]
    )

    if len(boxes) == 0:
        return empty_results(batch_size)

    boxes = calibrate_and_pad_boxes(boxes=boxes)

    # Second stage: RNet
    boxes, image_inds = run_rnet(
        imgs, boxes, image_inds, nets["rnet"], config["threshold"]["rnet"]
    )

    if len(boxes) == 0:
        return empty_results(batch_size)

    # Third stage: ONet
    boxes, image_inds, points = run_onet(
        imgs, boxes, image_inds, nets["onet"], config["threshold"]["onet"]
    )

    return group_results_by_image(batch_size, boxes, points, image_inds)
