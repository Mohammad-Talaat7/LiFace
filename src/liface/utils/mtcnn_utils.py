import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn.functional import interpolate

from liface import configurations as Config  # type: ignore


def empty_results(batch_size):
    """
    Create empty detection results for a batch.

    This function returns two arrays of empty NumPy arrays to represent
    placeholder results for detection tasks. Each element in the arrays corresponds
    to one item in the batch.

    Returns:
        tuple:
            - boxes (np.ndarray of object): An array of length `batch_size`,
              where each element is an empty array of shape (0, 5). This is
              typically used to hold detection boxes with scores and class labels.
            - keypoints (np.ndarray of object): An array of length `batch_size`,
              where each element is an empty array of shape (0, 5, 2). This is
              typically used to hold keypoint data for detected objects.

    Args:
        batch_size (int): The number of items in the batch. Determines the
            length of the returned arrays.
    """
    return _empty_results(batch_size=batch_size)


def _empty_results(batch_size):
    """
    Create empty detection results for a batch.
    """
    return np.array(
        [np.empty((0, 5)) for _ in range(batch_size)], dtype=object
    ), np.array([np.empty((0, 5, 2)) for _ in range(batch_size)], dtype=object)


def fixed_batch_process(im_data, model, batch_size=Config.MTCNN_BATCH_SIZE):
    """
        Process input data in fixed-size batches using a model.

        This function divides a large input tensor into smaller batches to avoid
        memory overload, passes each batch through the model, and concatenates
        the results. It is especially useful for inference on datasets that
        do not fit into memory all at once.

        Args:
            im_data (torch.Tensor): The input data tensor to be processed,
                typically with shape (N, ...), where N is the number of samples.
            model (Callable): A model (e.g., a neural network) that takes a batch
                of `im_data` and returns one or more tensors.
            batch_size (int, optional): The number of samples to process per batch.
                Defaults to 512.

        Returns:
     The Egyptian Tax Authority
    Read More
            tuple of torch.Tensor: A tuple of concatenated model outputs.
                Each output in the tuple has shape corresponding to the full input
                size after batching (e.g., `(N, ...)`).
    """
    return _fixed_batch_process(
        im_data=im_data, model=model, batch_size=batch_size
    )


def _fixed_batch_process(im_data, model, batch_size):
    """
    Process input data in fixed-size batches using a model.
    """
    out = []
    for i in range(0, len(im_data), batch_size):
        batch = im_data[i : (i + batch_size)]
        out.append(model(batch))

    return tuple(torch.cat(v, dim=0) for v in zip(*out))


def get_size(img):
    """
    Get the size (width, height) of an image, supporting multiple formats.

    Args:
        img (Union[np.ndarray, torch.Tensor, PIL.Image.Image]):
            The input image. Can be a NumPy array, a Torch tensor, or a PIL image.

    Returns:
        tuple: A (width, height) tuple representing the spatial size of the image.

    Notes:
        - For NumPy arrays and Torch tensors, assumes shape (H, W, C).
        - For PIL images, uses the built-in `.size` attribute.
    """
    return _get_size(img=img)


def _get_size(img):
    """
    Get the size (width, height) of an image, supporting multiple formats.
    """
    if isinstance(img, (np.ndarray, torch.Tensor)):
        return img.shape[1::-1]
    return img.size


def save_img(img, path):
    """
    Save an image to disk, handling both NumPy arrays and PIL Images.

    Args:
        img (Union[np.ndarray, PIL.Image.Image]): The image to be saved. If a NumPy array,
            it is assumed to be in RGB format and will be converted to BGR for OpenCV.
        path (str): The destination file path (including filename and extension).

    Returns:
        None

    Notes:
        - For `np.ndarray`, uses `cv2.imwrite()` and converts RGB to BGR.
        - For PIL `Image`, uses `.save()` method.
    """
    _save_img(img=img, path=path)


def _save_img(img, path):
    """
    Save an image to disk, handling both NumPy arrays and PIL Images.
    """
    if isinstance(img, np.ndarray):
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        img.save(path)


def imresample(img, sz):
    """
    Resample an image tensor to a given size using area interpolation.

    This function resizes the input image tensor to the specified size using
    `torch.nn.functional.interpolate` with `"area"` mode, which is suitable
    for downsampling operations.

    Args:
        img (torch.Tensor): Input image tensor of shape (C, H, W) or
            (N, C, H, W), where:
            - C: number of channels
            - H: height
            - W: width
            - N: optional batch dimension
        sz (tuple): Desired output size as (height, width).

    Returns:
        torch.Tensor: Resampled image tensor of shape (C, sz[0], sz[1]) or
        (N, C, sz[0], sz[1]) depending on input shape.

    Notes:
        - Assumes input is a float tensor.
        - Area interpolation is most appropriate for downsampling tasks.
        - If input is not batched, you may need to `unsqueeze(0)` before and
          `squeeze(0)` after, depending on context.
    """
    return _imresample(img=img, sz=sz)


def _imresample(img, sz):
    """
    Resample an image tensor to a given size using area interpolation.
    """
    im_data = interpolate(img, size=sz, mode="area")
    return im_data


def pad(boxes, w, h):
    """
    Clip bounding box coordinates to lie within image boundaries.

    Ensures that all coordinates of the input bounding boxes do not exceed
    the image dimensions or fall below 1 (1-based indexing). This prevents
    out-of-bounds indexing when cropping regions from images.

    Args:
        boxes (torch.Tensor): Tensor of shape (N, 4) containing bounding boxes
            in the format (x1, y1, x2, y2).
        w (int): Width of the image.
        h (int): Height of the image.

    Returns:
        tuple:
            - y (np.ndarray): Clipped top y-coordinates.
            - ey (np.ndarray): Clipped bottom y-coordinates.
            - x (np.ndarray): Clipped left x-coordinates.
            - ex (np.ndarray): Clipped right x-coordinates.

    Notes:
        - Converts input boxes to integer NumPy arrays (after truncating decimals).
        - Clipping uses 1-based indexing (minimum value is 1).
        - Assumes input boxes are in pixel coordinates, possibly float values.
    """
    return _pad(boxes=boxes, w=w, h=h)


def _pad(boxes, w, h):
    """
    Clip bounding box coordinates to lie within image boundaries.
    """
    boxes = boxes.trunc().int().cpu().numpy()
    x = boxes[:, 0]
    y = boxes[:, 1]
    ex = boxes[:, 2]
    ey = boxes[:, 3]

    x[x < 1] = 1
    y[y < 1] = 1
    ex[ex > w] = w
    ey[ey > h] = h

    return y, ey, x, ex


def rerec(bbox_a):
    """
    Convert bounding boxes to square boxes while keeping centers fixed.

    Expands each bounding box to a square by setting the side length equal
    to the longer edge (width or height), and adjusting the coordinates to
    keep the center position unchanged.

    Args:
        bbox_a (torch.Tensor): Tensor of shape (N, 5) or (N, 4), where the first
            four columns represent bounding boxes in (x1, y1, x2, y2) format.

    Returns:
        torch.Tensor: Tensor of the same shape as `bbox_a`, with the first
        four columns adjusted to form square boxes.

    Notes:
        - Modifies the input tensor in-place.
        - Useful in face detection pipelines (like MTCNN) to ensure all crops
          have a uniform aspect ratio.
    """
    return _rerec(bbox_a=bbox_a)


def _rerec(bbox_a):
    """
    Convert bounding boxes to square boxes while keeping centers fixed.
    """
    h = bbox_a[:, 3] - bbox_a[:, 1]
    w = bbox_a[:, 2] - bbox_a[:, 0]

    l = torch.max(w, h)
    bbox_a[:, 0] = bbox_a[:, 0] + w * 0.5 - l * 0.5
    bbox_a[:, 1] = bbox_a[:, 1] + h * 0.5 - l * 0.5
    bbox_a[:, 2:4] = bbox_a[:, :2] + l.repeat(2, 1).permute(1, 0)

    return bbox_a


def crop_resize(img, box, image_size):
    """
    Crop and resize an image region to a fixed size.

    Handles multiple image formats (NumPy array, Torch tensor, or PIL Image)
    and crops the input using the given bounding box, then resizes the crop
    to a square of shape `(image_size, image_size)` using appropriate interpolation.

    Args:
        img (Union[np.ndarray, torch.Tensor, PIL.Image.Image]):
            The input image in NumPy, Torch, or PIL format.
        box (tuple): A bounding box (x1, y1, x2, y2) indicating the region to crop.
        image_size (int): Target size for the output image (width and height).

    Returns:
        Union[np.ndarray, torch.Tensor, PIL.Image.Image]: The cropped and resized image
        in the same type as the input.

    Notes:
        - For NumPy input: uses `cv2.resize` with `cv2.INTER_AREA`.
        - For Torch tensors: assumes shape (H, W, C) and uses `imresample()`.
        - For PIL images: uses `.crop()` and `.resize()` with bilinear interpolation.
    """
    return _crop_resize(img=img, box=box, image_size=image_size)


def _crop_resize(img, box, image_size):
    """
    Crop and resize an image region to a fixed size.
    """
    if isinstance(img, np.ndarray):
        img = img[box[1] : box[3], box[0] : box[2]]
        out = cv2.resize(
            img, (image_size, image_size), interpolation=cv2.INTER_AREA
        ).copy()
    elif isinstance(img, torch.Tensor):
        img = img[box[1] : box[3], box[0] : box[2]]
        out = (
            imresample(
                img.permute(2, 0, 1).unsqueeze(0).float(),
                (image_size, image_size),
            )
            .byte()
            .squeeze(0)
            .permute(1, 2, 0)
        )
    else:
        out = (
            img.crop(box)
            .copy()
            .resize((image_size, image_size), Image.BILINEAR)  # type: ignore # pylint: disable=no-member
        )
    return out
