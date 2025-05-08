import cv2
import torch
import numpy as np
from config import cfg
from typing import List
from pathlib import Path

from code_loader import leap_binder
from utils.dataloaders import create_dataloader
from utils.general import check_dataset, colorstr
from code_loader.contract.responsedataclasses import BoundingBox
from code_loader.visualizers.default_visualizers import LeapImage
from utils.general import non_max_suppression, xyxy2xywh, xywh2xyxy
from code_loader.contract.enums import LeapDataType, MetricDirection
from code_loader.contract.visualizer_classes import LeapImageWithBBox
from code_loader.contract.datasetclasses import PreprocessResponse, SamplePreprocessResponse
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_preprocess, tensorleap_gt_encoder, tensorleap_input_encoder, tensorleap_custom_metric,
    tensorleap_metadata, tensorleap_custom_loss, tensorleap_custom_visualizer
)
from leap_utils import create_loss, compute_iou_matrix

compute_loss = create_loss()

# ------------------------------
# Preprocessing and Encoders
# ------------------------------

@tensorleap_preprocess()
def preprocess_func_leap() -> List[PreprocessResponse]:
    """
    Loads datasets for 'train', 'val', and 'test' splits and wraps them in PreprocessResponse objects.

    Returns:
        List[PreprocessResponse]: List of datasets prepared for further processing.
    """
    data_path = Path(__file__).resolve().parent / 'data/VisDrone.yaml'
    data = check_dataset(data_path)

    imgsz = 1024 # Follow the train protocol
    responses = []
    for split in ['train', 'val', 'test']:
        _, dataset = create_dataloader(
                data[split],
                imgsz,
                batch_size=1,
                stride=32,
                single_cls=False,
                rect=False,
                workers=1,
                prefix=colorstr(f"{split}: "),
            )

        responses.append(PreprocessResponse(data=dataset, length=(len(dataset))))
    return responses

@tensorleap_input_encoder('image')
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    """
    Retrieves and normalizes an image from the dataset.

    Args:
        idx (int): Index of the image.
        preprocess (PreprocessResponse): Dataset wrapper.

    Returns:
        np.ndarray: Normalized image array.
    """
    image = preprocess.data[idx][0].numpy().astype(np.float32)/255
    return image.transpose(1,2,0)


@tensorleap_gt_encoder('classes')
def gt_encoder(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    """
    Extracts and adjusts bounding boxes for the specified image index.

    Args:
        idx (int): Image index.
        preprocessing (PreprocessResponse): Dataset wrapper with labels and shapes.

    Returns:
        np.ndarray: Array of adjusted bounding boxes in [cls, x, y, w, h] format.
    """
    mask = preprocessing.data.batch==idx
    img_size = preprocessing.data.img_size
    labels_arr = []
    for i, is_selected in enumerate(mask):
        if not is_selected:
            continue
        labels = preprocessing.data.labels[i] # shape: [N, 5] (label, x, y, w, h)
        original_w, original_h = preprocessing.data.shapes[i]
        new_h = original_h*img_size/original_w
        pad_size = img_size - new_h

        cls = np.expand_dims(labels[:,0], axis=1)
        x = np.expand_dims(labels[:,1], axis=1) # x_center stays the same
        y = np.expand_dims(labels[:,2], axis=1) * new_h + pad_size / 2
        y /= img_size
        w = np.expand_dims(labels[:, 3], axis=1)
        h = np.expand_dims(labels[:,4], axis=1) * new_h / img_size # scale height

        adjusted = np.concatenate([cls, x, y, w, h], axis=1)
        labels_arr.append(adjusted)

    return np.array(labels_arr,dtype=np.float32).squeeze(0)

# ------------------------------
# Metadata
# ------------------------------

@tensorleap_metadata('metadata_sample_index')
def metadata_sample_index(idx: int, preprocess: PreprocessResponse) -> int:
    """
    Returns the sample index as metadata.

    Args:
        idx (int): Sample index.
        preprocess (PreprocessResponse): Not used here.

    Returns:
        int: The same index.
    """
    return idx

@tensorleap_metadata('sharpness')
def measure_sharpness_from_image(idx: int, preprocessing: PreprocessResponse) -> dict:
    """
    Computes the sharpness of the image using Laplacian variance.

    Args:
        idx (int): Index of the image.
        preprocessing (PreprocessResponse): Dataset wrapper.

    Returns:
        dict: Dictionary with sharpness value.
    """
    image = (preprocessing.data[idx][0].numpy().transpose(1,2,0)*255).astype(np.uint8)
    laplacian = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F) # Compute Laplacian variance
    variance = laplacian.var()
    return {"value": variance}

# ------------------------------
# Custom Loss
# ------------------------------

@tensorleap_custom_loss("yolov5_loss")
def yolov5_loss(pred0: np.ndarray, pred1: np.ndarray, pred2: np.ndarray, gt: np.ndarray, demo_pred: np.ndarray):
    """
    Computes YOLOv5-style object detection loss.

    Args:
        pred0, pred1, pred2 (np.ndarray): Prediction tensors for each detection scale.
        gt (np.ndarray): Ground truth bounding boxes.
        demo_pred (np.ndarray): Not used in loss computation. Added due to technical Tensorleap reason

    Returns:
        np.ndarray: Loss scalar.
    """
    preds = [torch.from_numpy(pred.transpose(0,4,1,2,3)) for pred in (pred0, pred1, pred2)]

    gt = torch.from_numpy(gt).squeeze(0)
    gt = torch.cat([torch.zeros_like(gt[:,1]).unsqueeze(1), gt], dim=1) # Add "batch idx" column for the loss
    return compute_loss(preds, gt)[0].numpy()

# ------------------------------
# Visualizers
# ------------------------------

@tensorleap_custom_visualizer('image_visualizer', LeapDataType.Image)
def image_visualizer(image: np.ndarray) -> LeapImage:
    """
    Returns a LeapImage without compression for visualization.

    Args:
        image (np.ndarray): Input image array.

    Returns:
        LeapImage: Visualizable image object.
    """
    return LeapImage((image.squeeze(0)*255).astype(np.uint8), compress=False)


@tensorleap_custom_visualizer("bb_gt_decoder", LeapDataType.ImageWithBBox)
def gt_bb_decoder(image: np.ndarray, bb_gt: np.ndarray) -> LeapImageWithBBox:
    """
    Overlays ground truth bounding boxes on the image.

    Args:
        image (np.ndarray): Input image.
        bb_gt (np.ndarray): Ground truth bounding boxes.

    Returns:
        LeapImageWithBBox: Image with bounding boxes drawn.
    """
    image = (image.squeeze(0)*255).astype(np.uint8)
    bboxes = [
        BoundingBox(
            x=bbx[1],
            y=bbx[2],
            width=bbx[3],
            height=bbx[4],
            confidence=1.,
            label=cfg["names"][int(bbx[0])] if not np.isnan(bbx[0]) else 'Unknown Class'
        )
        for bbx in bb_gt.squeeze(0)
    ]
    return LeapImageWithBBox(data=image, bounding_boxes=bboxes)

@tensorleap_custom_visualizer("bb_decoder", LeapDataType.ImageWithBBox)
def bb_decoder(image: np.ndarray, predictions: np.ndarray) -> LeapImageWithBBox:
    """
    Overlays predicted bounding boxes on the image after NMS and format conversion.

    Args:
        image (np.ndarray): Input image.
        predictions (np.ndarray): Raw prediction tensor.

    Returns:
        LeapImageWithBBox: Image with predicted bounding boxes.
    """
    # Convert raw predictions into xyxy bboxes
    preds = non_max_suppression(torch.from_numpy(predictions.transpose(0,2,1)))[0].numpy()
    preds = xyxy2xywh(preds)
    image = (image.squeeze(0) * 255).astype(np.uint8)
    h, w, _ = image.shape

    bboxes = [
        BoundingBox(
            x=pred[0]/w,
            y=pred[1]/h,
            width=pred[2]/w,
            height=pred[3]/h,
            confidence=pred[4],
            label=cfg["names"][int(pred[5])] if not np.isnan(pred[5]) else 'Unknown Class'
        )
        for pred in preds
    ]
    return LeapImageWithBBox(data=image, bounding_boxes=bboxes)

# ------------------------------
# Custom Metrics
# ------------------------------

@tensorleap_custom_metric("ious", direction=MetricDirection.Upward)
def get_iou(y_pred: np.ndarray, preprocess: SamplePreprocessResponse):
    """
    Computes mean Intersection over Union (IoU) between predicted and ground truth bounding boxes for a sample.

    Args:
        y_pred (np.ndarray): Predicted bounding boxes in YOLO format.
        preprocess (SamplePreprocessResponse): Contains access to original ground truth data and metadata.

    Returns:
        np.ndarray: Single-element array with mean IoU for the sample.
    """
    dataloader = preprocess.preprocess_response.data
    gt_bbox = dataloader[int(preprocess.sample_ids)][1]
    gt_bbox = xywh2xyxy(gt_bbox[:,2:])

    preds = non_max_suppression(torch.from_numpy(y_pred.transpose(0, 2, 1)))[0]
    preds = preds[:,:4]/dataloader.img_size

    iou_mat = compute_iou_matrix(gt_bbox, preds)
    return np.expand_dims(iou_mat.max(dim=1).values.numpy().mean(),axis=0)


@tensorleap_custom_metric("accuracy", direction=MetricDirection.Upward)
def get_accuracy(y_pred: np.ndarray, preprocess: SamplePreprocessResponse):
    """
    Computes mean accuracy of the label classification between the most overlapping gt bbox and pred bbox of the sample.

    Args:
        y_pred (np.ndarray): Predicted bounding boxes in YOLO format.
        preprocess (SamplePreprocessResponse): Contains access to original ground truth data and metadata.

    Returns:
        np.ndarray: Single-element array with mean accuracy for the sample.
    """
    dataloader = preprocess.preprocess_response.data
    gt = dataloader[int(preprocess.sample_ids)][1]
    gt_bbox = xywh2xyxy(gt[:, 2:])
    gt_labels = gt[:, 1]

    preds = non_max_suppression(torch.from_numpy(y_pred.transpose(0, 2, 1)))[0]
    preds_bbox = preds[:, :4]/dataloader.img_size
    preds_labels = preds[:, 5]

    iou_mat = compute_iou_matrix(gt_bbox, preds_bbox)
    succ = (preds_labels[iou_mat.max(dim=1)[1].numpy()]==gt_labels).numpy()
    return np.expand_dims(succ.mean(),axis=0)

# ------------------------------
# Prediction Binding
# ------------------------------

leap_binder.add_prediction(
    name='object detection',
    labels=["x", "y", "w", "h", "obj_conf"] + cfg["names"],
    channel_dim=1
)
leap_binder.add_prediction(name='concatenate_128', labels=[str(i) for i in range(128)], channel_dim=1)
leap_binder.add_prediction(name='concatenate_64', labels=[str(i) for i in range(64)], channel_dim=1)
leap_binder.add_prediction(name='concatenate_32', labels=[str(i) for i in range(32)], channel_dim=1)

if __name__ == '__main__':
    leap_binder.check()
