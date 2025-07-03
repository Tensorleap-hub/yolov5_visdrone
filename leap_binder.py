import cv2
import torch
import numpy as np
from config import cfg
from typing import List
from pathlib import Path

from code_loader import leap_binder
from utils.dataloaders import create_dataloader
from utils.general import check_dataset, colorstr
from leap_utils import compute_precision_recall_f1
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
from leap_utils import create_loss, compute_iou, compute_accuracy

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
    data = check_dataset(data_path, autodownload=False)

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
                shuffle=False,
            )

        responses.append(PreprocessResponse(data=dataset, length=(len(dataset))))
    return responses

@tensorleap_input_encoder('image', channel_dim=1)
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
    return image


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

@tensorleap_metadata('metadata')
def sample_metadata(idx: int, preprocessing: PreprocessResponse) -> dict:
    """
    Computes the sample's metadata.

    Args:
        idx (int): Index of the image.
        preprocessing (PreprocessResponse): Dataset wrapper.

    Returns:
        dict: Dictionary with metadata values.
    """
    sample = preprocessing.data[idx]
    image = (sample[0].numpy().transpose(1,2,0)*255).astype(np.uint8)
    gt = sample[1].numpy()

    if gt.shape[0] != 0:
        gt_class = gt[:, 1]
        gt_bbox = gt[:,2:]
        bbox_areas = gt_bbox[:,2]*gt_bbox[:,3]
    else:
        gt_class, bbox_areas = np.array([]), np.array([])

    unique_classes, counts = np.unique(gt_class, return_counts=True)

    metadata_dict = {}

    laplacian = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F) # Compute Laplacian variance
    sharpness = laplacian.var()
    metadata_dict.update({
        "image_sharpness": float(sharpness),
        "# of objects": gt.shape[0],
        "# of unique objects": len(unique_classes),
        "bbox area mean": float(bbox_areas.mean()),
        "bbox area median": float(np.median(bbox_areas)),
        "bbox area min": float(bbox_areas.min()),
        "bbox area max": float(bbox_areas.max()),
        "bbox area var": float(bbox_areas.var()),
    })
    return metadata_dict

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
    # preds = [torch.from_numpy(pred.transpose(0,4,1,2,3)) for pred in (pred0, pred1, pred2)]
    preds = [torch.from_numpy(pred) for pred in (pred0, pred1, pred2)]

    gt = torch.from_numpy(gt).squeeze(0)
    gt = torch.cat([torch.zeros_like(gt[:,1]).unsqueeze(1), gt], dim=1) # Add "batch idx" column for the loss
    loss = compute_loss(preds, gt)[0] # compute_loss returns a tuple, the full loss is the first item
    loss = loss.unsqueeze(0) # Add batch dimension
    return loss.numpy()

# ------------------------------
# Visualizers
# ------------------------------

@tensorleap_custom_visualizer('image_visualizer', LeapDataType.Image)
def image_visualizer(image: np.ndarray) -> LeapImage:
    """
    Returns a LeapImage for visualization.

    Args:
        image (np.ndarray): Input image array.

    Returns:
        LeapImage: Visualizable image object.
    """
    image = image.squeeze(0)
    image = image.transpose(1,2,0) # LeapImage visualizer expects inputs as channel last.
    return LeapImage((image*255).astype(np.uint8), compress=False)


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
    image = image.squeeze(0)
    image = image.transpose(1, 2, 0)  # LeapImageWithBBox visualizer expects inputs as channel last.
    image = (image*255).astype(np.uint8)
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
    preds = non_max_suppression(torch.from_numpy(predictions))[0].numpy()
    preds = xyxy2xywh(preds)

    image = image.squeeze(0)
    image = image.transpose(1, 2, 0)  # LeapImageWithBBox visualizer expects inputs as channel last.
    image = (image * 255).astype(np.uint8)
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

@tensorleap_custom_metric(name="per_sample_metrics", direction=MetricDirection.Upward)
def get_per_sample_metrics(y_pred: np.ndarray, preprocess: SamplePreprocessResponse):
    """
    Calculates metrics per sample on the model's prediction

    Args:
        y_pred (np.ndarray): Prediction from model.
        preprocessing (PreprocessResponse): Dataset wrapper.

    Returns:
        dict: Dictionary with metric values.
    """
    def _make_metrics(precision, recall, f1, iou, accuracy):
        return {
            "precision": np.array([precision], dtype=np.float32),
            "recall": np.array([recall], dtype=np.float32),
            "f1": np.array([f1], dtype=np.float32),
            "iou": np.array([iou], dtype=np.float32),
            "accuracy": np.array([accuracy], dtype=np.float32),
        }

    dataloader = preprocess.preprocess_response.data
    gt = dataloader[int(preprocess.sample_ids)][1] # shape: [N, 6] (_,label,x,y,w,h)
    preds = non_max_suppression(torch.from_numpy(y_pred))[0]

    if gt.shape[0] == 0 and preds.shape[0] == 0:
        return _make_metrics(1, 0, 0, 1, 1) # Edge case: no objects, assume perfect

    if preds.shape[0] == 0:
        return _make_metrics(0, 0, 0, 0, 0)  # No predictions at all

    if gt.shape[0] == 0:
        return _make_metrics(0, 0, 0, 0, 0) # No GT but has predictions

    preds_boxes = preds[:, :4] / dataloader.img_size # normalize to be [0,1]
    preds_labels = preds[:, 5]

    gt_boxes = xywh2xyxy(gt[:, 2:])
    gt_labels = gt[:, 1]

    p, r, f1 = compute_precision_recall_f1(gt_boxes, preds_boxes, iou_threshold=0.5)
    iou = compute_iou(gt_boxes, preds_boxes)
    acc = compute_accuracy(gt_boxes, gt_labels, preds_boxes, preds_labels)

    return _make_metrics(float(p), float(r), float(f1), float(iou), float(acc))

# ------------------------------
# Prediction Binding
# ------------------------------
# The model outputs a list of 4 tensors:
# 1. Processed object detection results for visualization
# 2. 3 raw prediction outputs used for computing loss

# Bind the object detection output for visualization/interpretation
# - This tensor contains bounding box predictions after NMS
# - Shape: (Batch, Prediction scores, Num_BBoxes)
# - Prediction scores contain the following scores:
#   ["x", "y", "w", "h", "obj_conf"] + class names from cfg["names"]
# - 'channel_dim=1' indicates that the prediction scores are arranged along dimension 1
leap_binder.add_prediction(
    name='object detection',
    labels=["x", "y", "w", "h", "obj_conf"] + cfg["names"],
    channel_dim=1
)

# Bind intermediate feature outputs for analysis or debugging.
leap_binder.add_prediction(name='concatenate_128', labels=[str(i) for i in range(128)], channel_dim=2)
leap_binder.add_prediction(name='concatenate_64', labels=[str(i) for i in range(64)], channel_dim=2)
leap_binder.add_prediction(name='concatenate_32', labels=[str(i) for i in range(32)], channel_dim=2)

if __name__ == '__main__':
    leap_binder.check()
