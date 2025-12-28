import cv2
import torch
import textwrap
import numpy as np
from config import cfg
from typing import List

from utils.loss import ComputeLoss
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
from leap_utils import load_model, compute_iou, compute_accuracy
from leap_config import CONFIG, abs_path_from_root


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
    data_yaml_path = abs_path_from_root(CONFIG["data_yaml_path"])
    data = check_dataset(data_yaml_path, autodownload=True)

    responses = []
    for split in ['train', 'val', 'test']:
        _, dataset = create_dataloader(
                data[split],
                imgsz = CONFIG["image_size"],
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

        cls = np.expand_dims(labels[:,0], axis=1)
        x = np.expand_dims(labels[:,1], axis=1)
        y = np.expand_dims(labels[:,2], axis=1)
        w = np.expand_dims(labels[:, 3], axis=1)
        h = np.expand_dims(labels[:,4], axis=1)

        if not preprocessing.data.rect:
            # Get the original width and height of the image at index i
            original_w, original_h = preprocessing.data.shapes[i]
            # Calculate the new image height after resizing the width to img_size
            new_h = original_h * img_size / original_w
            # Compute the padding size required to make the final image square (only vertical padding is considered)
            pad_size = img_size - new_h

            assert original_w >= original_h, "Only horizontal images are currently supported when dataloader's rect is False, since padding is assumed to be vertical only"

            # Adjust the vertical coordinate y to account for resizing and vertical padding
            y = y * new_h + pad_size / 2 # scale y to new height and add half of the total vertical padding
            y = y / img_size             # normalize y to the range [0, 1]
            # Scale the height h based on the resized image height and normalize it
            h = h * new_h / img_size

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
        "bbox area min": float(bbox_areas.min() if len(bbox_areas) > 0 else np.nan),
        "bbox area max": float(bbox_areas.max() if len(bbox_areas) > 0 else np.nan),
        "bbox area var": float(bbox_areas.var()),
    })
    return metadata_dict

# ------------------------------
# Custom Loss
# ------------------------------
def yolov5_loss_factory(num_anchors):
    # Build predictions list
    preds_list = ', '.join([f'pred{i}' for i in range(num_anchors)])
    all_args = f'{preds_list}, gt, demo_pred'

    # Dynamically generate function code
    fn_code = f'''
    @tensorleap_custom_loss("yolov5_loss")
    def yolov5_loss({all_args}):
        preds = [torch.from_numpy(p) for p in [{preds_list}]]
        gt_torch = torch.from_numpy(gt).squeeze(0)
        gt_torch = torch.cat([torch.zeros_like(gt_torch[:, 1]).unsqueeze(1), gt_torch], dim=1)
        loss = yolov5_loss_compute(preds, gt_torch)[0]
        return loss.numpy()
    '''
    local_ns = {}
    exec(textwrap.dedent(fn_code), globals(), local_ns)
    return local_ns['yolov5_loss']

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
# 2. N raw prediction outputs used for computing loss

# Bind the object detection output for visualization/interpretation
# - This tensor contains bounding box predictions before NMS
# - Shape: (Batch, Prediction scores, Num_BBoxes)
# - Prediction scores contain the following scores:
#   ["x", "y", "w", "h", "obj_conf"] + class names from cfg["names"]
# - 'channel_dim=1' indicates that the prediction scores are arranged along dimension 1
leap_binder.add_prediction(
    name='object detection',
    labels=["x", "y", "w", "h", "obj_conf"] + cfg["names"],
    channel_dim=-1
)

torch_model = load_model(CONFIG["torch_model_weights_name"])
yolov5_loss_compute = ComputeLoss(torch_model)
yolov5_loss = yolov5_loss_factory(yolov5_loss_compute.na)

if __name__ == '__main__':
    leap_binder.check()
