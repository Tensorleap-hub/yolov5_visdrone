import torch
import numpy as np
from pathlib import Path
from utils.loss import ComputeLoss
from models.experimental import attempt_load
from ultralytics.utils.metrics import box_iou

def export_onnx(pytorch_weights_path=Path(__file__).resolve().parent / "weights/yolov5s-visdrone.pt", onnx_path=None):
    model = attempt_load(pytorch_weights_path, device='cpu')
    input = torch.rand(1,3,1024,1024)
    if not onnx_path:
        pytorch_weights_path = Path(pytorch_weights_path)
        onnx_path = pytorch_weights_path.with_suffix("onnx")
    try:
        torch.onnx.export(model,input,onnx_path,
                    input_names=['images'],
                    output_names=['output'],
                    dynamic_axes={
                                'images': {
                                    0: 'batch',
                                    2: 'height',
                                    3: 'width'},
                                'output': {
                                    0: 'batch',
                                    1: 'anchors'}
                            }
        )
        print(f"Exported onnx model to {onnx_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def create_loss():
    torch_weights_path = Path(__file__).resolve().parent / "weights/yolov5s-visdrone.pt"
    model = torch.load(torch_weights_path, map_location='cpu')["model"]
    compute_loss = ComputeLoss(model)  # create loss calculator
    return compute_loss

def compute_iou(gt_bbox, preds_bbox):
    iou_mat = box_iou(gt_bbox, preds_bbox)
    if iou_mat.numel() == 0 or iou_mat.shape[1] == 0 or iou_mat.shape[0] == 0:
        return np.zeros((1,1))
    max_iou = iou_mat.max(dim=0, keepdim=True).values
    filtered_iou = iou_mat * iou_mat.eq(max_iou)
    return filtered_iou.max(dim=1).values.numpy().mean()

def compute_accuracy(gt_bbox, gt_labels, preds_bbox, preds_labels):
    iou_mat = box_iou(gt_bbox, preds_bbox)
    if iou_mat.numel() == 0 or iou_mat.shape[1] == 0 or iou_mat.shape[0] == 0:
        return np.zeros((1, 1))
    max_iou = iou_mat.max(dim=0, keepdim=True).values
    filtered_iou = iou_mat * iou_mat.eq(max_iou)
    succ = (preds_labels[filtered_iou.max(dim=1)[1].numpy()] == gt_labels).numpy()
    return succ.mean()

def compute_precision_recall_f1(gt_boxes, pred_boxes, iou_threshold=0.5):
    iou_mat = box_iou(gt_boxes, pred_boxes)  # Shape: (num_gt, num_pred)

    matched_gt = set()
    matched_pred = set()
    TP = 0

    # Loop through all predictions and try to match to GT
    for pred_idx in range(iou_mat.shape[1]):
        gt_idx = iou_mat[:, pred_idx].argmax().item()
        max_iou = iou_mat[gt_idx, pred_idx].item()

        if max_iou >= iou_threshold and gt_idx not in matched_gt and pred_idx not in matched_pred:
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
            TP += 1

    FP = pred_boxes.shape[0] - TP
    FN = gt_boxes.shape[0] - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1