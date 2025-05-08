import torch
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


def compute_iou_matrix(gt_bbox, preds_bbox):
    iou_mat = box_iou(gt_bbox, preds_bbox)
    if iou_mat.numel() == 0 or iou_mat.shape[1] == 0 or iou_mat.shape[0] == 0:
        return torch.zeros((1,1))

    return iou_mat * (iou_mat == iou_mat.max(dim=0, keepdim=True).values)