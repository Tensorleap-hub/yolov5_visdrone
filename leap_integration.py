import sys
import numpy as np
from config import cfg

from code_loader.contract.datasetclasses import PredictionTypeHandler
from code_loader.plot_functions.visualize import visualize
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_load_model, tensorleap_integration_test
from code_loader.contract.datasetclasses import PreprocessResponse, SamplePreprocessResponse
import onnxruntime as ort

from leap_binder import (
    input_encoder, preprocess_func_leap, gt_encoder, sample_metadata, 
    gt_bb_decoder, image_visualizer, bb_decoder, get_per_sample_metrics, yolov5_loss
)

sys.setrecursionlimit(10000)

# Define prediction type for object detection
# YOLOv5 outputs predictions with bounding boxes, objectness, and class scores
prediction_type1 = PredictionTypeHandler('object_detection', ['x', 'y', 'w', 'h', 'obj_conf'] + cfg["names"], channel_dim=1)

prediction_type2 = PredictionTypeHandler(name='concatenate_128', labels=[str(i) for i in range(128)], channel_dim=1)
prediction_type3 = PredictionTypeHandler(name='concatenate_64', labels=[str(i) for i in range(64)], channel_dim=1)
prediction_type4 = PredictionTypeHandler(name='concatenate_32', labels=[str(i) for i in range(31)], channel_dim=1)

@tensorleap_load_model([prediction_type1,prediction_type2,prediction_type3,prediction_type4])
def load_model():

    model_path = 'weights/yolov5s-visdrone.onnx'

    print("started custom tests")
    # yolo = tf.keras.models.load_model(model_path)
    yolo = ort.InferenceSession(model_path)
    return yolo

@tensorleap_integration_test()
def integration_test(idx, subset):
    """
    Integration test function that runs inference, calculates metrics, and visualizes results.
    
    Args:
        idx (int): Index of the sample to test.
        subset (PreprocessResponse): The dataset subset (train/val/test).
    """
    plot_vis = True
    yolo = load_model()

    # Get inputs and ground truth
    x = input_encoder(idx, subset)
    gt = gt_encoder(idx, subset)

    input_name = yolo.get_inputs()[0].name
    preds = yolo.run(None, {input_name: x})

    pred_main = preds[0]

    gt_input = np.expand_dims(gt, 0)  # add batch dim
    loss = yolov5_loss(preds[1], preds[2], preds[3], gt_input, pred_main)

    # Calculate metrics
    metrics = get_per_sample_metrics(pred_main, gt_input)

    # Visualizations
    img_vis = image_visualizer(x)
    image_with_pred_bbox = bb_decoder(x, pred_main)
    image_with_gt_bbox = gt_bb_decoder(x, gt)

    if plot_vis:
        visualize(img_vis)
        visualize(image_with_pred_bbox)
        visualize(image_with_gt_bbox)

    # Print metadata
    metadata = sample_metadata(idx, subset)

if __name__ == "__main__":
    # Get train and val subsets from preprocess function
    # preprocess_func_leap returns [train, val, test]
    subsets = preprocess_func_leap()
    train_subset = subsets[0]  # train
    val_subset = subsets[1]    # val
    
    # Run integration test on first sample of training set
    integration_test(0, train_subset)

