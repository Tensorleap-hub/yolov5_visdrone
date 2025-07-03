from leap_binder import (
    input_encoder, preprocess_func_leap, gt_encoder, sample_metadata, leap_binder, yolov5_loss,
    gt_bb_decoder, image_visualizer, bb_decoder, get_per_sample_metrics
)
import matplotlib
import numpy as np
from code_loader.helpers.visualizer.visualize import visualize
from code_loader.contract.datasetclasses import SamplePreprocessResponse
import onnxruntime as ort


def check_custom_test():
    check_generic = False
    if check_generic:
        leap_binder.check()
    print("started custom tests")

    # load the model
    model_path = r"weights/yolov5s-visdrone.onnx"
    session = ort.InferenceSession(model_path)

    # Get model input name(s)
    input_name = session.get_inputs()[0].name

    responses = preprocess_func_leap()
    for subset in responses:
        for idx in range(2):
            image = input_encoder(idx, subset)

            preds = session.run(None, {input_name: np.expand_dims(image, axis=0)})
            gt = gt_encoder(idx, subset)

            img = image_visualizer(np.expand_dims(image, 0))
            image_with_bbox = bb_decoder(np.expand_dims(image, 0), preds[0])
            image_with_gt_bbox = gt_bb_decoder(np.expand_dims(image, 0),np.expand_dims(gt, 0))

            visualize(img)
            visualize(image_with_bbox)
            visualize(image_with_gt_bbox)

            d_loss=yolov5_loss(preds[1], preds[2], preds[3], np.expand_dims(gt, 0), preds[0])
            metadata = sample_metadata(idx, subset)
            metrics = get_per_sample_metrics(preds[0], SamplePreprocessResponse(np.array(idx), subset))
    print("finish tests")

if __name__ == '__main__':
    check_custom_test()
