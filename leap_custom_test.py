from leap_binder import (
    input_encoder, preprocess_func_leap, gt_encoder, sample_metadata, leap_binder, yolov5_loss,
    gt_bb_decoder, image_visualizer, bb_decoder, get_per_sample_metrics, confusion_matrix_metric
)
import numpy as np
from code_loader.helpers.visualizer.visualize import visualize
from code_loader.contract.datasetclasses import SamplePreprocessResponse
import onnxruntime as ort
import matplotlib

matplotlib.use('TkAgg')


def check_custom_test():
    check_generic = False
    if check_generic:
        leap_binder.check()
    print("started custom tests")

    # load the model
    # model_path = r"/home/aime/tensorleap/new_best.onnx"
    model_path = r"/Users/amitcohen/projects/yolov5/weights/yolov5s-visdrone.onnx"
    session = ort.InferenceSession(model_path)

    # Get model input name(s)
    input_name = session.get_inputs()[0].name

    responses = preprocess_func_leap()
    import time
    curr = time.time()
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

            anchor_preds = preds[1:]  # predictions from anchors
            gt_input = np.expand_dims(gt, 0)  # ground truth, add batch dim
            main_pred = preds[0]
            d_loss=yolov5_loss(*anchor_preds, gt_input, main_pred)
            metadata = sample_metadata(idx, subset)
            metrics = get_per_sample_metrics(main_pred, gt_input)
            confusion_matrix = confusion_matrix_metric(main_pred, gt_input)
    end = time.time()
    print("finish tests")
    print(f"tests took {end - curr} seconds")

if __name__ == '__main__':
    check_custom_test()
