from leap_binder import (
    input_encoder, preprocess_func_leap, gt_encoder, measure_sharpness_from_image, get_iou,
    get_accuracy, leap_binder, yolov5_loss, metadata_sample_index, gt_bb_decoder, image_visualizer, bb_decoder
)
import tensorflow as tf
import matplotlib
matplotlib.use('MacOSX')
import numpy as np
from code_loader.helpers.visualizer.visualize import visualize
from code_loader.contract.datasetclasses import SamplePreprocessResponse


def check_custom_test():
    check_generic = False
    if check_generic:
        leap_binder.check()
    print("started custom tests")

    # load the model
    model_path = r"weights/yolov5s-visdrone.h5"
    cnn = tf.keras.models.load_model(model_path)

    responses = preprocess_func_leap()
    for subset in responses:
        for idx in range(10):
            image = input_encoder(idx, subset)

            preds = cnn([np.expand_dims(image, axis=0)])
            gt = gt_encoder(idx, subset)

            img = image_visualizer(np.expand_dims(image, 0))
            image_with_bbox = bb_decoder(np.expand_dims(image, 0), preds[0].numpy())
            image_with_gt_bbox = gt_bb_decoder(np.expand_dims(image, 0),np.expand_dims(gt, 0))

            visualize(img)
            visualize(image_with_bbox)
            visualize(image_with_gt_bbox)

            d_loss=yolov5_loss(preds[1].numpy(), preds[2].numpy(), preds[3].numpy(), np.expand_dims(gt, 0), preds[0].numpy())
            sharpness = measure_sharpness_from_image(idx, subset)
            iou = get_iou(preds[0].numpy(), SamplePreprocessResponse(np.array(idx), subset))
            acc = get_accuracy(preds[0].numpy(), SamplePreprocessResponse(np.array(idx), subset))
            metadata_sample=metadata_sample_index(idx,subset)
    print("finish tests")


if __name__ == '__main__':
    check_custom_test()
