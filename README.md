# General
This project demonstrates the integration of a YOLOv5 model with the VisDrone dataset on the Tensorleap platform. The integration logic is implemented in leap_binder.py, which handles data loading, metadata computation, loss definition, image and bounding box visualization, metrics, and more.

Before pushing the project to Tensorleap, it’s recommended to run the leap_custom_test.py script after setting up the data (as explained below). This allows you to verify that data loading and visualizations function as expected. Once the test passes, proceed to the deployment step.

# Data Handling
To replicate the data handling used in this project, refer to the VisDrone.yaml configuration file. The path field should point to the root directory of your dataset, which must be located within the Tensorleap-mounted folder (~/tensorleap). If you are creating your own yaml, you do not have to implement the download part as it is optional.  

In order to use your own yaml file, refer to "preprocess_func_leap" function within the leap_binder.py and change the data_path to point to the new yaml.

The train, val, and test fields should each reference the corresponding images subfolder within the dataset. Each of these directories (train, val, and test) must contain two subfolders:
	•	images: containing the image files
	•	labels: containing a .txt file for each image (e.g., 0000002_00005_d_0000014.jpg should have a corresponding 0000002_00005_d_0000014.txt)

Each line in the label file should represent a bounding box in the following format:
<label> <x> <y> <w> <h>
where x, y, w, and h are normalized values in the range [0, 1].


# Pushing Project to Tensorleap
To upload the code and model to the Tensorleap platform, navigate to the project directory and run the following command, specifying the path to the model weights file:  
```
leap project push weights/yolov5s-visdrone.onnx --transform-input true
```
Replace the path with your specific model weights file if different. When you have a new model and want to export it to onnx, you can use the ```export_onnx``` function in ```leap_utils.py```

# Leap Binder Overview 
The leap_binder.py file contains all the necessary functions to integrate the YOLOv5 model and VisDrone dataset with the Tensorleap platform. It defines how data is preprocessed, encoded, visualized, and evaluated within Tensorleap. Below is a breakdown of its key components:

### Preprocessing
* ```preprocess_func_leap()``` -
Loads and wraps the train, val, and test splits using the VisDrone.yaml file. It prepares the dataset into a format suitable for Tensorleap, returning a list of PreprocessResponse objects.

### Input and Ground Truth Encoding
* ```input_encoder()``` -
Converts the dataset image to a normalized NumPy array in channel-last (H, W, C) format, which Tensorleap requires.
* ```gt_encoder()``` -
Retrieves and adjusts bounding boxes for each image sample, performing image-size normalization to ensure accurate alignment.

### Metadata Computation
* ```sample_metadata()``` -
Extracts per-sample statistics like image sharpness (via Laplacian variance), number of objects, and bounding box area metrics (mean, median, min, max, variance). These metadata fields can be used for filtering, analysis, or grouping samples on the platform.

### Custom Loss
* ```yolov5_loss()``` -
Implements a YOLOv5-style loss function using the model’s raw prediction outputs across multiple scales. The loss is computed using Tensor-based operations and supports end-to-end training on Tensorleap.

### Visualizers
* ```image_visualizer()``` -
Displays the raw input image in the Tensorleap UI.
* ```bb_decoder()``` -
Overlays ground truth bounding boxes on the image for visual comparison and inspection.
* ```bb_decoder()``` -
Applies non-max suppression (NMS) and overlays the model’s predicted bounding boxes onto the image.

### Custom Metrics
* ```get_per_sample_metrics()``` - 
Computes standard object detection metrics per sample:
* 	Precision
* 	Recall
* 	F1 Score
*	IoU
*	Accuracy

These metrics are used to evaluate how well each individual sample is handled by the model.

### Prediction Binding

Three types of predictions are registered via ```leap_binder.add_prediction()```:
*	object detection: Final post-NMS predictions used for visualization and metrics.
*	concatenate_128, concatenate_64, concatenate_32: Intermediate feature maps from the model, useful for internal analysis and debugging.