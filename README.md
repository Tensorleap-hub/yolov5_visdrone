# General
This project demonstrates the integration of a YOLOv5 model with the VisDrone dataset on the Tensorleap platform. The integration logic is implemented in leap_binder.py, which handles data loading, metadata computation, loss definition, image and bounding box visualization, metrics, and more.

Before pushing the project to Tensorleap, it’s recommended to run the leap_custom_test.py script after setting up the data (as explained below). This allows you to verify that data loading and visualizations function as expected. Once the test passes, proceed to the deployment step.

# Data handling
To replicate the data handling used in this project, refer to the VisDrone.yaml configuration file. The path field should point to the root directory of your dataset, which must be located within the Tensorleap-mounted folder (~/tensorleap). If you are creating your own yaml, you do not have to implement the download part as it is optional.  

In order to use your own yaml file, refer to "preprocess_func_leap" function within the leap_binder.py and change the data_path to point to the new yaml.

The train, val, and test fields should each reference the corresponding images subfolder within the dataset. Each of these directories (train, val, and test) must contain two subfolders:
	•	images: containing the image files
	•	labels: containing a .txt file for each image (e.g., 0000002_00005_d_0000014.jpg should have a corresponding 0000002_00005_d_0000014.txt)

Each line in the label file should represent a bounding box in the following format:
<label> <x> <y> <w> <h>
where x, y, w, and h are normalized values in the range [0, 1].


# Pushing project to Tensorleap
To upload the code and model to the Tensorleap platform, navigate to the project directory and run the following command, specifying the path to the model weights file:  
```
leap project push weights/yolov5s-visdrone.h5
```
Replace the path with your specific model weights file if different.
