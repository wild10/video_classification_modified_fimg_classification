# VIDEO CLASSIFICATION

This is a repo that contains a peace of code modified in purpose to do video classification using the previous
image classification code contained in TensorFlow examples.

### file Content:
- label_image.py (modified from the original) see original [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/label_image.py)
- in order to simulate we created a new folder called **'screens'** folder that stores frames  inside of [data folder] (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/label_image/data)
- video sample: this should be in the data folder, so the
    - **video_path** and **screens_folder_path** should be changed to your local system dir path
    - additionally the video screens are recorded (recognized.avi) to visualize the test.

## If you want to see experiments you can access the YouTube demo

click [youtube video demo](https://www.youtube.com/watch?v=mZdsx-WhCwo&ab_channel=ErrolWilderdMamaniCondori)

## installation dependencies
* tensorflow clone [this] (https://github.com/tensorflow/tensorflow)
* and mainly python 3.7
* virtual env with anaconda
* bazel 5.3.0 to install full tensorflow examples

```
numpy 1.21.6
opencv2 4.2.0
tensorflow 1.2.3

```
## run
* train (in your virtual env or setting site)

```
python label_image.py
```

## results


<img src="demo.png" height="400" width="300" > <img src="demo_1.png" height="400" width="500" >
<img src="demo_2.png" height="400" width="300" > <img src="demo_3.png" height="400" width="500" >

