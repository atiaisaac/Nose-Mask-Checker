# Nose-Masker-Checker
This is a deep learning project built to classifier images in two categories - mask and no mask. This system has two deep learning models to it.

* An image classification model for predicting the two classes trained using feature extraction transfer learning. It is built using tensorflow 
* A caffe deep learning model for face detection

Putting the two together, a model to detect faces wearing or not wearing a nose mask is created. The model is optimized to work on an edge device like the raspberry pi by converting it into a tflite file using tensorflowlite.
OpenCV is used to capture a live stream from the raspberry pi camera drawing different colored boundary box around faces with nose mask and vice versa.

The various files in this repo include
1. spot the mask.ipynb a jupyter notebook to show how the training was done
2. detect and classify.py which is the main script that detects and classifies faces with/without nose mask from an opencv video capture.
3. face detection with opencv and dnn - a folder that contains the caffe model and weights for face detection
4. nose_mask.tflite the optimized tensorflowlite buffer file optimized to run on a number of edge cpu's. 

Documentation of the entire design and hardware process can be found in my hackster account (visit https://hackster.io/Atia/nose-mask-checker)


# Note

This project was done in response to the COVID-19 detect and protect challenge organized by hackster and showcases how enthusiasts can contribute their quota into curbing the crisis. Even though not all of us can be at the forefront like our brave scientist, as individuals we can play our part no matter how small it may seem. You never know who it might have a positive impact on. 

# Credits

Credit goes to;
* Shubham Rath for their code in face detection with opencv and dnn (https://github.com/sr6033/face-detection-with-OpenCV-and-DNN). It is from here that I got the caffe model and weights file to use.
