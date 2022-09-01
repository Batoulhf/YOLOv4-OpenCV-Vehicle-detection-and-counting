# YOLOv4-OpenCV-Vehicle-detection-and-counting
Vehicle detection, classification  and counting using the YOLOv4 model with OpenCV (Open-CV is a real-time computer vision library of Python)

This Repository walks you through building, training and running your own YOLOv4 object detector. Next, after the vehicles are detected, they are classified into different classes. Then We’ve used the YOLOv4 weights file and cfg file, along with OpenCV to count vehicles. 

[![open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KpfV77oGyY7KUzbByfhF8_i0kXi-jObL?usp=sharing)

## Implementation


### Train Your Own YOLOv4 Custom Object Detector!
We create your own custom YOLOv4 object detector to recognize any classes/objects we want!
In order to create a custom YOLOv4 detector we will need the following:

* Labeled Custom Dataset using [labelImg](https://github.com/heartexlabs/labelImg)
* Custom .cfg file
* obj.data and obj.names files
* train.txt file (test.txt is optional here as well)

### Preparing the training data
Bounding boxes List We need a .txt-file for each .jpg-image-file - in the same directory and with the same name, but with .txt-extension, and put to file: object number and object coordinates on this image, for each object in new line:

Where:

* `<object-class>` `<x_center>` `<y_center>` `<width>` `<height>` Where:
* `<object-class>` - integer object number from 0 to (classes-1)
* `<x_center>` `<y_center>` `<width>` `<height>` - float values relative to width and height of image, it can be equal from (0.0 to 1.0]
* for example: `<x>` = `<absolute_x>` / `<image_width>` or `<height>` = `<absolute_height>` / `<image_height>`
* atention: `<x_center>` `<y_center>` - are center of rectangle (are not top-left corner)

For example for `img1.jpg` you will be created `img1.txt` containing: 

```
1 0.716797 0.395833 0.216406 0.147222
0 0.687109 0.379167 0.255469 0.158333
1 0.420312 0.395833 0.140625 0.166667
```

Create file `train.txt` in directory build\darknet\x64\data\, with filenames of your images, each filename in new line, with path relative to darknet.exe, for example containing:

```
data/obj/img1.jpg
data/obj/img2.jpg
data/obj/img3.jpg
```
### Configuring Files for Training

download cfg file from darknet and edit the .cfg to fit your needs based on your object.

I recommend having `batch = 64` and `subdivisions = 16` for ultimate results. If you run into any issues then up `subdivisions to 32`.

Make the rest of the changes to the cfg based on how many classes you are training your detector on.

Note: I set my `max_batches = 10000` , `steps = 8000` , `9000`, I changed the `classes = 5` in the three YOLO layers and `filters = 30` in the three convolutional layers before the YOLO layers.

How to Configure Your Variables:

`width = 416`

`height = 416` (these can be any multiple of 32, 416 is standard, you can sometimes improve results by making value larger like 608 but will slow down training)

`max_batches = (# of classes) * 2000` (but no less than 6000 so if you are training for 1, 2, or 3 classes it will be 6000, however detector for 5 classes would have max_batches=10000)

`steps = (80% of max_batches)`, `(90% of max_batches)` (so if your `max_batches = 10000`, then `steps = 8000, 9000` )

`filters = (# of classes + 5) * 3` (so if you are training for one class then your filters = 18, but if you are training for 4 classes then your filters = 27)

- Optional: If you run into memory issues or find the training taking a super long time. In each of the three yolo layers in the cfg, change one line from random = 1 to random = 0 to speed up training but slightly reduce accuracy of model. Will also help save memory if you run into any memory issues.

### obj.names and obj.data
Create a new file within a code or text editor called obj.names where you will have one class name per line in the same order as your classes.txt from the dataset generation step.
Example for multiclass obj.names file:

![image](https://github.com/Batoulhf/YOLOv4-OpenCV-Vehicle-detection-and-counting/blob/main/Implementation/objName.png)


You will also create a obj.data file and fill it in like this (change your number of classes accordingly, as well as your backup location)

![image](https://github.com/Batoulhf/YOLOv4-OpenCV-Vehicle-detection-and-counting/blob/main/Implementation/objData.png)

This backup path is where we will save the weights to of our model throughout training. Create a backup folder in your google drive and put its correct path in this file.

### Generating train.txt and test.txt
The last configuration files needed before we can begin to train our custom detector are the train.txt and test.txt files which hold the relative paths to all our training images and valdidation images.

Luckily I have found a scripts that eaily generate these two files withe proper paths to all images.

The scripts can be accessed from the [Github Repo](https://github.com/theAIGuysCode/YOLOv4-Cloud-Tutorial/tree/master/yolov4)

Just download the two files to your local machine and upload them to your Google Drive so we can use them in the Colab Notebook.

## Train Your Custom Object Detector
```
!./darknet detector train <path to obj.data> <path to custom config> yolov4.conv.137 -dont_show -map
```
## Run Your Custom Object Detector

On image:

run custom detector with this command (upload an image to your google drive to test, thresh flag sets accuracy that detection must be in order to show it)
```
!./darknet detector test data/obj.data cfg/yolov4-obj.cfg yolov4/backup/yolov4-obj_last.weights yourimage.jpg -thresh 0.3
```
On video :
```
!./darknet detector demo data/obj.data cfg/yolov4-obj.cfg /backup/yolov4-obj_last.weights -dont_show yourvideo.mp4 -i 0 -out_filename results.mp4
```
Result :

![image](https://github.com/Batoulhf/YOLOv4-OpenCV-Vehicle-detection-and-counting/blob/main/Implementation/vid.gif)

# Next up - Follow the jupyter notebook!
[![open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11HCG-yqNjG4nsJGZXEyYXxbL1SBULE2Y?authuser=4#scrollTo=8dfPY2h39m-T)

# Counting
* We used OpenCV's DNN (Deep Neural Network) module to work directly with YOLOv4 (OpenCV has a built-in function to run DNN algorithms). 
* We created two programs, the first was the tracker which uses the concept of Euclidean distance to keep track of an object. It calculates the difference between two center points of an object in the current frame compared to the previous frame, and if the distance is less than the threshold distance, it confirms that the object is the same object as the previous image, and the 2nd one was the main detection program.

* First, we import all the necessary packages we need for the project and initialize the EuclideanDistTracker() object from the tracker we created earlier. Next, we load the YO LOv4 model weights and configuration files. YOLOv4 is trained on our dataset which contains 5 classes (car, bus, van, truck, motorcycle), so we read the class names file and store the names in a list. And finally we Configure the network using the cv2.dnn.readNetFromDarknet() function.

* Created a function of detected objects from the network output. 
* Keep track of each detected object and update the position of objects. 
* Generating a random color for each class in our dataset. 
* Created a function that counts the number of vehicles that have crossed the road. 
* We keep track of each vehicle position and their corresponding IDs. 
* After that, we draw the counting texts on the frame to show the vehicle counting in real time, and display the output video in a new opencv window.

Here is the result :
![image](https://github.com/Batoulhf/YOLOv4-OpenCV-Vehicle-detection-and-counting/blob/main/Implementation/count.gif)


# More details

* Paper YOLOv4: https://arxiv.org/abs/2004.10934
* source code YOLOv4 - Darknet (use to reproduce results): https://github.com/AlexeyAB/darknet
* About Darknet framework: http://pjreddie.com/darknet/
