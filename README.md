# YOLOv4-OpenCV-Vehicle-detection-and-counting-using-
Vehicle detection, classification  and counting using the YOLOv4 model with OpenCV (Open-CV is a real-time computer vision library of Python)

This Repository walks you through building, training and running your own YOLOv4 object detector. Next, after the vehicles are detected, they are classified into different classes. Then We’ve used the YOLOv4 weights file and cfg file, along with OpenCV to count vehicles. 

## Implementation


### Train Your Own YOLOv4 Custom Object Detector!
We create your own custom YOLOv4 object detector to recognize any classes/objects we want!
In order to create a custom YOLOv4 detector we will need the following:

* Labeled Custom Dataset
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

