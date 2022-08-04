# YOLOv4-OpenCV-Vehicle-detection-and-counting-using-
Vehicle detection, classification  and counting using the YOLOv4 model with OpenCV (Open-CV is a real-time computer vision library of Python)

This Repository walks you through building, training and running your own YOLOv4 object detector. Next, after the vehicles are detected, they are classified into different classes. Then We’ve used the YOLOv4 weights file and cfg file, along with OpenCV to count vehicles. 

## Implementation

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
