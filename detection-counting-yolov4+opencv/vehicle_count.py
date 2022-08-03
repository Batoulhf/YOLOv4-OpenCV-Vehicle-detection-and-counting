# TechVidvan Vehicle counting and Classification

# Import necessary packages

import cv2
import csv
import collections
from tracker import *
import numpy as np
import argparse
import imutils
import time


# Initialize Tracker
tracker = EuclideanDistTracker()

# Initialize the videocapture object
cap = cv2.VideoCapture('vid6_trim.mp4')
input_size = 416

video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Detection confidence threshold
confThreshold =0.2
nmsThreshold= 0.2

font_color = (255, 255, 255)
font_size = 0.5
font_thickness = 1

a, b, c, d =0,0,198,120

# Middle cross line position
middle_line_position = 335 
up_line_position = middle_line_position - 15
down_line_position = middle_line_position + 15


# Store classes Names in a list
classesFile = "yolo/obj.names"
classNames = open(classesFile).read().strip().split('\n')
print(classNames)
print(len(classNames))

# class index for our required detection classes
required_class_index =  [0, 4, 1, 3, 2] #[2, 3, 5, 7]

detected_classNames = []

## Model Files
modelConfiguration = 'yolo/yolov4-obj.cfg'
modelWeigheights = 'yolo/yolov4-obj_best_final.weights'

# configure the network model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)

# Define random colour for each class
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')


# Function for finding the center of a rectangle
def find_center(x, y, w, h):
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1
    cy=y+y1
    return cx, cy
    
# List for store vehicle count information
temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0, 0, 0]
down_list = [0, 0, 0, 0,0]

# Function for count vehicle
def count_vehicle(box_id, img):

    x, y, w, h, id, index = box_id

    # Find the center of the rectangle for detection
    center = find_center(x, y, w, h)
    ix, iy = center
    
    # Find the current position of the vehicle
    if (iy > up_line_position) and (iy < middle_line_position):

        if id not in temp_up_list:
            temp_up_list.append(id)

    elif iy < down_line_position and iy > middle_line_position:
        if id not in temp_down_list:
            temp_down_list.append(id)
            
    elif iy < up_line_position:
        if id in temp_down_list:
            temp_down_list.remove(id)
            up_list[index] = up_list[index]+1

    elif iy > down_line_position:
        if id in temp_up_list:
            temp_up_list.remove(id)
            down_list[index] = down_list[index] + 1

    # Draw circle in the middle of the rectangle
    cv2.circle(img, center, 2, (0, 0, 255), -1)  # end here
    # print(up_list, down_list)

# Function for finding the detected objects from the network output
def postProcess(outputs,img):
    global detected_classNames 
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
                    # print(classId)
                    w,h = int(det[2]*width) , int(det[3]*height)
                    x,y = int((det[0]*width)-w/2) , int((det[1]*height)-h/2)
                    boxes.append([x,y,w,h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)
    # print(classIds)
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            # print(x,y,w,h)

            color = [int(c) for c in colors[classIds[i]]]
            name = classNames[classIds[i]]
            detected_classNames.append(name)
            # Draw classname and confidence score 
            cv2.putText(img,f'{name.upper()} {int(confidence_scores[i]*100)}%',
                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw bounding rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            detection.append([x, y, w, h, required_class_index.index(classIds[i])])

        # Update the tracker for each object
        boxes_ids = tracker.update(detection)
        for box_id in boxes_ids:
            count_vehicle(box_id, img)

writer = None
alpha = 0.4
#def realTime():
while True:
    success, img = cap.read()
    img = cv2.resize(img,(0,0),None,0.5,0.5)
    ih, iw, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

        # Set the input of the network
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i- 1]) for i in net.getUnconnectedOutLayers()]
    # Feed data to the network
    outputs = net.forward(outputNames)
    
        # Find the objects from the network output
    postProcess(outputs,img)

        # Draw the crossing lines

    cv2.line(img, (0, middle_line_position), (iw, middle_line_position), (255, 0, 0), 1)
    cv2.line(img, (0, up_line_position), (iw, up_line_position), (0, 250, 154), 1)
    cv2.line(img, (0, down_line_position), (iw, down_line_position), (0, 250, 154), 1)

    cv2.rectangle(img, (a,a), (a+c, b+d), (0,0,0),-1)
    
        # Draw counting texts in the frame
    cv2.putText(img, "Up", (112, 12), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Down", (147, 12), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Car :         " +str(up_list[0])+"     "+ str(down_list[0]), (5, 33), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Motorcycle :  " +str(up_list[1])+"     "+ str(down_list[1]), (5, 53), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Bus :         " +str(up_list[2])+"     "+ str(down_list[2]), (5, 73), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Truck :       " +str(up_list[3])+"     "+ str(down_list[3]), (5, 93), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    cv2.putText(img, "Van :         " +str(up_list[4])+"     "+ str(down_list[4]), (5, 113), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
    
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (video_width, video_height), True)
    
    
    writer.write(img)
        # Show the frames
    cv2.imshow('Output', img)
        
    if cv2.waitKey(1) == ord('q'):
        break

    # Write the vehicle counting information in a file and save it

with open("data-2.csv", 'w') as f1:
    cwriter = csv.writer(f1)
    cwriter.writerow(['Direction', 'car', 'motorcycle', 'bus', 'truck', 'van'])
    up_list.insert(0, "Up")
    down_list.insert(0, "Down")
    cwriter.writerow(up_list)
    cwriter.writerow(down_list)
f1.close()
print("Data saved at 'data.csv'")
    # Finally realese the capture object and destroy all active windows

writer.release()        
img.release() 


    # Closes all the frames
cv2.destroyAllWindows()
    
print("The video was successfully saved")

#if __name__ == '__main__':
    #realTime()
    
