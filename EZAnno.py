import os
import cv2
import numpy as np
import time
import sys
import os

# Input your video file path here
# For example: yellowSphere.mp4
videoPath = os.getcwd() + "\\haha.mp4"

if not os.path.exists(videoPath): print("Video file not found")

newVideoPath = os.getcwd() + '\\videoStuff'
if not os.path.exists(newVideoPath):
    os.mkdir(newVideoPath)


def extract_image_one_fps(video_source_path, numsec):
    vidcap = cv2.VideoCapture(video_source_path)
    count = 1
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * numsec))  # 2 second***
        success, image = vidcap.read()
        ## Stop when last frame is identified
        image_last = cv2.imread("frame{}.png".format(count))
        if np.array_equal(image, image_last):
            break
        cv2.imwrite(newVideoPath + "\\frame%d.png" % count, image)  # save frame as PNG file
        print('{}.sec reading a new frame:{}'.format(count, success))
        count += 1


extract_image_one_fps(videoPath, 1000)

# Now in order to get the bounding boxes for the objects we can
# run either the base yolo program 
# or get the largest contor program

# I think we should go with largest contonr program becuase that 
# the yolo program gives bad results from prelimary testing

# YOLO base model format
config_path = os.getcwd() + '\\yolov4.cfg'
weights_path = os.getcwd() + '\\yolov4.weights'
names = os.getcwd() + '\\coco.names'
dataExtracted = []

for imageFileName in os.listdir(newVideoPath):
    path_name = newVideoPath + "\\" + imageFileName
    CONFIDENCE = 0.1
    SCORE_THRESHOLD = 0.1
    IOU_THRESHOLD = 0.1
    # loading all the class labels (objects)
    labels = open(names).read().strip().split("\n")
    # generating colors for each object for later plotting
    # colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    image = cv2.imread(path_name)
    file_name = os.path.basename(path_name)
    filename, ext = file_name.split(".")
    h, w = image.shape[:2]
    # create 4D blob
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    print("Name: " + imageFileName)
    print("Image Shape:", image.shape)

    net.setInput(blob)

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    start = time.perf_counter()
    layer_outputs = net.forward(ln)
    time_took = time.perf_counter() - start
    print(f"Time took: {time_took:.2f}s")
    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        # loop over each of the object detections
        for detection in output:
            # extract the class id (label) and confidence (as a probability) of
            # the current object detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # discard out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > CONFIDENCE:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    for i in range(len(boxes)):
        x, y = boxes[i][0], boxes[i][1]
        w, h = boxes[i][2], boxes[i][3]
        print(class_ids[0])
        text = f"{labels[class_ids[0]]}: {confidences[i]:.2f}"
        print(text)
    if len(boxes) > 0:
        dataExtracted.append([imageFileName, [boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]]])
        boxes = np.array(boxes)
    print("Detected {} objects".format(len(boxes)))
    print("-" * 30)


# Data Extracted Format is 
# file name - [x, y, w, h]

# Contor Format
def get_contour_areas(contours):
    all_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)
    return all_areas


for imageFileName in os.listdir(newVideoPath):
    image = cv2.imread(newVideoPath + "\\" + imageFileName)
    original_image = image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 200)

    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

largest_item = sorted_contours[0]
print(largest_item)


# Incomplete code
# Contour points result in a list of points that are all the 
# circle points in our case, so we need to convert this into a 
# rectangle 4 points that encase the circle which can be done by 

# Get the first point and middle point and then using pythogram 
# therom calculate the diameter

# Using that add a threshold amount (around 20) then get the 25% 
# and 75% points which you can then build a rectangle from the 
# four conrner points. Then extract width and height and the 
# top left x and y point


def mask_to_rect(image):
    # Getting the Thresholds and ret
    ret, thresh = cv2.threshold(image, 0, 1, 0)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

    # Checking the version of open cv I tried for (version 4)
    #    Getting contours on the bases of thresh
    if int(cv2.__version__[0]) > 3:
        contours, hierarchy = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        im2, contours, hierarchy = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Getting the biggest contour
    if len(contours) != 0:
        # draw in blue the contours that were founded
        cv2.drawContours(output, contours, -1, 255, 3)

        # find the biggest countour (c) by the area
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

    return [x, y, w, h]


def get_center_point(image):
    # Get the largest contour object
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    # Get the center point of the largest contour object
    M = cv2.moments(biggest_contour)
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    return center_x, center_y


# Now for either ways you will result in an array with the following format
# [file name, [x, y, w, h]]
# Now just crop the image or input this data directly 
# into the model (idk how you trained ur model)
for imageFileName in os.listdir(newVideoPath):
    f = open(imageFileName + ".txt", "w+")
    f.write("0 ")
    image = cv2.imread(newVideoPath + "\\" + imageFileName)
    x = 0
    for box in mask_to_rect(image):
        x += 1
        if (x % 2) == 0:
            f.write(str(box / image.shape[0]) + " ")
        else:
            f.write(str(box/ image.shape[1]) + " ")
    f.write("\n")
