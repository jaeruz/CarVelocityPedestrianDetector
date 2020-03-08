# Import necessary libraries

from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
import imutils
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import numpy as np
import argparse
import cv2
import tkinter


# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser(
    description='Script to run MobileNet-SSD object detection network')
ap.add_argument('-v', '--video', type=str, default='',
                help='Path to video file. If empty, web cam stream will be used')
ap.add_argument('-p', '--prototxt', required=True,
                help="Path to Caffe 'deploy' prototxt file")
ap.add_argument('-m', '--model', required=True,
                help='Path to weights for Caffe model')
ap.add_argument('-l', '--labels', required=True,
                help='Path to labels for dataset')
ap.add_argument('-c', '--confidence', type=float, default=0.9,
                help='Minimum probability to filter weak detections')
ap.add_argument('-i', '--ignore', required=True,
                help="Path to ignore txt")

args = vars(ap.parse_args())


ct = CentroidTracker()
# Initialize class labels of the dataset
CLASSES = [line.strip() for line in open(args['labels'])]

IGNORE = [line.strip() for line in open(args['ignore'])]
objID = []*20
objdistance = []*20
#print('[INFO]', CLASSES)
#distance params
knownWidth =0.7
knownDistance =0.7
pixWidth = 550
focalLength= (pixWidth*knownWidth)/knownDistance
##
initDistance =15;


# Generate random bounding box colors for each class label
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 2))

# Load Caffe model from disk
#print("[INFO] Loading model")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Open video capture from file or capture device

def improc(carV,limitD):
    Vcar = carV
   
    start = time.process_time()
    print("[INFO] Starting video stream")
    if args['video']:
        cap = cv2.VideoCapture(args['video'])
    else:
        camera = PiCamera()
        camera.resolution = (640, 480)
        camera.framerate = 32
        rawCapture = PiRGBArray(camera, size=(640, 480))

    for fram in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # Capture frame-by-frame
        frame = fram.array

       

        (h, w) = frame.shape[:2]

        # MobileNet requires fixed dimensions for input image(s)
        # so we have to ensure that it is resized to 300x300 pixels.
        # set a scale factor to image because network the objects has differents size.
        # We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
        # after executing this command our "blob" now has the shape:
        # (1, 3, 300, 300)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        # Pass the blob through the network and obtain the detections and predictions
        net.setInput(blob)
        detections = net.forward()
        rects = []
        
        for i in range(detections.shape[2]):
            # Extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            
            if confidence > args["confidence"]:
                # Extract the index of the class label from the `detections`,
                # then compute the (x, y)-coordinates of the bounding box for
                # the object
                class_id = int(detections[0, 0, i, 1])
                if CLASSES[class_id] in IGNORE:
                    continue
                
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                rects.append(box.astype("int"))
                
                (startX, startY, endX, endY) = box.astype('int')

                # Draw bounding box for the object
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[class_id], 2)

                # Draw label and confidence of prediction in frame
                label = ""#"{}: {:.2f}%".format(CLASSES[class_id], confidence * 100)
                #print("[INFO] {}".format(label))
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                            (0,255,0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                #Width= endX-startX
                #distance = (knownWidth*focalLength)/Width
                
                
                #label=label+"/ distance:"+str(distance)
                

                objects = ct.update(rects)
                
            #loop
                for (objectID, centroid) in objects.items():
                    
                    # draw both the ID of the object and the centroid of the
                    # object on the output frame
                        text = "ID {}".format(objectID)
                        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                        cv2.putText(frame, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,# COLORS[class_id]
                            (0), 2)
                        #seconds = time.time()
                        #t = time.localtime(seconds)
                        #print("t1: ", t.tm_sec)
                        #if t.tm_sec == 0:
                        cWidth = centroid[3]-centroid[2]
                        cdistance = float("{0:.2f}".format((knownWidth*focalLength)/cWidth))
                        #print(cdistance)
                        
                        elapsed =(time.process_time() - start)                        
                        if elapsed > 10:
                            elapsed = 1
                            start = time.process_time()
                            del objID[:]
                            del objdistance[:]
                            
                        if objectID not in objID:
                            objID.append(objectID) 
                            objdistance.append(cdistance)
                        
                        #print(str(len(objID))+"/"+str(objdistance))
                            
                        
                            
                        try:
                            #if len(objID) == len(objdistance) and len(objID)!=0:
                                
                            #deltaDistance = cdistance - objdistance[objectID]
                            print(limitD)
                            if cdistance<limitD:
                                print("too close!!")
                            else:
                                print("good")
                            
                            
                            
                            
                            
                            #print("ID: "+str(objectID) +" || sec: "+ str(int(elapsed)) +" || Vtot: " + str(Vtot)+ " || Vobj: " + str(Vobj)+ " || Dinit:" + str(float("{0:.2f}".format(objdistance[objectID]))) + " || D:" + str(cdistance));
                            label = "distance:" + str(cdistance)# +"pix:" + str(cWidth)
                            
                            cv2.putText(frame, label, (centroid[0] - 10, centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,255),# COLORS[class_id]
                                2)
                        except IndexError as e:
                            objdistance.append(cdistance)
                            print(str(objID)+"||"+str(objdistance))
            
                        
                        
        # Show fame
        cv2.imshow("Frame", frame)
        rawCapture.truncate(0)
        key = cv2.waitKey(1) & 0xFF

        # Press `q` to exit
        if key == ord("q"):
            break

    # Clean-up
    
    cv2.destroyAllWindows()




