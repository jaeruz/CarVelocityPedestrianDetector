from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
import imutils
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import numpy as np
import argparse
import cv2


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
ap.add_argument('-c', '--confidence', type=float, default=0.5,
                help='Minimum probability to filter weak detections')
ap.add_argument('-i', '--ignore', required=True,
                help="Path to ignore txt")

args = vars(ap.parse_args())


ct = CentroidTracker()

CLASSES = [line.strip() for line in open(args['labels'])]

IGNORE = [line.strip() for line in open(args['ignore'])]
objID = []*20
objdistance = []*20
#print('[INFO]', CLASSES)
#distance params
knownWidth =123
knownDistance =130
pixWidth = 440
focalLength= (pixWidth*knownDistance)/knownWidth

##
initDistance =15;



COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 2))


net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


def vproc(Vcar):
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

       
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        net.setInput(blob)
        detections = net.forward()
        rects = []
        
        for i in range(detections.shape[2]):
           
            confidence = detections[0, 0, i, 2]

        
            
            if confidence > args["confidence"]:
                
                class_id = int(detections[0, 0, i, 1])
                if CLASSES[class_id] in IGNORE:
                    continue
                
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                rects.append(box.astype("int"))
                
                (startX, startY, endX, endY) = box.astype('int')

               
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[class_id], 2)

               
                label = ""#"{}: {:.2f}%".format(CLASSES[class_id], confidence * 100)
                
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0,255,0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15

                objects = ct.update(rects)
                
            #loop
                for (objectID, centroid) in objects.items():
                    
                        #text = "ID {}".format(objectID)
                        #cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        #        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        #cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                       
                       
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
                 
                            
                        try:
                          
                                
                            Vobj = float("{0:.2f}".format((objdistance[objectID]- cdistance + (Vcar*elapsed))/elapsed))
                            print("ID: "+str(objectID) +" || sec: "+ str(int(elapsed)) + " || Vobj: " + str(Vobj)+ " || Dinit:" + str(float("{0:.2f}".format(objdistance[objectID]))) + " || D:" + str(cdistance));
                            label = "velocity:" + str(Vobj)
                            
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
