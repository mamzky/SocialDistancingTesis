# imports
from configs import config
from configs.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
                help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
                help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
                help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

# MASK PROPERTIES
LABELS_MASK = ["Without Mask", "Mask"]
COLORS = [[0, 0, 255], [0, 255, 0]]

# weightsPath_mask = "yolo_utils/yolov4-tiny-mask.weights"
# configPath_mask = "yolo_utils/yolov4-tiny-mask.cfg"
weightsPath_mask = "yolo_utils/yolov4-face-mask.weights"
configPath_mask = "yolo_utils/yolov4-mask.cfg"

# load the COCO class labels the YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov4.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov4.cfg"])

# load the YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net_mask = cv2.dnn.readNetFromDarknet(configPath_mask, weightsPath_mask)
net_mask = cv2.dnn_DetectionModel(configPath_mask, weightsPath_mask)
net_mask.setInputSize(640, 640)
net_mask.setInputScale(1.0 / 255)
net_mask.setInputSwapRB(True)

# check if GPU is to be used or not
if config.USE_GPU:
    # set CUDA s the preferable backend and target
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    net_mask.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net_mask.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the "output" layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
# open input video if available else webcam stream
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None

# loop over the frames from the video stream
while True:
    # read the next frame from the input video
    (grabbed, frame) = vs.read()

    start_time = time.time()

    # if the frame was not grabbed, then that's the end fo the stream
    if not grabbed:
        break

    # MASK DETECTION
    classes, confidences, boxes = net_mask.detect(frame, 0.3, 0.3)
    mask_count = 0
    nomask_count = 0
    for cl, score, (left, top, height, width) in zip(classes, confidences, boxes):
        mask_count += cl[0]
        nomask_count += (1 - cl[0])
        start_point = (int(left), int(top))
        end_point = (int(left + width), int(top + height))
        color = COLORS[cl[0]]
        frame = cv2.rectangle(frame, start_point, end_point, color, 2)
        text_mask = f'{LABELS_MASK[cl[0]]}: {score[0]:0.2f}'
        (test_width, text_height), baseline = cv2.getTextSize(
            text_mask, cv2.FONT_ITALIC, 0.3, 1)
        end_point = (int(left + test_width + 2), int(top - text_height - 2))
        frame = cv2.rectangle(frame, start_point, end_point, color, -1)
        cv2.putText(frame, text_mask, start_point,
                    cv2.FONT_ITALIC, 0.3, (255, 255, 255), 1)
    print("MASK COUNT ", format(mask_count))
    print("NOMASK COUNT ", format(nomask_count))
    # resize the frame and then detect people (only people) in it
    frame = imutils.resize(frame, width=700)
    results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))

    # initialize the set of indexes that violate the minimum social distance
    violate = set()

    # ensure there are at least two people detections (required in order to compute the
    # the pairwise distance maps)
    if len(results) >= 2:
        # extract all centroids from the results and compute the Euclidean distances
        # between all pairs of the centroids
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")

        # loop over the upper triangular of the distance matrix
        for i in range(0, D.shape[0]):
            for j in range(i+1, D.shape[1]):
                # check to see if the distance between any two centroid pairs is less
                # than the configured number of pixels
                if D[i, j] < config.MIN_DISTANCE:
                    # update the violation set with the indexes of the centroid pairs
                    violate.add(i)
                    violate.add(j)

    # loop over the results
    for (i, (prob, bbox, centroid)) in enumerate(results):
        # extract teh bounding box and centroid coordinates, then initialize the color of the annotation
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)

        # if the index pair exists within the violation set, then update the color
        if i in violate:
            color = (0, 0, 255)

        # draw (1) a bounding box around the person and (2) the centroid coordinates of the person
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)

    # draw the total number of social distancing violations on the output frame
    text = "Jumlah Pelanggaran Social Distancing : {} Orang".format(
        len(violate))
    text_total = "Jumlah Orang Terdeteksi : {} Orang".format(len(results))

    cv2.rectangle(frame, (0, frame.shape[0] - 40),
                  (393, 393), (255, 255, 255), -1)
    cv2.putText(frame, text, (2, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(frame, text_total, (2,
                frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    fps = 1.0 / (time.time() - start_time)
    print("FPS : %.2f" % fps)

    # check to see if the output frame should be displayed to the screen
    if args["display"] > 0:
        # show the output frame
        cv2.imshow("Output", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key is pressed, break from the loop
        if key == ord("q"):
            break

    # if an output video file path has been supplied and the video writer ahs not been
    # initialized, do so now
    if args["output"] != "" and writer is None:
        # initialize the video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(
            args["output"], fourcc, 25, (frame.shape[1], frame.shape[0]), True)

    # if the video writer is not None, write the frame to the output video file
    if writer is not None:
    #     print("[INFO] writing stream to output")
        writer.write(frame)
