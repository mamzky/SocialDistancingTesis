STEP 1 : CLONE REPO

STEP 2 : DOWNLOAD PRE-TRAINED MODEL

  download YOLOv4 object detection model di link berikut :
    YOLOv4 WEIGHTS : https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
    simpan model YOLOv4 Object Detection di direktori "/yolo-coco"

  download YOLOv4 Mask Detection di link beikut :
    YOLOv4 FULL MASK WEIGHTS : https://drive.google.com/file/d/1pdOQMZ4OMxk6deHhxffA03_dHMB83msu/view
    YOLOv4 TINY MASK WEIGHTS : https://github.com/BogdanMarghescu/Face-Mask-Detection-Using-YOLOv4/blob/main/Face%20Mask%20Detector/yolo_utils/yolov4-tiny-mask.weights
    simpan YOLOv4 Mask Detection di direktori "yolo_utils"

STEP 3 : INSTALL REQUIREMENTS
  jalankan "pip install -r requirements.txt"
  
STEP 4 : JALANKAN PROGRAM
  jalankan "python social_distancing_detector.py --input pedestrians.mp4 --output output.avi --display 1"
