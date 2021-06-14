# Cyclists-Helmet-Detection

An Implementation of YOLOv3 Object Detection Algorithm to detect Cyclists and their Helmets in a set of images.
These images are scrapped from Google Images using the Keyword "Cyclists". Although this keyword can be customised like many other things in the script.
Thus for the script to work with its default parameters and paths, I'd highly recommend to please place files as shown in the below Directory Tree. 

The Structure should be as follows:

└── Cyclists-Helmet-Detection(Main Branch)
    ├── chromedriver
    ├── Grab_that_image.py
    ├── main.py
    ├── requirements.txt
    ├── Run_cyclists_counter.py
    ├── Run_helmet_detector.py
    ├── yolo
    │   ├── coco.names
    │   ├── yolov3.cfg
    │   └── yolov3.weights
    └── yolo_helmet
        ├── obj.names
        ├── yolov3-obj_2400.weights
        └── yolov3-obj.cfg

3 directories, 12 files





