# Cyclists-Helmet-Detection

An Implementation of YOLOv3 Object Detection Algorithm to detect Cyclists and their Helmets in a set of images.
These images are scrapped from Google Images using the Keyword "Cyclists". Although this keyword can be customised like many other things in the script.
Thus for the script to work with its default parameters and paths, I'd highly recommend to please place files as shown in the below Directory Tree. 

The Structure should be as follows:

└── Cyclists-Helmet-Detection(Main Branch)\
&nbsp;&nbsp;&nbsp;&nbsp;├── chromedriver\
&nbsp;&nbsp;&nbsp;&nbsp;├── Grab_that_image.py\
&nbsp;&nbsp;&nbsp;&nbsp;├── main.py\
&nbsp;&nbsp;&nbsp;&nbsp;├── requirements.txt\
&nbsp;&nbsp;&nbsp;&nbsp;├── Run_cyclists_counter.py\
&nbsp;&nbsp;&nbsp;&nbsp;├── Run_helmet_detector.py\
&nbsp;&nbsp;&nbsp;&nbsp;├── yolo\
&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── coco.names\
&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── yolov3.cfg\
&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── yolov3.weights\
&nbsp;&nbsp;&nbsp;&nbsp;└── yolo_helmet\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── obj.names\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── yolov3-obj_2400.weights\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── yolov3-obj.cfg\
        
3 directories, 12 files

# Web Scraping:-
Web Scraping is done using Selenium. It takes access of Google Chrome, thus chromedriver is mandatory and the chromedriver version should specifically match with the GoogleChrome Version installed on your system.
1. Check Version of Google Chrome -----> Open Chrome > Settings > About Chromw (Leftmost Panel)> Check version.
2. To download relevant chromedriver, this link should be used else lastest version is already uploaded in the repository -----> https://chromedriver.chromium.org/downloads

# YOLO:-
This Script uses the concept of Transfer Learning and the yolo pretrained-weights.
YOLOv3 for Cyclist Detection and YOLOV3-Obj for Helmet Detection. Thus the weights, cfg, obj files are needed to be downloaded and are easily accessible from the link given below.

Yolo & Yolo_helmet Folder - https://drive.google.com/drive/folders/1QV9DQj2oqdrruuhSiu9pl6SP19unTafi?usp=sharing

# Points to Remember:
1. main.py consists of two classes, One for Cyclists Detection and Counter and Second for Helmet+Cyclists Detection and Counter. 
2. Run Run_cyclist_counter.py file for Cyclists Detection
3. Run Run_helmet_detector.py file for Cyclists and Helmet Detection
4. While OpenCV uses its window to display the detections, pls press 'Esc' Key to scroll and view the image one by one.
5. All the detections will be saved in the "images" folder which will be created after scraping.
6. Make sure DRIVER_PATH in Grab_that_image.py leads to the chromedriver file
7. Number of images to be scraped can be entered in Grab_that_image.py. Default is 5. 

# STEPS TO RUN THE SCRIPT:
1. Run Requirements.txt file as follows:
    $ pip install -r requirements.txt
2. Run Grab_that_image.py
3. Run Run_helmet_detector.py

# Outputs:
"images" folder will have 2 sub-folders which will contain all the detected images with the counts on them.
The outputs looks something just like this:
Cyclists Detection:-
![alt text](https://github.com/CuFFaz/Cyclists-Helmet-Detection/blob/main/images/cyclist/f80b6ad9e4.jpg)
![alt text](https://github.com/CuFFaz/Cyclists-Helmet-Detection/blob/main/images/cyclist/a065255504.jpg)

Helmet Detections:-
![alt text](https://github.com/CuFFaz/Cyclists-Helmet-Detection/blob/main/images/Helmet_detections/Image.00032.jpeg)
![alt text](https://github.com/CuFFaz/Cyclists-Helmet-Detection/blob/main/images/Helmet_detections/Image.00049.jpeg)



