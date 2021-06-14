#Importing necessary Libraries and Packages.
from main import cyclistCounter
from imutils import paths

###Initializing cyclists counter
#Get Image Paths in the form of a list
image_paths = list(paths.list_images("images"))

#Empty list for Counting Total Cyclists in all Images
cnt_cyclist = [] 
#Looping through every individual path
for image_path in image_paths:
    #Just some space to make things look clean
	print('\n')
    #Feeding our individual image path to our Cyclist Counter Class
    #In return we'll receive our Count of Total Cyclists in all Images
    #Plus every image will be overwritten in the original directory.
	detected_img = cyclistCounter.image_detect(image_path, cnt_cyclist)
	"""
    Press Esc Key to go through the processed images one by one in the Window.
    """
print("Total Cyclists in all Images: {}".format(sum(cnt_cyclist)))

