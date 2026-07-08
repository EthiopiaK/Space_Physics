#Steps to use clickable.py
#For MACCS spectrograms: (if not using MACCS spectrograms, cropping dimensions are needed)
#create a txt file to output values onto
#add the following header to it starthr endhr   startfreq   endfreq color
#on Notepad++ change the input folder directory to where the sprectrograms are found
#Set filex to your output folder's name near line 126 (with the quotation marks)
#save program 
#in your environment's terminal type python clickable.py and hit enter
#click on leftmost, rightmost, topmost and lowest points of the EMIC event (the order doesnot matter). There will be an output on the output file on four clicks. 
#do the same for another EMIC event if there is more than one event on a spectrogram
#click tab/q to move to the next spectrogram


# Import libraries
import cv2 as cv
import math
import numpy as np
import os 
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from skimage import restoration, measure
from skimage.feature import blob_dog, blob_log, blob_doh, canny
from skimage.color import rgb2gray
from os import listdir

start_time= datetime.now()
#The following program structuring and color mapping was inspired by Dr. Nyssa Capman's CNN_Training.py script
#Attempting to create a clickable blob property extractor
# 0) First define functions to map pixel locations to the corresponding time and frequency on spectrogram, as well as what to do on a click
#NOTE: the time is not evenly divided on the spectrograms. 
#Some hour ranges are 10.8375 pixels wide where the largest hour range is 11.899 pixels wide.
#In contrast frequency ranges are evenly spaced.

#-----------------------------------------------------------------------------------------------------------------
total_time = 86400 #in seconds
highest_freq = 1000 #in mHz
#-----------------------------------------------------------------------------------------------------------------


def time_mapper(columnindex, width, height):
    time_per_pixel = total_time/width
    timevalue = time_per_pixel * columnindex
    #time=[]
    #for y,x,r in bloblist:
    formatted = str(timedelta(seconds=timevalue))
    #time.append(formatted)
    return formatted

def freq_mapper(rowindex, width, height):
    freq_per_pixel = highest_freq/height
    freq =  (highest_freq - (freq_per_pixel*rowindex))/1000  #because the row index of top most point is zero , and 13.407 to account for the one pixel gap at the top 
    if freq < 0:
        freq = 0
    return freq                                              #returned in Hz

def color_mapper(subimage, xmin, xmax, ymin, ymax):
    xmin = int(xmin)
    xmax = int(xmax)
    ymin = int(ymin)
    ymax = int(ymax)
    
    roi = subimage[ymin:ymax, xmin:xmax]
    temp_blue = np.median(roi[:,:,0])
    temp_green = np.median(roi[:,:,1])
    temp_red = np.median(roi[:,:,2])
    
    if (temp_red == 255 and 102 <= temp_green <= 255 and 102 <= temp_blue <= 255):
       color = 'white'
    elif (212 <= temp_red <= 255 and 50 <= temp_green <= 111  and 0 <= temp_blue <= 102):
        color = 'red'
    elif (212 <= temp_red <= 255 and 111 <= temp_green <= 188 and temp_blue <= 85):
        color = 'orange'
    elif (154 <= temp_red <= 255 and 188 <= temp_green <= 255 and 0 <= temp_blue <= 150): #edited this to include yellow
        color = 'yellow'
    elif (50 <= temp_red <= 154 and 175 <= temp_green <= 255 and 50 <= temp_blue <= 220):
        color = 'green'
    elif (0 <= temp_red <= 80 and 0 <= temp_green <= 221 and 0 <= temp_blue <= 255):
        color = 'blackblue'
    else:
        color = 'unassigned'
    return color
    
    
    
counter = 0
coordinates = []
columnindex = []
rowindex = []
#cv.mouseCallback expects a function with five parameters
def on_click(event, x, y, flags, params):
    global coordinates, counter, columnindex, rowindex
    if event == cv.EVENT_LBUTTONDOWN:
       columnindex = np.append(columnindex, x)
       rowindex = np.append(rowindex, y)
       counter += 1
       #print (columnindex, counter)
    if event == cv.EVENT_LBUTTONDOWN and counter == 4:
        #columnindex = coordinates[0]
        #rowindex = coordinates[1]
        xmin = np.min(columnindex)
        xmax = np.max(columnindex)
        ymin = np.min(rowindex)
        ymax = np.max(rowindex)
        
        starthr = time_mapper(xmin, wx, hx)
        endhr = time_mapper(xmax, wx, hx)
        startfreq = freq_mapper(ymax, wx, hx)
        endfreq = freq_mapper(ymin, wx, hx)
        ave_power = color_mapper(imagex, xmin, xmax, ymin, ymax)
        
        with open(filex, 'a') as f:
                f.write(no_ext + "\t" + f'{starthr}' + "\t" + f'{endhr}' + "\t" + f'{startfreq}' + "\t" + f'{endfreq}' +"\t" + ave_power + "\n")
                
        #set counter and index holders to zero here in case there is more than one event on a spectrogram, if not exit spectrogram with the q key 
        counter = 0 
        columnindex = []
        rowindex = []
        
        
# 1) ---------------------------INPUT AND OUTPUT DIRECTORIES----------------------------------------------------------------------------------------------------
input_folder='C:/users/ethio/Zenodo/CDRjpg/2'

#opening textfiles in the current directory to write blob information on 
filex = 'CDR2017.txt'
#filex = 'trial.txt'

#comment the following two lines out if working to add lines to an existing document
#with open(filex, 'w') as outputx:  
#    outputx.write('filename'+ "\t" + 'starthr' + "\t" + 'endhr' + "\t" + 'startfreq' + "\t" + 'endfreq' + "\t" + 'color' +"\n")

#filey = 'blobsy.txt'
#with open(filey, 'w') as outputy:
#    outputy.write('filename'+ "\t" + 'starthr' + "\t" + 'endhr' + "\t" + 'startfreq' + "\t" + 'endfreq' + "\t" + 'color' +"\n")

filez = 'blobsz.txt'
with open(filez, 'w') as outputz:
    outputz.write('filename'+ "\t" + 'starthr' + "\t" + 'endhr' + "\t" + 'startfreq' + "\t" + 'endfreq' + "\t" + 'color' +"\n")
#----------------------------------------------------------------------------------------------------------------------------------------------------------------


# 2) + 3) Loop through images and crop each

# Halley Image cropping dimensions (this assumes ALL spectrograms are being cropped to the same dimensions). Note: to not crop at all, set all dimensions to 0
#left = [36.181, 36.181, 36.181]
#top = [14.775, 95.082, 174.932]
#right = [279.693, 279.693, 279.693]
#bottom = [79.518, 159.407, 239.746]

#cropping dimensions for a resized MACCS spectrogram (down to 255 on the shortest side)
#left = [23.375, 23.375, 23.375]
#top = [7.8625, 86.7, 165.325]
#right = [289.425, 289.425, 289.425]
#bottom = [82.45, 161.075, 240.125]

#x = (23.375, 7.8625, 289.425, 82.45)
#y = (23.375, 86.7, 289.425, 161.075)
#z = (23.375, 165.325, 289.425, 240.125)

#cropping diemnsions for full size MACCS spectrograms
#x = (110, 37, 1362, 388)
#y = (110, 408, 1362, 758)
#z = (110, 778, 1362, 1130)

#cropping dimensions for MACCS spectrograms (resized by half)
x = (55, 18.5, 681, 194)
y = (55, 204, 681, 379)
z = (55, 389, 681, 565)


# Loop through positively-classified image filenames
for filename in os.listdir(input_folder):
    image_path = os.path.join(input_folder, filename)
    fullname=os.path.basename(filename)
    no_ext, ext=os.path.splitext(fullname)
    img = Image.open(image_path)
    
    #getting dimensions and resizing an image
    w, h = img.size
    img = img.resize(((int(600*(w/h))),600))
    w, h = img.size
   
    # Crop the current image
    imagex = img.crop(x)
    imagey = img.crop(y)
    imagez = img.crop(z)
    
    #get dimensions of cropped images
    wx, hx = imagex.size
    wy, hy = imagey.size
    wz, hz = imagez.size
    
    #converting images to numpy arrays
    imagex = np.array(imagex)
    imagey = np.array(imagey)
    imagez = np.array(imagez)
    
    #converting image arrays from RGB to BGR 
    imagex = cv.cvtColor(imagex, cv.COLOR_RGB2BGR)
    imagey = cv.cvtColor(imagey, cv.COLOR_RGB2BGR)
    imagez = cv.cvtColor(imagez, cv.COLOR_RGB2BGR)
    
    #opening the x spectrogram in a new window
    cv.namedWindow(f'{no_ext}')
    cv.imshow(f'{no_ext}', imagex)
    
    #interacting with it 
    cv.setMouseCallback(f'{no_ext}', on_click)
    cv.waitKey(0)
    cv.destroyWindow(f'{no_ext}')
