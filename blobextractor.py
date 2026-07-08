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
#The following program structure was inspired by Dr. Nyssa Capman's CNN_Training.py script
#Attempting to create a file that extracts blobs and writes their properties onto a text file
# 0) First define functions to map pixel locations to the corresponding time and frequency on spectrogram
#NOTE: the time is not evenly divided on the spectrograms. Some hour ranges are 10.8375 pixels wide where the largest hour range is 11.899 pixels wide.
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

# 1) ---------------------------INPUT AND OUTPUT DIRECTORIES----------------------------------------------------------------------------------------------------
input_folder='C:/users/ethio/Zenodo/nain2023jpg/2'
output_folder= 'C:/users/ethio/Zenodo/nain2023jpg/'
#----------------------------------------------------------------------------------------------------------------------------------------------------------------

#opening textfiles in the current directory to write blob information on 
filex = 'blobsx.txt'
with open(filex, 'w') as outputx:
    outputx.write('filename'+ "\t" + 'starthr' + "\t" + 'endhr' + "\t" + 'startfreq' + "\t" + 'endfreq' +"\n")

filey = 'blobsy.txt'
with open(filey, 'w') as outputy:
    outputy.write('filename'+ "\t" + 'starthr' + "\t" + 'endhr' + "\t" + 'startfreq' + "\t" + 'endfreq' +"\n")

filez = 'blobsz.txt'
with open(filez, 'w') as outputz:
    outputz.write('filename'+ "\t" + 'starthr' + "\t" + 'endhr' + "\t" + 'startfreq' + "\t" + 'endfreq' +"\n")

# 2) + 3) Loop through images and crop each

# Image cropping dimensions (this assumes ALL spectrograms are being cropped to the same dimensions). Note: to not crop at all, set all dimensions to 0
#left = [36.181, 36.181, 36.181]
#top = [14.775, 95.082, 174.932]
#right = [279.693, 279.693, 279.693]
#bottom = [79.518, 159.407, 239.746]

#cropping dimensions for Nain spectrogram
#left = [23.375, 23.375, 23.375]
#top = [7.8625, 86.7, 165.325]
#right = [289.425, 289.425, 289.425]
#bottom = [82.45, 161.075, 240.125]

x = (23.375, 7.8625, 289.425, 82.45)
y = (23.375, 86.7, 289.425, 161.075)
z = (23.375, 165.325, 289.425, 240.125)


# Loop through positively-classified image filenames
for filename in os.listdir(input_folder):      
    image_path = os.path.join(input_folder, filename)
    fullname=os.path.basename(filename)
    no_ext, ext=os.path.splitext(fullname)
    img = Image.open(image_path)
    
    #getting dimensions and resizing an image
    w, h = img.size
    img = img.resize(((int(255*(w/h))),255))
    w, h = img.size
   
    # Crop the current image
    imagex = img.crop(x)
    imagey = img.crop(y)
    imagez = img.crop(z)
    
    #get dimensions of cropped images
    wx, hx = imagex.size
    wy, hy = imagey.size
    wz, hz = imagez.size
    
    image_grayx = rgb2gray(imagex)
    image_grayy = rgb2gray(imagey)
    image_grayz = rgb2gray(imagez)
   
    
    #img.show(image_grayx)
    
    
    #apply processing to each subimage (smoothing, canny filter with sigma=2.3)
    imagegrayx=np.array(image_grayx)
    smoothedx = restoration.denoise_bilateral(imagegrayx, sigma_spatial=1)
    #process differently based on the mean pixel value of the cropped original image (before any smoothing), thresholding was more effective when applied to quieter images.
    #Mean was chosen instead of median because some images have black regions followed by noisy segments, lowering the median value
    v=np.mean(imagex)
    if v<=40:
        threshold = 0.2
        threshx = smoothedx > threshold
        edges_x = canny(threshx, sigma = 3.6)
    elif v>40 and v<=72:
        edges_x = canny(smoothedx, sigma = 3.4)
    elif v>72:
       edges_x = canny(smoothedx, sigma=2.3)
       
    
    imagegrayy=np.array(image_grayy)
    smoothedy = restoration.denoise_bilateral(imagegrayy, sigma_spatial=1)
    v=np.mean(imagey)
    if v<=40:
        threshold = 0.2
        threshy = smoothedy > threshold
        edges_y = canny(threshy, sigma = 3.6)
    elif v>40 and v<=72:
        edges_y = canny(smoothedy, sigma = 3.4)
    elif v>72:
        edges_y = canny(smoothedy, sigma=2.3)
    
    imagegrayz=np.array(image_grayz)
    smoothedz = restoration.denoise_bilateral(imagegrayz, sigma_spatial=1)
    v=np.mean(imagez)
    if v<=40:
        threshold = 0.2
        threshz = smoothedz > threshold
        edges_z = canny(threshz, sigma = 3.6)
    elif v>40 and v<=72:
        edges_z =canny(smoothedz, sigma = 3.4)
    elif v>72:     
        edges_z = canny(smoothedz, sigma=2.3)



#----------------------------------------------------------------------------------------------------------------------------------    
    '''Blob detection using regionprops
    In an attempt to capture good EMIC events and avoid thermal noise, many properties of the shape including area, height, width and eccentricty of each detected shape will be used.  
    In an attempt to capture good events that start below 0.2Hz, the ymax value will be lowered upto 69. 
    centroid that is lower than 62 in less noisy images and centroid less than 50 in more noisy images will be applied along with solidity>=0.1 to 
    minimize the number of non events detected.'''    
    
    
    ## for x spectrograms
    labeled_image_x = measure.label(edges_x)
    regions_x = measure.regionprops(labeled_image_x)
    
    for i, region in enumerate(regions_x):
        ymin, xmin , ymax, xmax = region.bbox
        rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, edgecolor='red', linewidth=2)
        area = region.area
        ecc = region.eccentricity
        sol = region.solidity
        cy, cx = region.centroid
        if v<=72:
            if ((ymax-ymin)<50 and (ymax-ymin)>= 8 and (xmax-xmin)>5.5427) and (ymax<60 or (ymax<69 and sol>=0.15 and area>2 and cy<=62)): 
                #y=60 is about 0.2Hz at which most noisy structures stop
                starthr = time_mapper(xmin, wx, hx)
                endhr = time_mapper(xmax, wx, hx)
                startfreq = freq_mapper(ymax, wx, hx)
                endfreq = freq_mapper(ymin, wx, hx)
                with open(filex, 'a') as f:
                    f.write(no_ext + "\t" + f'{starthr}' + "\t" + f'{endhr}' + "\t" + f'{startfreq}' + "\t" + f'{endfreq}' +"\n")
                    #print(f"Region{i+1}: Area ={region.area}").....convert corners and write properties to file here
        elif v>72: 
            if ((ymax-ymin)<50 and (ymax-ymin)>= 8 and (xmax-xmin)>5.5427) and (ymax<60 or (ymax<69 and sol>=0.15 and area>2 and ecc<=0.95 and cy<=50)):
                #y=60 is about 0.2Hz at which most noisy structures stop
                starthr = time_mapper(xmin, wx, hx)
                endhr = time_mapper(xmax, wx, hx)
                startfreq = freq_mapper(ymax, wx, hx)
                endfreq = freq_mapper(ymin, wx, hx)
                with open(filex, 'a') as f:
                    f.write(no_ext + "\t" + f'{starthr}' + "\t" + f'{endhr}' + "\t" + f'{startfreq}' + "\t" + f'{endfreq}' +"\n")
        
 ## for y spectrograms
    labeled_image_y = measure.label(edges_y)
    regions_y = measure.regionprops(labeled_image_y)
    
    for i, region in enumerate(regions_y):
        ymin, xmin , ymax, xmax = region.bbox
        rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, edgecolor='red', linewidth=2)
        area = region.area
        ecc = region.eccentricity
        sol = region.solidity
        cy, cx = region.centroid
        if v<=72:
            if ((ymax-ymin)<50 and (ymax-ymin)>= 8 and (xmax-xmin)>5.5427) and (ymax<60 or (ymax<69 and sol>=0.15 and area>2 and cy<=62)): 
                #y=60 is about 0.2Hz at which most noisy structures stop
                starthr = time_mapper(xmin, wy, hy)
                endhr = time_mapper(xmax, wy, hy)
                startfreq = freq_mapper(ymax, wy, hy)
                endfreq = freq_mapper(ymin, wy, hy)
                with open(filey, 'a') as f:
                    f.write(no_ext + "\t" + f'{starthr}' + "\t" + f'{endhr}' + "\t" + f'{startfreq}' + "\t" + f'{endfreq}' +"\n")
                    #print(f"Region{i+1}: Area ={region.area}").....convert corners and write properties to file here
        elif v>72: 
            if ((ymax-ymin)<50 and (ymax-ymin)>= 8 and (xmax-xmin)>5.5427) and (ymax<60 or (ymax<69 and sol>=0.15 and area>2 and ecc<=0.95 and cy<=50)):
                #y=60 is about 0.2Hz at which most noisy structures stop
                starthr = time_mapper(xmin, wy, hy)
                endhr = time_mapper(xmax, wy, hy)
                startfreq = freq_mapper(ymax, wy, hy)
                endfreq = freq_mapper(ymin, wy, hy)
                with open(filey, 'a') as f:
                    f.write(no_ext + "\t" + f'{starthr}' + "\t" + f'{endhr}' + "\t" + f'{startfreq}' + "\t" + f'{endfreq}' +"\n")
        
 ## for z spectrograms
    labeled_image_z = measure.label(edges_z)
    regions_z = measure.regionprops(labeled_image_z)
    
    for i, region in enumerate(regions_z):
        ymin, xmin , ymax, xmax = region.bbox
        rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, edgecolor='red', linewidth=2)
        area = region.area
        ecc = region.eccentricity
        sol = region.solidity
        cy, cx = region.centroid
        if v<=72:
            if ((ymax-ymin)<50 and (ymax-ymin)>= 8 and (xmax-xmin)>5.5427) and (ymax<60 or (ymax<69 and sol>=0.15 and area>2 and cy<=62)): 
                #y=60 is about 0.2Hz at which most noisy structures stop
                starthr = time_mapper(xmin, wz, hz)
                endhr = time_mapper(xmax, wz, hz)
                startfreq = freq_mapper(ymax, wz, hz)
                endfreq = freq_mapper(ymin, wz, hz)
                with open(filez, 'a') as f:
                    f.write(no_ext + "\t" + f'{starthr}' + "\t" + f'{endhr}' + "\t" + f'{startfreq}' + "\t" + f'{endfreq}' +"\n")
                    #print(f"Region{i+1}: Area ={region.area}").....convert corners and write properties to file here
        elif v>72: 
            if ((ymax-ymin)<50 and (ymax-ymin)>= 8 and (xmax-xmin)>5.5427) and (ymax<60 or (ymax<69 and sol>=0.15 and area>2 and ecc<=0.95 and cy<=50)):
                #y=60 is about 0.2Hz at which most noisy structures stop
                starthr = time_mapper(xmin, wz, hz)
                endhr = time_mapper(xmax, wz, hz)
                startfreq = freq_mapper(ymax, wz, hz)
                endfreq = freq_mapper(ymin, wz, hz)
                with open(filez, 'a') as f:
                    f.write(no_ext + "\t" + f'{starthr}' + "\t" + f'{endhr}' + "\t" + f'{startfreq}' + "\t" + f'{endfreq}' +"\n")


end_time= datetime.now()
run_time = end_time-start_time
run_time=(run_time.total_seconds())/60

print ('Run Time: ', run_time, 'minutes')
# Program is finished
import winsound
duration1 = 500  # milliseconds
duration2 = 250
freq1 = 440  # Hz
freq2 = 554
freq3 = 659
winsound.Beep(freq1, duration1)
winsound.Beep(freq2, duration2)
winsound.Beep(freq3, duration1)
winsound.Beep(freq2, duration2)
winsound.Beep(freq1, duration1)