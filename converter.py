from PIL import Image
import os
from os import listdir
import numpy as np

#img_png=Image.open('.\Nain2022')
#img_png.save('.\newnain2022')
#def png_to_jpg (input_folder, output_folder):
#    img=Image.open(input_folder)
#    rgb_img=img.convert('RGB')
#    conv =rgb_img.save('output_folder/image_path.jpg')
#   return conv
#--------------------------------------------------------------------------------------------------------------------------------------    

#-------INPUT AND OUTPUT DIRECTORIES----------------------------------------------------------------------------------------------------
input_folder='C:/users/ethio/Zenodo/CDRjpg/png'
output_folder= 'C:/users/ethio/Zenodo/CDRjpg/'
#---------------------------------------------------------------------------------------------------------------------------------------


#converting each image in the input folder to jpg format and saving it into the output folder while keeping the original filename
for filename in os.listdir(input_folder):
   image_path = os.path.join(input_folder, filename)
   fullname=os.path.basename(filename)
   no_ext, ext=os.path.splitext(fullname)
   #no_ext=no_ext.append()
   #print(len(no_ext))
   #for i in len(no_ext):
   img=Image.open(image_path)
   rgb_img=img.convert('RGB')
   full_output_path=os.path.join(output_folder + no_ext + '.jpg')
   converted =rgb_img.save(full_output_path)
    










