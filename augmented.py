# Importing necessary functions 
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img 
import glob  
# Initialising the ImageDataGenerator class. 
# We will pass in the augmentation parameters in the constructor. 

datagen = ImageDataGenerator( 
        rotation_range = 40, 
        shear_range = 0.2, 
        zoom_range = 0.2, 
        horizontal_flip = True, 
        brightness_range = (0.5, 1.5)) 

l = 0
file = glob.glob("fcv/*.jpg")
for files in range(len(file)):
    # Loading a sample image  
    img = load_img(file[l])
    # Converting the input sample image to an array 
    x = img_to_array(img)
    # Reshaping the input image 
    x = x.reshape((1, ) + x.shape)
    # Generating and saving 5 augmented samples  
    # using the above defined parameters.  
    i = 0
    for batch in datagen.flow(x, batch_size = 1,
                              save_to_dir ='result',
                              save_prefix ='image',
                              save_format ='jpg'):
        i += 1
        if i > 5: 
            break
    l += 1
