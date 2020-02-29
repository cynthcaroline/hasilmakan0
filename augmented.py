# Import library 
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img 
import glob  

datagen = ImageDataGenerator( 
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False, # randomly flip images
        zoom_range=[.8, 1],
        channel_shift_range=30,
        fill_mode='reflect')

l = 0
file = glob.glob("foodtrain/*.jpg")
for files in range(len(file)):
    # Load sampel gambar 
    img = load_img(file[l])
    # Ubah input sampel gambar ke array
    x = img_to_array(img)
    # Reshaping input image 
    x = x.reshape((1, ) + x.shape)
    # Membuat dan menyimpan 5 sampel yang augmented
    # dengan parameter diatas
    i = 0
    for batch in datagen.flow(x, batch_size = 1,
                              save_to_dir ='result',
                              save_prefix ='augmented',
                              save_format ='jpg'):
        i += 1
        if i > 5: 
            break
    l += 1
