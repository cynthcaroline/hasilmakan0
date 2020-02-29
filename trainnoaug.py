
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.layers import Input

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import SGD
from keras.regularizers import l2
import keras.backend as K
import math

from keras.preprocessing.image import ImageDataGenerator

def main(n_classes, epochs = 100, batch_size = 64):
    datagen = ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow_from_directory(
        directory='food-101/train/',
        target_size=(299,299),
        color_mode='rgb',
        batch_size=batch_size,
        seed=11
    )

    test_generator = datagen.flow_from_directory(
        directory='food-101/test/',
        target_size=(299,299),
        color_mode='rgb',
        batch_size=batch_size,
        seed=11
    )

    #create model
    base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(299, 299, 3)))
    x = base_model.output
    x = AveragePooling2D(pool_size=(8, 8))(x)
    x = Dropout(.4)(x)
    x = Flatten()(x)
    predictions = Dense(n_classes, kernel_initializer='glorot_uniform', kernel_regularizer=l2(.0005), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    opt = SGD(lr=.01, momentum=.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # callbacks
    base_path = 'model_noaugment/'
    patience = 50
    log_file_path = base_path + 'training.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                    patience=int(patience/4), verbose=1)
    early_stop = EarlyStopping('val_loss', patience=patience)
    trained_models_path = base_path + '_inceptionV3_'
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,
                                                        save_best_only=True)
    callbacks = [model_checkpoint, csv_logger,reduce_lr, early_stop]

    model.fit_generator(generator=train_generator,
        validation_data=test_generator,
        steps_per_epoch = train_generator.n // batch_size,
	    validation_steps = test_generator.n // batch_size,
        epochs=epochs,
        verbose=1,
        callbacks = callbacks
        )


if __name__ == '__main__':
    main(102)
