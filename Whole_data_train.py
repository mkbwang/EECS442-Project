import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from keras.models import Sequential # to create a cnn model
from keras.models import load_model, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input
from keras.applications.vgg16 import VGG16
from keras.optimizers import RMSprop,Adam
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import pickle

def prepareImages(train, shape, path):
    
    x_train = np.zeros((shape, 224, 224, 3))
    count = 0
    
    for fig in train['Image']:
        
        #load images into images of size 100x100x3
        img = image.load_img(os.path.join(path, fig), target_size=(224, 224, 3))
        # print(img.shape)
        x = image.img_to_array(img)
        x = preprocess_input(x)
        x_train[count] = np.copy(x/255.0)
        if (count%500 == 0):
            print("Processing image: ", count+1, ", ", fig)
        count += 1
    
    return x_train



if __name__=="__main__":

    # load training images and their ids
    train = pd.read_csv('train_withid.csv')
    # imgnames = train['Image'].tolist()

    x_train = prepareImages(train, train.shape[0], "crop_train")
    # x_train = np.load('data/train_processed.npy')
    # x_train = x_train/255.0
    
    # X_train = train.drop(labels=['Id'], axis=1)
    y_train = train["Id"]

    # np.save('data/train_processed_224.npy', x_train)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    # np.save("data/label_id_total.npy", y_train)
    y_train = to_categorical(y_train, num_classes = 5004)

    # validation data
    validation = pd.read_csv('train_validation.csv') 
    x_valid = prepareImages(validation, validation.shape[0], "crop_train")
    y_valid = validation['Id']
    y_valid = label_encoder.transform(y_valid)
    y_valid = to_categorical(y_valid, num_classes = 5004)


    # load test images and their labels
    test = pd.read_csv('sample_submission.csv')
    test_data = pd.DataFrame(list(test['Image']), columns=['Image'])
    test_data['Id'] = ''
    x_test = prepareImages(test_data, test_data.shape[0], "crop_test")
    # y_test = label_encoder.transform(test['Id'])
    # y_test = to_categorical(y_test, num_classes=5004)

    # Introduce data augmentation, imgaug
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    print('Start data augmentation')
    datagen.fit(x_train)

    # if you want to use VGG, uncomment the code below
    # model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
    # input = Input(shape=(100,100,3),name = 'image_input')
    # output_vgg16_conv = model_vgg16_conv(input)
    # x = Flatten(name='flatten')(output_vgg16_conv)
    # x = Dense(805, activation='softmax', name='predictions')(x)
    # model = Model(input=input, output=x)

    

    # if you want to use a simple network, uncomment the code below
    model = Sequential()

    model.add(Conv2D(filters = 16, kernel_size = (5,5), padding = 'Same', activation = 'relu', input_shape = (224, 224, 3)))
    model.add(Conv2D(filters = 16, kernel_size = (5,5), padding = 'Same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = (2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
    model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = (2,2), strides=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = (2,2), strides=(2,2)))
    model.add(Dropout(0.25))

    # fully connected
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dense(y_train.shape[1], activation = "softmax"))

    optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
    
    # # compile model
    model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])

    epochs = 100  # for better result increase the epochs
    batch_size = 300

    # model = load_model('data/model_0.h5')

    # Fit the neural net (without data augmentation)
    # history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[learning_rate_reduction],validation_data=(X_test, Y_test))
    
    # fit the neural net with data augmentation
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                              epochs=100, verbose = 2, 
                              steps_per_epoch=x_train.shape[0] // batch_size,
                              callbacks=[learning_rate_reduction],
                              validation_data=(x_valid, y_valid))

    # Check test accuracy
    

    # to save model uncomment the code below

    model.save('data/model_total_crop_aug.h5')
    with open("data/history_total_crop_aug.pkl", 'wb+') as f:
        pickle.dump(history.history, f)

    predictions = model.predict(np.array(x_test))

    np.save("data/Predictions_total_crop_aug.npy", predictions)# save the raw predicted probabilities
    for i, pred in enumerate(predictions):
        scores = pred[pred.argsort()[-5:]]
        newwhale_location = np.where(scores<0.05)[0]
        candidates = label_encoder.inverse_transform(pred.argsort()[-5:][::-1])
        if len(newwhale_location)>0:
            candidates = np.insert(candidates, 4-np.amax(newwhale_location), 'new_whale')
            candidates = candidates[:-1]        
        test_data.loc[i, 'Id'] = ' '.join(candidates)
    test_data.to_csv("data/labels_total_crop_aug.csv", index=False)# save output csv file
