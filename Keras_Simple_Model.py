import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from keras.models import Sequential # to create a cnn model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

def prepareImages(train, shape, path):
    
    x_train = np.zeros((shape, 100, 100, 3))
    count = 0
    
    for fig in train['Image']:
        
        #load images into images of size 100x100x3
        img = image.load_img(os.path.join(path, fig), target_size=(100, 100, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)

        x_train[count] = x
        if (count%500 == 0):
            print("Processing image: ", count+1, ", ", fig)
        count += 1
    
    return x_train



if __name__=="__main__":
    train = pd.read_csv('subset_train.csv')
    imgnames = train['Image'].tolist()

    x_train = prepareImages(train, train.shape[0], "train")
    x_train = x_train/255.0
    
    X_train = train.drop(labels=['Id'], axis=1)
    y_train = train["Id"]

    np.save('data/train_processed.npy', x_train)

    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(y_train)
    np.save("data/label_id.npy", y_train)

    y_train = to_categorical(y_train, num_classes = 805)

    # initialize the network
    model = Sequential()

    model.add(Conv2D(filters = 16, kernel_size = (5,5), padding = 'Same', activation = 'relu', input_shape = (100,100,3)))
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
    
    # compile model
    model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])

    epochs = 80  # for better result increase the epochs
    batch_size = 500

    # Fit the neural net
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[learning_rate_reduction])


    # Check test accuracy
    test = pd.read_csv('subset_test.csv')

    test_data = pd.DataFrame(list(test['Image']), columns=['Image'])
    test_data['Id'] = ''

    x_test = prepareImages(test_data, test_data.shape[0], "train")
    x_test = x_test/255.0

    predictions = model.predict(np.array(x_test), verbose=1)

    np.save("data/Predictions.npy", predictions)
    for i, pred in enumerate(predictions):
        test_data.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))
    
    test_data.to_csv("Predicted_labels.csv", index=False)
