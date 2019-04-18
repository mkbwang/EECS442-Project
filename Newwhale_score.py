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
        x_train[count] = x
        if (count%500 == 0):
            print("Processing image: ", count+1, ", ", fig)
        count += 1
    
    return x_train

if __name__=="__main__":

    newwhale = pd.read_csv("subset_newwhale.csv")
    x_newwhale = prepareImages(newwhale, newwhale.shape[0], "train")
    x_newwhale /= 255.0
    model = load_model('data/model_crop.h5')
    predictions = model.predict(np.array(x_newwhale))
    np.save("data/newwhales_scores.npy", predictions)
