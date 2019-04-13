import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import os
import seaborn as sns
import matplotlib.pyplot as plt

if __name__=="__name__":
    train = pd.read_csv('train.csv')
    test = pd.read_csv('sample_submission.csv')
    train_frequency = train['Id'].value_counts()
    # train_frequency_subset = train_frequency[train_frequency['Id']>4]
    subset_frequency = list(train_frequency[train_frequency['Id']>4].index)
    np.save(os.path.join(dir, 'target_labels.npy'), subset_frequency)
    
