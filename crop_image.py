import pickle
from pandas import read_csv
import cv2
from PIL import Image as pil_image
from PIL.ImageDraw import Draw
from IPython.display import display
from tqdm import tqdm, tqdm_notebook

def crop_image(p, path):
    x0,y0,x1,y1 = p2bb[p]
    img = pil_image.open(path + p)
    crop_img = img.crop((x0, y0, x1, y1))
#     display(img)
#     display(crop_img)
    return crop_img


if __name__=="__main__":
    with open('bounding-box.pickle', 'rb') as f: 
        p2bb = pickle.load(f)
    new_tagged = [p for _,p,_ in read_csv('train.csv').to_records()]
    new_submit = [p for _,p,_ in read_csv('sample_submission.csv').to_records()]
    for train in tqdm_notebook(new_tagged):
        ci = crop_image(train, 'train/')
        ci.save('crop_train/'+train)
    print("Crop training image finished")
    for test in tqdm_notebook(new_submit):
        ci = crop_image(test, 'test/')
        ci.save('crop_test/'+test)
    print("Crop test image finished")
    