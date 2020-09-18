from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import numpy as np

import os

# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(240, 320))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 240, 320, 3)
    # center pixel data
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img
 
# load an image and predict the class
def run_example(filepath):
    # load the image
    img = load_image(filepath)
    # load model

    # predict the class
    result = model.predict(img)
    f.write(str(result[0])+"\n")
    result_ = result.argmax(axis=-1)

    if (result[0][0] < 0.5) :
      tmptxt = filename + " @@@@@@ b l u r @@@@@@" +"\n"
      f.write(tmptxt)
    else :
      tmptxt = filename + " ######## n o n   b l u r ########" +"\n"
      f.write(tmptxt)

 
# entry point, run the example

path_dir = "/PATH_TO_DIR/"
file_list = os.listdir(path_dir)
file_list.sort()
i = 0
print("start!")
model = load_model('/PATH_TO_MODEL_.h5_FILE')
print("model loaded")
f = open('/PATH_TO_SAVE_TXT_FILE', mode='wt', encoding='utf-8')

for filename in file_list:
    i = i+1
    f.write(str(i)+"\n")
    filepath = path_dir + filename
    run_example(filepath)
    K.clear_session()

f.close()    
