import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model= load_model('BrainTumor10EpochsCategorical.h5')
#/Users/alexazria/Desktop/2023-ML-SIDE-PROJECTS/Brain_Tumor_Classification/Good-dataset/Healthy/Not Cancer  (3).jpg
#/Users/alexazria/Desktop/2023-ML-SIDE-PROJECTS/Brain_Tumor_Classification/Good-dataset/tumor_test_predict/image(7).jpg
image=cv2.imread('/Users/alexazria/Desktop/2023-ML-SIDE-PROJECTS/Brain_Tumor_Classification/Good-dataset/tumor_test_predict/image(112).jpg')
img = Image.fromarray(image)
img = img.resize((64,64))
img = np.array(img)

input_img = np.expand_dims(img,axis=0) #To have the right dimension of the images as expected
result = model.predict(input_img)
result_final = np.argmax(result,axis=1)
if result_final ==0:
    print("you do not have a brain tumor")
else:
    print("Unfortunately you do have a brain tumor...")
#print(result_final)