import streamlit as st
import numpy  as np
import pandas as pd
import cv2
import os
from keras.models import load_model
from PIL import Image
import numpy as np
import jinja2

model= load_model('BrainTumor10EpochsCategorical.h5')

st.title('Hello, I am your radiologist AI assistant! Download the X-ray of your brain to see if there is a tumor or not. :smiley:')
uploaded_file = st.file_uploader('Upload your X-Ray Here 🤞!')

def predict():
    #image = Image.open(uploaded_file)
    #st.image(image)
    image=cv2.imread(uploaded_file.name)
    img = Image.fromarray(image)
    img = img.resize((64,64))
    input_img = np.expand_dims(np.array(img),axis=0) #To have the right dimension of the images as expected
    result = model.predict(input_img)
    result_final = np.argmax(result,axis=1)
    
    if result_final == 0:
        st.success('You do not have a Brain Tumor !')
    else:
        st.error('Please go see your patient, it seems that she/he has a brain tumor...')

st.button('Prediction', on_click=predict)


#c1, c2= st.columns(2)
# if uploaded_file is not None:
#     im= Image.open(uploaded_file)
#     img = np.asarray(im)
#     img = img.resize((64,64))
#     img= np.expand_dims(img, axis=0)
#     c1.header('Your Scanner')
#     c1.image(im)
#     c1.write(img.shape)

#     # prediction on model
#     model = model.predict(img)
#     result_final = np.argmax(model,axis=1)
#     c2.header('Result :')
#     c2.subheader('Predicted class :question:')
#     if result_final == 0:
#         c2.write("You do not have a brain Tumor !")
#     else:
#         c2.write("Please go see your patient, it seems that she/he has a brain tumor...")