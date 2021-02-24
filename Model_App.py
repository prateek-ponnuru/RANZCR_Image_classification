import streamlit as st
from PIL import Image
from model import Model
from functions import InitializeLoadedModel, PredictForWebApp


st.title("Chest X-ray Image classification - CNN")

# Image Upload and Display from User
st.set_option('deprecation.showfileUploaderEncoding', False) # Deprecating all Warnings
img = st.file_uploader('Upload the Image',type='jpg')# IMAGE Feature
myModel = InitializeLoadedModel(model_path='models/best_model.hdf5')
if img:
    image = Image.open(img)
    st.image(img, caption= " The Uploaded Image ",use_column_width = True )

st.subheader("Predict the Classification")
if st.button("PREDICT"): # Predict Button

    st.write(" Under Progress...! ")

    preds = PredictForWebApp(image, myModel)
    st.write(preds)


# SIDEBARS
add_selectbox = st.sidebar.selectbox(
    "Selcet Images  ",
    ("A", "B")
)




