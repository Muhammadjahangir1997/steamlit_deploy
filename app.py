import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image, ImageOps

model = keras.models.load_model("mnist_model.h5")

st.title("Digit Pehchan App")
st.markdown("<h3 style='text-align: center;'>Apni likhi hui digit upload karein</h3>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Image upload karein (0-9)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    
    img_array = np.array(image).reshape(1, 784).astype("float32") / 255
    prediction = model.predict(img_array)
    result = np.argmax(prediction)
    
    st.image(image.resize((140, 140)), caption="Aapki Image")
    st.markdown(f"<h2 style='text-align: center;'>Model keh raha hai yeh number hai: {result}</h2>", unsafe_allow_html=True)
