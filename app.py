from matplotlib import widgets
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2 as cv
from skimage.feature import hog
from skimage.color import rgb2grey, rgb2gray, rgba2rgb
from PIL import Image
import numpy as np
import pandas as pd
import time
import pickle
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# importamos todo lo de pickle, modelo, Scaler y PCA
pkl_filename = "sc.pkl"
with open(pkl_filename, 'rb') as file:
    sc = pickle.load(file)

pkl_filename = "model.pkl"
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)

pkl_filename = "pca.pkl"
with open(pkl_filename, 'rb') as file:
    pca = pickle.load(file)

st.set_page_config(
    page_title="Cyrillic Classifier",
    page_icon=":pencil:",
)


hide_streamlit_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

value = st.sidebar.selectbox(
    "Language/Idioma/Язык", ["English", "Español", "Русский"])

text = {"English":
        {
            "title": "Cyrillic Classifier",
            "predict": "Predict",
            "prediction": "Prediction",
            "stroke width": "Stroke width",
            "color": "Stroke color HEX"
        },
        "Español":
        {
            "title": "Clasificador Cirílico",
            "predict": "Predecir",
            "prediction": "Predicción",
            "stroke width": "Grosor",
            "color": "Color del lápiz HEX"
        },
        "Русский":
        {
            "title": "Классификатор Кириллического",
            "predict": "Распознавать",
            "prediction": "Распознавание",
            "stroke width": "Шинина",
            "color": "Цвет карандаша ШСС (HEX)"
        }
        }
alfabeto = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
alfabeto_espacio = "А Б В Г Д Е Ё Ж З И Й К Л М Н О П Р С Т У Ф Х Ц Ч Ш Щ Ъ Ы Ь Э Ю Я"

st.title(text[value]["title"])
st.markdown(f"**{alfabeto_espacio}**")

# Specify brush parameters and drawing mode
stroke_width = st.sidebar.slider("{}: ".format(
    text[value]["stroke width"]), 1, 100, 25)

stroke_color = st.sidebar.color_picker("{}:".format(text[value]["color"]))

# Create a canvas component
canvas_result = st_canvas(
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color="#FFF",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)


predict = st.button(text[value]["predict"])


def get_prediction(image):
    pass


    # get the prediction from your model and return it
if canvas_result.image_data is not None and predict:
    imagen = canvas_result.image_data
    imagen = imagen.astype('uint8')

    img_prueba_recortada = cv.resize(imagen, (64, 64))
    #st.text("Imagen recortada")
    # st.image(img_prueba_recortada.astype('int64'))

    #st.text("Imagen blanco y negro")
    imagen_prueba_gris = cv.cvtColor(img_prueba_recortada, cv.COLOR_BGR2GRAY)
    #st.image(cv.resize(img_prueba_recortada, (256, 256)).astype('int64'))

    #st.text("Imagen blur")
    image_prueba_blur = cv.GaussianBlur(imagen_prueba_gris, (5, 5), 0)
    #st.image(cv.resize(image_prueba_blur, (256, 256)).astype('int64'))

    #st.text("Imagen HOG")
    hog_prueba, hog_image = hog(image_prueba_blur, visualize=True, orientations=8,
                                block_norm='L2-Hys', pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    hog_prueba = hog_prueba.reshape(-1, 1)
    #st.image(cv.resize(hog_prueba, (256, 256)).astype('int64'))

    hog_prueba_scaled = sc.transform(hog_prueba.transpose())

    hog_prueba = pca.transform(hog_prueba_scaled)

    y_pred = model.predict(hog_prueba)
    res = {np.linspace(1, 33, 33)[i]: list(alfabeto)[i]
           for i in range(len(np.linspace(1, 33, 33)))}
    if y_pred[0] in res:
        st.success("{} : {}".format(text[value]["prediction"], res[y_pred[0]]))

    #st.text("Prediction : {}".format(prediction))
    # st.balloons()
    #outputs = predict(canvas_result.image_data)
