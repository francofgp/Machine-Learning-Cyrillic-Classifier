import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2 as cv
from skimage.feature import hog
import numpy as np
import pickle


# importamos todo lo de pickle, modelo, Scaler y PCA
# para despues usarlos con esos nombres de variables
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


#
value = st.sidebar.selectbox(
    "Language/Idioma/Язык", ["English", "Español", "Русский"])
# Diccionario con todas las traducciones
text = {"English":
        {
            "instruction": "Draw a letter and hit the predict button",
            "title": "Cyrillic Classifier",
            "predict": "Predict",
            "prediction": "Prediction",
            "stroke width": "Stroke width",
            "color": "Stroke color HEX",
            "message": "See how it works!",
            "created": "Created by Pértile Franco Giuliano"
        },
        "Español":
        {"instruction": "Dibuje una letra, y presione el botón Predecir",
            "title": "Clasificador Cirílico",
            "predict": "Predecir",
            "prediction": "Predicción",
            "stroke width": "Grosor",
            "color": "Color del lápiz HEX",
            "message": "Vea cómo funciona!",

            "created": "Creado by Pértile Franco Giuliano"

         },
        "Русский":
        {"instruction": "Нарисуйте букву а потом нажмите кнопку Распознавать",
            "title": "Классификатор Кириллического",
            "predict": "Распознавать",
            "prediction": "Распознавание",
            "stroke width": "Ширина",
            "color": "Цвет карандаша (HEX)",
            "message": "Посмотрите, как работает!",

            "created": "Сделано Пертиле Франко Джулиано"

         }
        }
alfabeto = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
alfabeto_espacio = "А Б В Г Д Е Ё Ж З И Й К Л М Н О П Р С Т У Ф Х Ц Ч Ш Щ Ъ Ы Ь Э Ю Я"

st.title(text[value]["title"])
st.markdown(f"**{alfabeto_espacio}**")
st.markdown(f"{text[value]['instruction']}")
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


if canvas_result.image_data is not None and predict:
    # 1: Obtengo la imagen del canvas
    imagen = canvas_result.image_data
    imagen = imagen.astype('uint8')
    # 2: Recorto la imagen
    img_prueba_recortada = cv.resize(imagen, (64, 64))

    # 3:Hago Blanco y negro la imagen
    imagen_prueba_gris = cv.cvtColor(img_prueba_recortada, cv.COLOR_BGR2GRAY)
    # 4: Aplico filtro Gaussiano
    image_prueba_blur = cv.GaussianBlur(imagen_prueba_gris, (5, 5), 0)
    # 5: Obtengo la matriz HOG
    hog_prueba, hog_image = hog(image_prueba_blur, visualize=True, orientations=8,
                                block_norm='L2-Hys', pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    hog_prueba = hog_prueba.reshape(-1, 1)
    # 6: escalos los datos para PCA
    hog_prueba_scaled = sc.transform(hog_prueba.transpose())
    # 7: aplico PCA
    hog_prueba = pca.transform(hog_prueba_scaled)
    # 8: realizo la prediccion
    y_pred = model.predict(hog_prueba)

    # 9:Obtengo el la prediccion en un resultado mas visible, osea la letra
    res = {np.linspace(1, 33, 33)[i]: list(alfabeto)[i]
           for i in range(len(np.linspace(1, 33, 33)))}
    if y_pred[0] in res:
        st.success("{} : {}".format(text[value]["prediction"], res[y_pred[0]]))


st.markdown(
    f"{text[value]['message']} [GitHub](https://github.com/francofgp/Machine-Learning-Cyrillic-Classifier)")

st.markdown(f"#### *{text[value]['created']}* ")
