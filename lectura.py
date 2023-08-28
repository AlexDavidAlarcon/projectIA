import cv2
import numpy as np
import pytesseract
from gensim.models import KeyedVectors
from unidecode import unidecode  # quitar tildes
import re

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract"
img = cv2.imread("imagen2.png")
cv2.imshow("im",img)
color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

glove_file = 'embeddings-l-model.vec'

    # Carga los vectores GloVe
embeddings_model = KeyedVectors.load_word2vec_format(glove_file, binary=False, limit=50000) #, no_header=True
dic_lleno = {'vacante': 'gerente', 'compañía': 'disney', 'ciudadanía': 'ecuatoriano', 'nombre': 'moises', 'teléfono': '02562145'}
dic_vacio = {'cargo': '', 'empresa': '', 'nacionalidad': '', 'nombre': '', 'celular': '02562145' }

claves_formlleno = dic_lleno.keys()
claves_formvacio = dic_vacio.keys()
for clave1 in claves_formvacio:
    if clave1 not in claves_formlleno:
        print(f"Buscando sinónimos para clave '{clave1}':")

        # Verificar si la palabra clave existe en el vocabulario
        if clave1 in embeddings_model:
            similares = embeddings_model.most_similar(clave1, topn=50)
            palabras_similares = [similar[0] for similar in similares]

            print(palabras_similares)
            encontrada = False
            for palabra_similar in palabras_similares:
                if palabra_similar in claves_formlleno:
                    print(f"Palabra similar encontrada en dic_lleno: '{palabra_similar}'")
                    encontrada = True
                    dic_vacio[clave1] = dic_lleno[palabra_similar]
                    break  # Salir del bucle una vez que se encuentra una palabra similar

            if not encontrada:
                print(f"Palabra clave '{clave1}' no tiene sinónimos en el vocabulario")
        else:
            print(f"Palabra clave '{clave1}' no está presente en el vocabulario")
    else:
        dic_vacio[clave1] = dic_lleno[clave1]

    print('Ahora el campo en el diccionario vacio queda asi')
    print(dic_vacio)

    print('Ahora el campo en el diccionario vacio queda asi')
    print(dic_vacio)
