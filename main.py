import cv2
import numpy as np
import pytesseract
from gensim.models import KeyedVectors
from unidecode import unidecode  # quitar tildes

# Configurar Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract"

# Cargar la imagen y convertirla a escala de grises
img_lleno = cv2.imread("imagen2.png")
img_vacio = cv2.imread("imagen_sinonimo.png")

def leerFormulario(img):
    color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow("gray", gray)
    cv2.waitKey(0)

    # Calcular la mediana del color de fondo para determinar si es color o blanco
    median_color = np.median(color)
    background_is_white = median_color > 245  # Ajusta este valor según tus necesidades

    # Si el fondo es de color, utilizar la imagen umbralizada, de lo contrario usar la escala de grises
    if background_is_white:
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
    else:
        thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        binary = cv2.adaptiveThreshold(
            thresholded, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )


    cv2.imshow("binary", binary)
    cv2.waitKey(0)

    # Buscar contornos en la imagen binarizada
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lista para almacenar las regiones de los textbox
    textbox_regions = []

    for contour in contours:
        # Filtrar contornos por área y forma aproximada a un rectángulo
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        # Ajustar valores según el tamaño de tus campos
        if 500 < area:
            textbox_regions.append((x, y, x + w, y + h))
    print("x, y, x + w, y + h:", x, y, x + w, y + h)

    # Procesar el contenido de cada región
    textbox_contents = {}

    for region in textbox_regions:
        x1, y1, x2, y2 = region
        print("x, y, x2, y2", x1, y1, x2 , y2)

        roi = gray[y1:y2, x1:x2]
        content = pytesseract.image_to_string(roi, lang="spa")
        textbox_contents[(x1, y1, x2, y2)] = content

    # Dibujar las coordenadas en la imagen
    for coords in textbox_contents.keys():
        x1, y1, x2, y2 = coords
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Procesar y almacenar el contenido a la izquierda de cada región de textbox en un diccionario
    contents_dictionary = {}

    for coords, content in textbox_contents.items():
        x1, y1, x2, y2 = coords
        roi_left = gray[y1:y2, 0:x1]
        content_left = pytesseract.image_to_string(roi_left, lang="spa")
        # Eliminar espacios en blanco alrededor de la clave y valor

        texto_sin_tildes = unidecode(content_left)
        texto_sin_tilde = unidecode(content)
        #Quitar espacios
        content_left = texto_sin_tildes.strip()
        content = texto_sin_tilde.strip()
        # Quitar el punto al final del texto, si existe
        if content_left.endswith('.'):
            content_left = content_left[:-1]
        elif content.endswith("."):
            content = content[:-1]

        contents_dictionary[content_left] = content

    # Mostrar el contenido almacenado en el diccionario
    for clave, valor in contents_dictionary.items():
        print(f"Clave: {clave}, Valor: {valor}")

    print(contents_dictionary)
    # Mostrar la imagen con las coordenadas dibujadas
    cv2.imshow("Textbox Coords", img)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return contents_dictionary


dic_lleno = leerFormulario(img_lleno)

dic_vacio = leerFormulario(img_vacio)

print("diccionario lleno: ", dic_lleno)
print("diccionario vacio: ", dic_vacio)

def llenar_diccionario(dic_lleno,dic_vacio):
    # Obtener las claves de ambos diccionarios
    claves_formlleno = dic_lleno.keys()
    claves_formvacio = dic_vacio.keys()
    # Cambia la ruta al archivo GloVe que tienes
    glove_file = 'embeddings-l-model.vec'

    # Carga los vectores GloVe
    embeddings_model = KeyedVectors.load_word2vec_format(glove_file, binary=False, limit=50000) #, no_header=True

    for clave1 in claves_formvacio:
        if clave1 not in claves_formlleno:
                print(f"Buscando sinónimos para clave '{clave1}':")
                similares = embeddings_model.most_similar(clave1, topn = 50) #topn=50
                palabras_similares = [similar[0] for similar in similares]
                print(palabras_similares)
                encontrada = False
                for palabra_similar in palabras_similares:
                    if palabra_similar in claves_formlleno:
                        print(f"Palabra similar encontrada en dic_lleno: '{palabra_similar}'")
                        encontrada = True
                        dic_vacio[clave1]=dic_lleno[palabra_similar]
                        break  # Salir del bucle una vez que se encuentra una palabra similar
        else:
            dic_vacio[clave1] = dic_lleno[clave1]

    print('Ahora el campo en el diccionario vacio queda asi')
    print(dic_vacio)
    return dic_vacio


diccionario = llenar_diccionario(dic_lleno,dic_vacio)
print("diccionario llenado:",diccionario)

# # Escribir texto
# cv2.putText(img_vacio, texto, ubicacion, font, tamañoLetra, colorLetra, grosorLetra)


# def escribirImagen(img, diccionario):
#     color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     cv2.imshow("gray", gray)
#     cv2.waitKey(0)
#
#     # Calcular la mediana del color de fondo para determinar si es color o blanco
#     median_color = np.median(color)
#     background_is_white = median_color > 245  # Ajusta este valor según tus necesidades
#
#     # Si el fondo es de color, utilizar la imagen umbralizada, de lo contrario usar la escala de grises
#     if background_is_white:
#         binary = cv2.adaptiveThreshold(
#             gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
#         )
#     else:
#         thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#         binary = cv2.adaptiveThreshold(
#             thresholded, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
#         )
#
#     cv2.imshow("binary", binary)
#     cv2.waitKey(0)
#
#     cv2.imshow("Imagen llenada", img)
#     cv2.waitKey(0)
#
#     # Buscar contornos en la imagen binarizada
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Lista para almacenar las regiones de los textbox
#     textbox_regions = []
#
#     for contour in contours:
#         # Filtrar contornos por área y forma aproximada a un rectángulo
#         area = cv2.contourArea(contour)
#         x, y, w, h = cv2.boundingRect(contour)
#
#         # Ajustar valores según el tamaño de tus campos
#         if 500 < area:
#             textbox_regions.append((x, y, x + w, y + h))
#
#     # Procesar el contenido de cada región
#     textbox_contents = {}
#
#     for region in textbox_regions:
#         x1, y1, x2, y2 = region
#         roi = gray[y1:y2, x1:x2]
#         content = pytesseract.image_to_string(roi, lang="spa")
#         textbox_contents[(x1, y1, x2, y2)] = content
#
#     # Dibujar las coordenadas en la imagen
#     for coords in textbox_contents.keys():
#         x1, y1, x2, y2 = coords
#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
#     # Procesar y almacenar el contenido a la izquierda de cada región de textbox en un diccionario
#     # contents_dictionary = {}
#
#     for coords, content in textbox_contents.items():
#         x1, y1, x2, y2 = coords
#         roi_left = gray[y1:y2, 0:x1]
#         content_left = pytesseract.image_to_string(roi_left, lang="spa")
#         # Eliminar espacios en blanco alrededor de la clave y valor
#         #
#         # texto_sin_tildes = unidecode(content_left)
#         # texto_sin_tilde = unidecode(content)
#         # # Quitar espacios
#         # content_left = texto_sin_tildes.strip()
#         # content = texto_sin_tilde.strip()
#         # # Quitar el punto al final del texto, si existe
#         # if content_left.endswith("."):
#         #     content_left = content_left[:-1]
#         # elif content.endswith("."):
#         #     content = content[:-1]
#         #
#         # contents_dictionary[content_left] = content
#     # Características del texto
#     for region in textbox_regions:
#         x1, y1, x2, y2 = region
#
#     texto = "Quito"
#     ubicacion = (10, 25)
#     font = cv2.FONT_HERSHEY_TRIPLEX
#     tamañoLetra = 10
#     colorLetra = (0,0,0)
#     grosorLetra = 10
#     for dic in diccionario.keys():
#         if content_left == dic:
#             cv2.putText(img, texto, ubicacion, font, tamañoLetra, colorLetra, grosorLetra)
#
#
#     # Mostrar la imagen
#     cv2.imshow("Imagen llenada", img)
#     cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
#
# lista = escribirImagen(img_vacio,diccionario)
# print("lista:",lista)


def llenar_campos_en_imagen(img, dic_lleno, dic_vacio):

    color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow("gray", gray)
    cv2.waitKey(0)

    # Calcular la mediana del color de fondo para determinar si es color o blanco
    median_color = np.median(color)
    background_is_white = median_color > 245  # Ajusta este valor según tus necesidades

    # Si el fondo es de color, utilizar la imagen umbralizada, de lo contrario usar la escala de grises
    if background_is_white:
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
    else:
        thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        binary = cv2.adaptiveThreshold(
            thresholded, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

    cv2.imshow("binary", binary)
    cv2.waitKey(0)

    # Obtener el contenido de la imagen
    img_content = pytesseract.image_to_string(gray, lang="spa")

    lista_content= img_content.split("\n")

    lista_contenido = []
    espacio = ""
    for contenido in lista_content:
        print(contenido)
        if contenido is not espacio:
            lista_contenido.append(contenido)
    print("contenido de imagen:",img_content)
    print("contenido de imagen:",lista_contenido)

    # Buscar contornos en la imagen binarizada
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lista para almacenar las regiones de los textbox
    textbox_regions = []

    for contour in contours:
        # Filtrar contornos por área y forma aproximada a un rectángulo
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        # Ajustar valores según el tamaño de tus campos
        if 500 < area:
            textbox_regions.append((x, y, x + w, y + h))
    print("x, y, x + w, y + h:", x, y, x + w, y + h)

    # Procesar el contenido de cada región
    textbox_contents = {}

    for region in textbox_regions:
        x1, y1, x2, y2 = region

        roi = gray[y1:y2, x1:x2]
        content = pytesseract.image_to_string(roi, lang="spa")
        textbox_contents[(x1, y1, x2, y2)] = content

    # Dibujar las coordenadas en la imagen
    for coords in textbox_contents.keys():
        x1, y1, x2, y2 = coords
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        print("x1, y1, x2, y2", x1, y1, x2, y2)

    # # Procesar la imagen y obtener las coordenadas de los textbox
    # textbox_regions = []
    # contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for contour in contours:
    #     area = cv2.contourArea(contour)
    #     if 500 < area:
    #         x, y, w, h = cv2.boundingRect(contour)
    #         textbox_regions.append((x, y, x + w, y + h))
    # print("x, y, x + w, y + h:", x, y, x + w, y + h)

    # Crear una copia de la imagen para escribir en ella
    img_con_texto = img.copy()

    # Recorrer las claves del diccionario vacío
    for clave_vacio in dic_vacio:
        # Buscar la clave en el contenido de la imagen
        if clave_vacio in img_content:
            # Si la clave está en el contenido, obtener el valor correspondiente del diccionario lleno
            valor_lleno = dic_lleno.get(clave_vacio, '')

            # Encontrar la posición de la clave en la imagen
            posicion = img_content.find(clave_vacio)
            print("posicion:",x1,y1)
            # Agregar el valor en la imagen utilizando putText
            for coords in textbox_contents.keys():
                x1, y1, x2, y2 = coords
                # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img_con_texto,
                    valor_lleno,
                    (x1+5, y1+20),  # Ajusta la posición según tus necesidades
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )


    return img_con_texto


# Cargar la imagen en la que quieres llenar los campos
img_a_llenar = cv2.imread("imagen_sinonimo.png")

# Llamar a la función para llenar y escribir en los campos de la imagen
img_con_campos_llenos = llenar_campos_en_imagen(img_a_llenar, diccionario, dic_vacio)

# Mostrar la imagen con los campos llenados y escritos
cv2.imshow("Imagen con campos llenados", img_con_campos_llenos)
cv2.waitKey(0)




