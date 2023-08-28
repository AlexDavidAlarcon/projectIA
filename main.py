import cv2
import numpy as np
import pytesseract
from gensim.models import KeyedVectors
from unidecode import unidecode  # quitar tildes
import re
# Configurar Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract"

# Cargar la imagen y convertirla a escala de grises
img_lleno = cv2.imread("dataset_prueba/campos_arriba.png")
img_vacio = cv2.imread("dataset_prueba/imagen_sinonimo.png")

cv2.imshow("Formulario a procesar", img_lleno)
cv2.waitKey(0)

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
            textbox_regions.append((x, y, x + w, y + h)) #(x, y, x + width, y + height)

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
    # Procesar y almacenar el contenido a la izquierda de cada región de textbox en un diccionario
    contents_dictionary = {}

    for coords, content in textbox_contents.items():
        x1, y1, x2, y2 = coords
        # Probar la parte izquierda del textbox
        roi_left = gray[y1:y2, 0:x1]
        content_left = pytesseract.image_to_string(roi_left, lang="spa")
        content_left = unidecode(content_left).strip().lower()

        # Probar la parte superior del textbox
        # Buscar el contenido arriba de la región actual
        roi_top = gray[max(0, y1 - 20):y1, x1:x2]
        content_top = pytesseract.image_to_string(roi_top, lang="spa")
        content_top = unidecode(content_top).strip().lower()

        # Probar la parte inferior del textbox
        roi_bottom = gray[y2:min(y2 + 20, img.shape[0]), x1:x2]
        content_bottom = pytesseract.image_to_string(roi_bottom, lang="spa")
        content_bottom = unidecode(content_bottom).strip().lower()

        # Si hay contenido a la izquierda, arriba o abajo, almacenar en el diccionario
        if content_left:
            # texto_sin_tildes = unidecode(content_left)
            # texto_sin_tilde = unidecode(content)
            # Quitar espacios
            content_left = content_left.strip().lower()
            content = content.strip()
            # Quitar el punto al final del texto, si existe
            if content_left.endswith('.') or content_left.endswith(':'):
                content_left = content_left[:-1]
            elif content.endswith("."):
                content = content[:-1]
            contents_dictionary[content_left] = content


        elif content_top:
            texto_sin_tildes = unidecode(content_top)
            texto_sin_tilde = unidecode(content)
            # Quitar espacios
            content_top = texto_sin_tildes.strip().lower()
            content = texto_sin_tilde.strip()
            # Quitar el punto al final del texto, si existe
            if content_top.endswith('.') or content_top.endswith(':'):
                content_top = content_top[:-1]
            elif content.endswith("."):
                content = content[:-1]
            contents_dictionary[content_top] = content


        elif content_bottom:
            texto_sin_tildes = unidecode(content_bottom)
            texto_sin_tilde = unidecode(content)
            # Quitar espacios
            content_bottom = texto_sin_tildes.strip().lower()
            content = texto_sin_tilde.strip()
            # Quitar el punto al final del texto, si existe
            if content_bottom.endswith('.') or content_bottom.endswith(':'):
                content_bottom = content_bottom[:-1]
            elif content.endswith("."):
                content = content[:-1]
            contents_dictionary[content_bottom] = content
        # Eliminar espacios en blanco alrededor de la clave y valor

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
    return dic_vacio

diccionario = llenar_diccionario(dic_lleno,dic_vacio)
print("diccionario llenado:",diccionario)

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
    for coords, valor_lleno in zip(textbox_contents.keys(), dic_lleno.values()):
        x1, y1, x2, y2 = coords
        # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # print("x1, y1, x2, y2", x1, y1, x2, y2)

        match = re.search(r"(gmail|outlook|yahoo|hotmail)", valor_lleno)
        if match:
            start_index = match.start()
            valor_lleno = valor_lleno[:start_index-1] + "@" + valor_lleno[start_index :]

        # Agregar el valor desde dic_lleno en la imagen utilizando putText
        cv2.putText(
            img,
            valor_lleno,
            (x1 + 5, y1 + 20),  # Ajusta la posición según tus necesidades
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    img_con_texto = img.copy()
    return img_con_texto
# Llamar a la función para llenar y escribir en los campos de la imagen
img_con_campos_llenos = llenar_campos_en_imagen(img_vacio, diccionario, dic_vacio)

# Mostrar la imagen con los campos llenados y escritos
cv2.imshow("Imagen con campos llenados", img_con_campos_llenos)
cv2.waitKey(0)