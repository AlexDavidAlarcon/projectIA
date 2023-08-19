import cv2
import numpy as np
import pytesseract
from gensim.models import KeyedVectors


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
    print(median_color)
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
        roi_left = gray[y1:y2, 0:x1]
        content_left = pytesseract.image_to_string(roi_left, lang="spa")
        # Eliminar espacios en blanco alrededor de la clave y valor
        content_left = content_left.strip()
        content = content.strip()

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





