import cv2
import numpy as np

# Cargar la imagen
ruta1 = "/home/maucasco/Documents/maestria/proyecto_grado/assets/Heilipus72.jpg"
im = cv2.imread(ruta1)

# Visualizar la imagen
cv2.imshow("Imagen", im)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Filtrar la tercera banda
banda_azul = im[:, :, 0]
banda_verde = im[:, :, 1]
banda_roja = im[:, :, 2]

filtro = banda_azul < 100

# Enmascarar valores
im[filtro, :] = [0, 0, 0]  # Establecer píxeles filtrados como negro

# Visualizar la imagen filtrada
cv2.imshow("Imagen Filtrada", im)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Repetir el proceso para otra imagen
ruta2 = "/home/maucasco/Documents/maestria/proyecto_grado/assets/sano1.jpg"
im_sano = cv2.imread(ruta2)

filtro_sano = im_sano[:, :, 2] < 100
im_sano[filtro_sano, :] = [0, 0, 0]

cv2.imshow("Imagen Sana Filtrada", im_sano)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Función para depurar imágenesy

def extrae_solo_aguacate(ruta):
    im = cv2.imread(ruta)
    filtro = im[:, :, 2] < 100
    im[filtro, :] = [0, 0, 0]
    cv2.imshow("Imagen Filtrada", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

ruta = "/home/maucasco/Documents/maestria/proyecto_grado/assets/Heilipus563.jpg"
extrae_solo_aguacate(ruta)