
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


class ProcessImages:
    def __init__(self):
        self.directorio = ""

    def eliminar_sombras(self, image):
        if len(image.shape) == 3 and image.shape[2] == 3:  # Verificar si la imagen es en color
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:  # La imagen ya está en escala de grises
            gray = image
        dilated_img = cv2.dilate(gray, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(gray, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        return norm_img

    def mejorar_contraste(self, imagen):
        ecualizada = cv2.equalizeHist(imagen)
        contraste = 3.0 # Ajustar el valor del contraste
        brillo = 100      # Ajustar el valor del brillo
        mejorada = cv2.convertScaleAbs(ecualizada, alpha=contraste, beta=brillo)
        return mejorada

    def resaltar_contornos(self, imagen):
        bordes = cv2.Canny(imagen, 50, 150)  # Ajustar los umbrales para mejorar la visibilidad de los bordes
        return bordes

    def eliminar_texto(self, imagen, x_inicio, y_inicio, x_fin, y_fin):
        if len(imagen.shape) == 3 and imagen.shape[2] == 3:  # Verificar si la imagen es en color
            mascara = np.zeros_like(imagen[:, :, 0])
        else:  # La imagen ya está en escala de grises
            mascara = np.zeros_like(imagen)
        mascara[y_inicio:y_fin, x_inicio:x_fin] = 255
        dilatada = cv2.dilate(mascara, (1, 1), iterations=2)  # Dilatar para cubrir más región del texto
        return cv2.inpaint(imagen, dilatada, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    def segmentar_imagen(self, imagen_sin_texto):
        # Eliminar sombras
        imagen_sin_texto = self.eliminar_sombras(imagen_sin_texto)
        if len(imagen_sin_texto.shape) == 3 and imagen_sin_texto.shape[2] == 3:
            gray = cv2.cvtColor(imagen_sin_texto, cv2.COLOR_BGR2GRAY)
        else:
            gray = imagen_sin_texto

        # Aplicar la corrección gamma para eliminar las sombras
        gamma_corrected = np.array(255 * (gray / 255) ** 0.8, dtype='uint8')  # Ajustar la corrección gamma
        
        # Usar thresholding de Otsu
        _, imagen_segmentada = cv2.threshold(gamma_corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Detección de contornos y conservar solo el contorno más grande
        contours, _ = cv2.findContours(imagen_segmentada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar contornos por área
        min_area = 1000  # Ajustar el área mínima para capturar mejor el contorno
        contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        mask = np.zeros_like(gamma_corrected)
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            cv2.drawContours(mask, [contours[0]], -1, (255), thickness=cv2.FILLED)
        
        aguacate = cv2.bitwise_and(gamma_corrected, gamma_corrected, mask=mask)
        
        # Resaltar contornos
        aguacate_contornos = self.resaltar_contornos(aguacate)
        
        # Mejorar el contraste y brillo del aguacate
        aguacate_resaltado = self.mejorar_contraste(aguacate_contornos)
        
        return aguacate_resaltado

    def redimensionar(self, image, desired_width=100, desired_height=100):
        resized_img = cv2.resize(image, (desired_width, desired_height))
        return resized_img

    def procesar(self, imagen, desired_width, desired_height):
        imagen_sin_texto = imagen
        print("Type:", type(imagen))
        print("Shape of Image:", imagen.shape)
        print('Total Number of pixels:', imagen.size)
        print("Image data type:", imagen.dtype)
        print("Dimension:", imagen.ndim)
        
        if imagen.size > 1000000 and imagen.size <= 9999999:
            imagen_sin_texto = self.eliminar_texto(imagen, 0, 0, 3000, 50)
        elif imagen.size > 10000000 and imagen.size <= 29999999:
            imagen_sin_texto = self.eliminar_texto(imagen, 0, 0, 3000, 1000)
        elif imagen.size >= 30000000 and imagen.size <= 35000000:
            imagen_sin_texto = imagen
        elif imagen.size >= 35000001:
            imagen_sin_texto = self.eliminar_texto(imagen, 0, 0, 3000, 1100)
        



                  # Convertir la imagen a escala de grises
        gray = cv2.cvtColor(imagen_sin_texto, cv2.COLOR_BGR2GRAY)

        # Aplicar un filtro de mediana para reducir el ruido
        gray = cv2.medianBlur(gray, 5)

                # Umbralizar la imagen para obtener una máscara de las letras
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Aplicar inpainting para eliminar las letras
        result = cv2.inpaint(imagen, mask, 3, cv2.INPAINT_TELEA)

        # Aumentar el contraste en la imagen
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_img = clahe.apply(gray)

        # Detectar círculos utilizando la Transformada de Hough
        circles = cv2.HoughCircles(contrast_img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=5, maxRadius=30)

        # Si se detectan círculos, resaltar el círculo negro
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(result, (x, y), r, (0, 255, 0), 4)

        redimensionada = self.redimensionar(contrast_img, desired_width, desired_height)
    
        return gray, result, redimensionada


    # Carga y preprocesamiento
    def redimensionar(self, image,desired_width = 100,desired_height=100 ):
        original_height, original_width = image.shape[:2]
        

        # Calcula el factor de escala y el alto deseado
        #scale_factor = desired_width / original_width
        #desired_height = int(original_height * scale_factor)

        # Redimensiona la imagen proporcionalmente
        resized_img = cv2.resize(image, (desired_width, desired_height))
        return resized_img

    

    def procesar_imagen(self,ruta_imagen,nombre,directorio,desired_width,desired_height):
        imagen = cv2.imread(ruta_imagen)[100:, :]

        imagen_sin_texto,aguacate_solo,redimenciada = self.procesar(imagen,desired_width,desired_height)
        nueva_imga=directorio+'/process/fil_'+nombre
        print(nueva_imga)
        cv2.imwrite(nueva_imga, redimenciada)
        return imagen, imagen_sin_texto, aguacate_solo,redimenciada


    def mostrar_imagenes(self,titulos, imagenes):
        count = len(titulos)
        for i in range(count):
            plt.subplot(1, count, i + 1)
            plt.title(titulos[i])
            plt.imshow(imagenes[i], cmap='gray')
        plt.show()

        
