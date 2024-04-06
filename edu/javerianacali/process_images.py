
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


class ProcessImages:
    def __init__(self):
        self.directorio = ""

    def eliminar_sombras(self,image):

    # Convertir la imagen a HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Extraer el canal V
        v = hsv[:,:,2]

        # Aplicar la corrección gamma al canal V
        gamma_corrected = np.array(255*(v / 255) ** 0.5 , dtype='uint8')

        # Reemplazar el canal V en la imagen HSV con el canal V corregido
        hsv[:,:,2] = gamma_corrected

        # Convertir la imagen HSV de vuelta a BGR
        shadow_removed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return shadow_removed


    def eliminar_texto(self,imagen, x_inicio, y_inicio, x_fin, y_fin):
        mascara = np.zeros_like(imagen[:, :, 0])
        mascara[y_inicio:y_fin, x_inicio:x_fin] = 255
        dilatada = cv2.dilate(mascara, (1,1), iterations = 2)  # Dilatar para cubrir más región del texto
        return cv2.inpaint(imagen, dilatada, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    def mejorar_contraste(self,imagen):
        # Ecualización del histograma
        ecualizada = cv2.equalizeHist(imagen)
        
        # Ajuste de contraste y brillo
        contraste = 1.5  # Factor de contraste (mayor que 1 aumenta el contraste)
        brillo = 30      # Ajuste de brillo
        mejorada = cv2.convertScaleAbs(ecualizada, alpha=contraste, beta=brillo)
        
        # Filtro bilateral para suavizar el ruido manteniendo los bordes
        filtrada = cv2.bilateralFilter(mejorada, 9, 75, 75)
        
        return filtrada

    def segmentar_imagen(self,imagen_sin_texto):
        
    # Eliminar sombras
        imagen_sin_texto= self.eliminar_sombras(imagen_sin_texto)
        gray = cv2.cvtColor(imagen_sin_texto, cv2.COLOR_BGR2GRAY)

    # Aplicar la corrección gamma para eliminar las sombras
        gamma_corrected = np.array(255*(gray / 255) ** 0.2 , dtype='uint8')     
        # Usar thresholding de Otsu
        _, imagen_segmentada = cv2.threshold(gamma_corrected, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Detección de contornos y conservar solo el contorno más grande
        contours, _ = cv2.findContours(imagen_segmentada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar contornos por área
        min_area = 3000  # Puedes ajustar este valor
        contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        mask = np.zeros_like(gamma_corrected)
        if contours:
            # Ordenar contornos y dibujar el más grande
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            cv2.drawContours(mask, [contours[0]], -1, (255), thickness=cv2.FILLED)
        
        aguacate = cv2.bitwise_and(gamma_corrected, gamma_corrected, mask=mask)
        
        # Mejorar el contraste y brillo del aguacate
        aguacate_resaltado = self.mejorar_contraste(aguacate)
        
        return aguacate_resaltado

    # Carga y preprocesamiento
    def redimensionar(self, image):
        original_height, original_width = image.shape[:2]
        desired_width = 100

        # Calcula el factor de escala y el alto deseado
        scale_factor = desired_width / original_width
        desired_height = int(original_height * scale_factor)

        # Redimensiona la imagen proporcionalmente
        resized_img = cv2.resize(image, (desired_width, desired_height))
        return resized_img

    def procesar(self,imagen):

        imagen_sin_texto=[]
        print("Type:",type(imagen))
        print("Shape of Image:", imagen.shape)
        print('Total Number of pixels:', imagen.size)
        print("Image data type:",imagen.dtype)
            # print("Pixel Values:\n", img)
        print("Dimension:", imagen.ndim)
      
       
           
        if imagen.size>1000000 and imagen.size<=9999999:
            imagen_sin_texto = self.eliminar_texto(imagen, 0, 0, 3000, 250)
        elif imagen.size>10000000 and imagen.size<=29999999:
            imagen_sin_texto = self.eliminar_texto(imagen, 0, 0, 3000, 1000)
        elif imagen.size>=30000000 and imagen.size<=35000000:
            imagen_sin_texto = imagen
        else:
            imagen_sin_texto = self.eliminar_texto(imagen, 0, 0, 3000, 500)    
            
        aguacate_solo = self.segmentar_imagen(imagen_sin_texto)
        redimenciada = self.redimensionar(aguacate_solo)
    
        return imagen_sin_texto,aguacate_solo, redimenciada
    

    def procesar_imagen(self,ruta_imagen,nombre,directorio):
        imagen = cv2.imread(ruta_imagen)[100:, :]

        imagen_sin_texto,aguacate_solo,redimenciada = self.procesar(imagen)
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

        
