import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from edu.javerianacali.show_images import ShowImages


class CreateDataSet:
    def __init__(self):
        self.directorio = ""


    def create_dataset(self,directorio):
        archivos = os.listdir(directorio+'/process')
        print(archivos)
        data = []
        labels = []
        dimensiones = []
        for archivo in archivos:
            if archivo.endswith(".jpg"):
                ruta_imagen = os.path.join(directorio+'/process/', archivo)
                print(ruta_imagen)
                img = cv2.imread(ruta_imagen)
                #mostrar_histograma(img)
                if img is not None:
                    print(directorio+'/process'+archivo)
                    flattened = img.flatten()
                    data.append(flattened)
                  #  ShowImages().mostrar_densidad(flattened,archivo)
                    dimensiones.append(img.shape)
                    # Aquí necesitas una forma de obtener la etiqueta para cada imagen
                    # Ejemplo: si el nombre del archivo indica la clase
                    if "sano" in archivo:
                        labels.append(0)
                    else:
                        labels.append(1)
        return data , labels, dimensiones