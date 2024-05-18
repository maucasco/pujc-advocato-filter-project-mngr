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
            if archivo.endswith((".jpg", ".JPG", ".jpeg", ".png")):
                ruta_imagen = os.path.join(directorio+'/process/', archivo)
                print(ruta_imagen)
                img = cv2.imread(ruta_imagen)
                #mostrar_histograma(img)
                if img is not None:

                    size_desired = (100, 96) 
                    img = cv2.resize(img, size_desired)  # Redimensionar imagen

                    print(directorio+'/process'+archivo)
                    flattened = img.flatten()
                    data.append(flattened)
                  #  ShowImages().mostrar_densidad(flattened,archivo)
                    dimensiones.append(img.shape)
                    # Aquí necesitas una forma de obtener la etiqueta para cada imagen
                    # Ejemplo: si el nombre del archivo indica la clase
                    if "sano" in archivo or "Sano" in archivo:
                        labels.append(0)
                    else:
                        labels.append(1)
        return data , labels, dimensiones
    
    def create_dataset_cnn(self, directorio):
        
      

        archivos = os.listdir(directorio+'/process')
        print(archivos)
        self.delete_images(directorio+'/train/Sano')
        self.delete_images(directorio+'/train/Heilipus')
        self.delete_images(directorio+'/test/Sano')
        self.delete_images(directorio+'/test/Heilipus')

        self.delete_images(directorio+'/test')
        imagenes_entrenamiento, imagenes_test = self.split_images(archivos)

        self.copy_images(imagenes_entrenamiento, directorio, '/train')
        self.copy_images(imagenes_test, directorio, '/test')
    def delete_images(self, directorio):
        archivos = os.listdir(directorio)

        # Limpiar directorio de entrenamiento
        for archivo in archivos:
           if archivo.endswith((".jpg", ".JPG", ".jpeg", ".png")):
               ruta_imagen = os.path.join(directorio, archivo)
               os.remove(ruta_imagen)


    def split_images(self,archivos):
        total_imagenes = len(archivos)
        num_entrenamiento = int(total_imagenes * 0.7)
        imagenes_entrenamiento = archivos[:num_entrenamiento]
        imagenes_test = archivos[num_entrenamiento:]
        return imagenes_entrenamiento, imagenes_test

    def copy_images(self,imagenes, directorio, destino):
        for archivo in imagenes:
            #simplify this condition
            if archivo.endswith((".jpg", ".JPG", ".jpeg", ".png")):
                ruta_imagen = os.path.join(directorio+'/process/', archivo)
                print(ruta_imagen)
                img = cv2.imread(ruta_imagen)
                if "sano" in archivo or "Sano" in archivo:
                    if "aug" not in archivo:
                        nueva_imga = directorio + destino + '/Sano/' + archivo
                        print(nueva_imga)
                        cv2.imwrite(nueva_imga, img)
                if "Heilipus" in archivo or "heilipus" in archivo:
                    nueva_imga = directorio + destino + '/Heilipus/' + archivo
                    print(nueva_imga)
                    cv2.imwrite(nueva_imga, img)
           