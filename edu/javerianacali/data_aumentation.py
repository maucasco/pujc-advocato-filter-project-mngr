
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
import os
import random


class DataAugmentation:
    # Función para crear un generador de aumentación de datos con parámetros aleatorios
    def random_datagen(self):
        return ImageDataGenerator(
            rotation_range=random.randint(0, 30),  # Reducir el rango de rotación
            width_shift_range=random.uniform(0.05, 0.15),  # Reducir el rango de cambio de ancho
            height_shift_range=random.uniform(0.05, 0.15),  # Reducir el rango de cambio de altura
            shear_range=random.uniform(0.05, 0.15),  # Reducir el ángulo de cizallamiento
            zoom_range=random.uniform(0.05, 0.15),  # Reducir el rango de zoom
            horizontal_flip=random.choice([True, False]),  # Volteo horizontal aleatorio
            fill_mode='nearest'  # Relleno de pixeles faltantes
        )


    def augment_images(self,directory):
        # Inicializar el generador de aumentación 
    
        # Obtener una lista de archivos de imagen en el directorio
        image_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        save_dir = directory + '/augmented_images'
        print(save_dir)
        # Procesar cada imagen
        for image_file in image_files:
            img_path = os.path.join(directory, image_file)
            img = load_img(img_path)  # Cargar imagen
            x = img_to_array(img)     # Convertir la imagen a un array
            x = np.expand_dims(x, axis=0)  # Añadir una dimensión extra
            save_prefix = 'aug_' + image_file.split('.')[0]
            datagen = self.random_datagen()
            # Generar y guardar imágenes aumentadas
            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=save_dir, save_prefix=save_prefix, save_format='jpg'):
                i += 1
                if i >= 20:  # Generar 10 imágenes aumentadas por archivo
                    break
