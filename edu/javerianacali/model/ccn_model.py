import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from edu.javerianacali.create_dataset import CreateDataSet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

from keras.models import Model
import matplotlib.pyplot as plt

class ConvulationalNeuralNetwork:
    def __init__    (self):
        pass    
    def prepare_dataset(self,directorio):



        # Crea un generador de datos de imagen con normalización
        datagen = ImageDataGenerator(rescale=1./255)
        print(directorio)
        # Carga las imágenes desde el directorio
        # Asegúrate de que dentro de este directorio, las imágenes estén organizadas en subdirectorios según su etiqueta/clase
        generator = datagen.flow_from_directory(
        directorio,
        target_size=(100, 130),  # Asegúrate de que estas dimensiones coincidan con el preprocesamiento que hiciste
        batch_size=48,
        class_mode='categorical' ) # Usa 'binary' si solo tienes dos clases

        return generator

    def train_model(self,generator, num_epocas, tamano_lote, canales=3):  
        modelo = Sequential([
            Conv2D(32, (5,5), activation='relu', input_shape=(100, 130, 3)),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(64, (5,5), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),  # Capa de abandono para regularización
            Dense(5, activation='softmax')
        ])
        modelo.compile(optimizer=Adam(learning_rate=0.001),  # Ajustar la tasa de aprendizaje
            loss='categorical_crossentropy',  
            metrics=['accuracy'])
        historial = modelo.fit(
            generator,
            epochs=10,  # Aumentar el número de épocas
            steps_per_epoch=generator.samples // 180
        )

        plt.plot(historial.history['accuracy'])
        plt.title('Precisión del modelo')
        plt.ylabel('Precisión')
        plt.xlabel('Época')
        plt.legend(['Entrenamiento'], loc='upper left')
        plt.show()

        # Gráfica de pérdida
        plt.plot(historial.history['loss'])
        plt.title('Pérdida del modelo')
        plt.ylabel('Pérdida')
        plt.xlabel('Época')
        plt.legend(['Entrenamiento'], loc='upper left')
        plt.show()


        #necesito generar un metodo que me permita predecir una imagen
    def predict(self, image):
        # Preprocess the image
        preprocessed_image = preprocess_image(image)
        
        # Make predictions using the model
        predictions = self.model.predict(preprocessed_image)
        
        # Get the predicted class label
        predicted_class = np.argmax(predictions)
        
        return predicted_class

    

    def save_model(self):

        pass

    def load_model(self):
            
            pass    
    def evaluate_model(self):   
         
        pass    
    def create_dataset(self):
        pass    
    def preprocess_data(self):  
        pass    
    def show_results(self,modelo,generator):


        # Crear un modelo que devuelva las activaciones de la primera capa convolucional
        activation_model = Model(inputs=modelo.input, outputs=modelo.layers[0].output)

        # Obtener las activaciones de la primera capa convolucional para la primera imagen en el conjunto de entrenamiento
        activations = activation_model.predict(generator[0][0][0].reshape(1, 100, 130, 3))

        # Visualizar las activaciones de los primeros 6 filtros
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
        for i, ax in enumerate(axes.flat):
            ax.imshow(activations[0, :, :, i], cmap='viridis')
            ax.axis('off')
        plt.show()
    def show_model(self):
        pass
# Path: edu/javerianacali/model/logistic_regression_model.py