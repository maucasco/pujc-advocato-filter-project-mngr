from sklearn import svm
from keras.models import Model
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from edu.javerianacali.process_images import ProcessImages
from tabulate import tabulate

class SupportVectorMachineModel:


    def __init__    (self):
        pass    
    
    def train_svm_model(self, data, labels):
            # Convertir la lista aplanada en un array de NumPy
            X = np.array(data)
            y = np.array(labels)

            # Dividir el conjunto de datos en entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Crear el modelo SVM
            self.model = svm.SVC(kernel='linear', probability=True)

            # Entrenar el modelo SVM
            self.model.fit(X_train, y_train)

            # Predecir etiquetas para el conjunto de prueba
            y_pred = self.model.predict(X_test)

            # Evaluar el rendimiento del modelo
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {accuracy:.2f}")
    

            report_table = []

            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv('classification_report.csv', index=True)
            

            print(report_df)
            # Agrega los datos al report_table
            for label, metrics in report.items():
                if isinstance(metrics, dict):  # Asegura que metrics sea un diccionario
                    row = [label] + [metrics.get(metric, '') for metric in ['precision', 'recall', 'f1-score', 'support']]
                    report_table.append(row)

            # Rest of the code...

            # Use tabulate to format the classification report
            print("Classification Report:")
            print(tabulate(report_table, headers=['label', 'precision', 'recall', 'f1-score', 'support'], tablefmt='fancy_grid', numalign='right'))

            # Rest of the code...
            plt.xlabel('Predicción')
            plt.ylabel('Verdaderos')
            plt.title('Confusion Matrix')
            plt.show()
            

            return self.model
    
    def predict(self, image):
        # Preprocess the image
        preprocessed_image = ProcessImages().process(image)
        
        # Make predictions using the model
        predictions = self.model.predict(preprocessed_image)
        
        # Get the predicted class label
        predicted_class = np.argmax(predictions)
        
        return predicted_class

