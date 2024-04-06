import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from edu.javerianacali.create_dataset import CreateDataSet
from tabulate import tabulate


class LogisticRegressionModel:
    def __init__(self):
        pass

    def prepare_dataset(self, directorio):
        data, labels, dimensiones = CreateDataSet().create_dataset(directorio=directorio)

        # Aplanar cada elemento en data para tener una longitud consistente
        flattened_data = [d.flatten() for d in data]

        # Convertir la lista aplanada en un array de NumPy
        X = np.array(flattened_data)

        dims = np.array(dimensiones)
        df_data = pd.DataFrame(X)

        # Agregar columnas para dimensiones
        df_data['height'] = dims[:, 0]
        df_data['width'] = dims[:, 1]

        # Guardar el DataFrame como CSV (opcional)
        df_data.to_csv(os.path.join(directorio, 'dataset_imagenes.csv'), index=False)


    def train_model(self,data, labels):
        # Suponiendo que 'data' es tu matriz de imágenes aplanadas
        # y 'labels' es un array con las etiquetas correspondientes (0 o 1, por ejemplo)

        X = data  # Tus datos (pixeles de las imágenes)
        y = labels  # Etiquetas correspondientes

        # Dividir el conjunto de datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Crear el modelo de regresión logística
        clf = LogisticRegression(max_iter=10000)  # max_iter es para asegurarnos de que converge

        # Entrenar el modelo
        clf.fit(X_train, y_train)

        # Predecir etiquetas para el conjunto de prueba
        y_pred = clf.predict(X_test)

        # Evaluar el rendimiento del modelo
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")
        
        # Genera el informe de clasificación y la matriz de confusión
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv('classification_report.csv', index=True)

        # Convierte el informe de clasificación en una estructura que tabulate pueda manejar
        report_table = []

        # Agrega los datos al report_table
        for label, metrics in report.items():
            if isinstance(metrics, dict):  # Asegura que metrics sea un diccionario
                row = [label] + [metrics.get(metric, '') for metric in ['precision', 'recall', 'f1-score', 'support']]
                report_table.append(row)

        # Agrega el promedio/total al final
        average_row = ['average/total'] + [report.get(metric, '') for metric in ['precision', 'recall', 'f1-score', 'support']]
        report_table.append(average_row)

        # Usa tabulate para formatear el informe de clasificación
        print("Classification Report:")
        print(tabulate(report_table, headers=['label', 'precision', 'recall', 'f1-score', 'support'], tablefmt='fancy_grid', numalign='right'))

        # Para la matriz de confusión
        matrix = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(tabulate(matrix, tablefmt='fancy_grid', numalign='right'))
