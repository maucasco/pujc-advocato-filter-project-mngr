
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class ShowImages:
    def __init__(self):
        pass
    # Calcular histograma
    def mostrar_histograma(self,img):
        histogram = cv2.calcHist([img], [0], None, [256], [0,256])

        # Graficar histograma
        plt.figure()
        plt.title("Histograma de Intensidad")
        plt.xlabel("Valor de pixel")
        plt.ylabel("Número de píxeles")
        plt.plot(histogram)
        plt.xlim([0, 256])
        plt.show()

    def mostrar_densidad(self,pixels,archivo):

        # Crear el gráfico de densidad
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        sns.kdeplot(pixels, bw_adjust=0.5, shade=True)

        plt.title('Gráfico de Densidad de Valores de Píxeles'+archivo)
        plt.xlabel('Valor del Píxel')
        plt.ylabel('Densidad')
        plt.show()