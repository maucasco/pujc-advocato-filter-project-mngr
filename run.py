import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from edu.javerianacali.create_dataset import CreateDataSet
from edu.javerianacali.model.ccn_model import ConvulationalNeuralNetwork
from edu.javerianacali.model.logistic_regression_model import LogisticRegressionModel
from edu.javerianacali.process_images import ProcessImages

directorio = "/home/maucasco/Documents/maestria/proyecto_grado/pujc-advocato-filter-project-mngr/assets"

archivos = os.listdir(directorio)
#for archivo in archivos:
#    if archivo.endswith(".jpg"):
 #       ruta_imagen = os.path.join(directorio, archivo)
  #      titulos = ["Ori", "SinTex","Sinsomb", archivo]
   #     imagenes = ProcessImages().procesar_imagen(ruta_imagen,archivo,directorio)
    #    #ProcessImages().mostrar_imagenes(titulos, imagenes)

#create_dataset = CreateDataSet().create_dataset(directorio)
#LogisticRegressionModel().create_dataset(directorio)
#LogisticRegressionModel().train_model(create_dataset[0], create_dataset[1])

generator=ConvulationalNeuralNetwork().prepare_dataset(directorio+'/tensor')
ConvulationalNeuralNetwork().train_model(generator, 10, 48, 3)