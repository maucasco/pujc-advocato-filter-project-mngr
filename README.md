# Proyecto de Clasificación de Imágenes de Aguacates

Este proyecto tiene como objetivo la clasificación de imágenes de aguacates para detectar enfermedades. Utiliza redes neuronales convolucionales (CNN) y máquinas de vectores de soporte (SVM) para realizar las predicciones.

## Estructura del Proyecto

- `assets_inicio/`: Directorio que contiene los activos iniciales del proyecto.
- `edu/`: Directorio principal del proyecto con los módulos de Python.
  - `create_dataset.py`: Script para crear el conjunto de datos.
  - `data_augmentation.py`: Script para aumentar el conjunto de datos mediante técnicas de aumento de datos.
  - `extract_properties.py`: Script para extraer propiedades de las imágenes.
  - `process_images.py`: Script para procesar las imágenes.
  - `show_images.py`: Script para visualizar las imágenes.
- `classification_report_cnn.csv`: Reporte de clasificación para la CNN.
- `classification_report.csv`: Reporte de clasificación.
- `cnn_final.ipynb`: Notebook con la implementación final de la CNN.
- `cnn_imagedatagenerator.ipynb`: Notebook que utiliza ImageDataGenerator para la CNN.
- `cnn.ipynb`: Notebook principal para la CNN.
- `install.sh`: Script para instalar dependencias.
- `installDependencies.sh`: Script para instalar dependencias específicas.
- `Layer(type);OutputShape,Param#.py`: Script con la definición de capas.
- `modelo_cnn.h5`: Modelo entrenado de la CNN.
- `process_images.py`: Script para procesar imágenes.
- `requirements.txt`: Archivo con las dependencias del proyecto.
- `run copy 2.ipynb`: Copia de un notebook de ejecución.
- `run copy 3.ipynb`: Otra copia de un notebook de ejecución.
- `svm_final.ipynb`: Notebook con la implementación final del SVM.
- `svm_pipeline_model.h5`: Modelo entrenado de la SVM.
- `svm.ipynb`: Notebook principal para el SVM.
- `svmsinbal.ipynb`: Notebook del SVM sin balanceo.

## Instalación de Dependencias

Para instalar todas las dependencias necesarias para ejecutar los notebooks, puedes usar el siguiente comando:

```bash
sh install.sh


Para crear un README que explique cómo ejecutar los notebooks de tu proyecto, puedes seguir esta estructura general. Aquí tienes un ejemplo que puedes adaptar según tus necesidades:

markdown
Copiar código
# Proyecto de Clasificación de Imágenes de Aguacates

Este proyecto tiene como objetivo la clasificación de imágenes de aguacates para detectar enfermedades. Utiliza redes neuronales convolucionales (CNN) y máquinas de vectores de soporte (SVM) para realizar las predicciones.

## Estructura del Proyecto

- `assets_inicio/`: Directorio que contiene los activos iniciales del proyecto.
- `edu/`: Directorio principal del proyecto con los módulos de Python.
  - `create_dataset.py`: Script para crear el conjunto de datos.
  - `data_augmentation.py`: Script para aumentar el conjunto de datos mediante técnicas de aumento de datos.
  - `extract_properties.py`: Script para extraer propiedades de las imágenes.
  - `process_images.py`: Script para procesar las imágenes.
  - `show_images.py`: Script para visualizar las imágenes.
- `classification_report_cnn.csv`: Reporte de clasificación para la CNN.
- `classification_report.csv`: Reporte de clasificación.
- `cnn_final.ipynb`: Notebook con la implementación final de la CNN.
- `cnn_imagedatagenerator.ipynb`: Notebook que utiliza ImageDataGenerator para la CNN.
- `cnn.ipynb`: Notebook principal para la CNN.
- `install.sh`: Script para instalar dependencias.
- `installDependencies.sh`: Script para instalar dependencias específicas.
- `Layer(type);OutputShape,Param#.py`: Script con la definición de capas.
- `modelo_cnn.h5`: Modelo entrenado de la CNN.
- `process_images.py`: Script para procesar imágenes.
- `requirements.txt`: Archivo con las dependencias del proyecto.
- `run copy 2.ipynb`: Copia de un notebook de ejecución.
- `run copy 3.ipynb`: Otra copia de un notebook de ejecución.
- `svm_final.ipynb`: Notebook con la implementación final del SVM.
- `svm_pipeline_model.h5`: Modelo entrenado de la SVM.
- `svm.ipynb`: Notebook principal para el SVM.
- `svmsinbal.ipynb`: Notebook del SVM sin balanceo.

## Instalación de Dependencias

Para instalar todas las dependencias necesarias para ejecutar los notebooks, puedes usar el siguiente comando:

```bash
sh install.sh
O bien, puedes instalar dependencias específicas con:

bash
Copiar código
sh installDependencies.sh
Alternativamente, puedes instalar las dependencias listadas en requirements.txt utilizando pip:

bash
Copiar código
pip install -r requirements.txt
```

## Ejecución de Notebooks de Dependencias
CNN
cnn_final.ipynb:

Este notebook contiene la implementación final de la CNN.
Para ejecutarlo, simplemente abre el notebook en Jupyter y ejecuta todas las celdas.
cnn_imagedatagenerator.ipynb:

Este notebook utiliza ImageDataGenerator para cargar y aumentar los datos de imagen.
Ábrelo en Jupyter y ejecuta todas las celdas para entrenar la CNN con el generador de datos.
cnn.ipynb:

Notebook principal de la CNN.
Ábrelo en Jupyter y ejecuta todas las celdas para realizar la clasificación de imágenes.
SVM
svm_final.ipynb:

Este notebook contiene la implementación final del SVM.
Ábrelo en Jupyter y ejecuta todas las celdas para entrenar y evaluar el modelo SVM.
svm.ipynb:

Notebook principal para la SVM.
Ábrelo en Jupyter y ejecuta todas las celdas para realizar la clasificación de imágenes utilizando SVM.
svmsinbal.ipynb:

Este notebook muestra el SVM sin balanceo de clases.
Ábrelo en Jupyter y ejecuta todas las celdas para ver los resultados sin balanceo.

