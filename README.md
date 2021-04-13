<img src="https://d92mrp7hetgfk.cloudfront.net/images/sites/misc/Acamica/original.png" width="500">

# 1. Alcance

En este repositorio subiré los proyectos entregados para el curso de Data Science dictado por Acámica en 2020/2021

# 2. Temas

## Módulo 1: Introducción a Data Science <img src="https://static.thenounproject.com/png/2245695-200.png" width="25">

### Temas:
- Programación (clases y funciones)
- Python y librerías como Numpy, Pandas, Matplotlib, Seaborn
- Análisis exploratorio de datos (EDA)
- Introducción a Machine Learning: clasificación y regresión
- Tipos de gráficos
- Modelos básicos (árboles, KNN, regresión lineal, regresión logística)
- Métricas de evaluación
- MAE
- Train/test Split

### [Proyecto: Primer modelo de Machine Learning (aprobado <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/Light_green_check.svg/1024px-Light_green_check.svg.png" width="15"> )](https://github.com/gpozzi/acamica-DS/blob/master/DSProyecto01.ipynb)
En primer lugar realicé un EDA a un dataset que contiene datos de propiedades en Argentina. Luego de identificar las particularidades del mismo, entrené un modelo sencillo de Machine Learning y realicé la optimización de sus hiperparámetros para predecir los precios de potenciales nuevas propiedades a partir de los atributos dados. La precisión del modelo no fue satisfactoria debido a falta de un adecuado preprocesamiento y a la simplicidad del mismo.

## Módulo 2: Herramientas avanzadas <img src="https://static.thenounproject.com/png/788416-200.png" width="25">

### Temas:
- Machine Learning en profundidad
- Fundamentos estadísticos
- Interpretación de modelos
-Estadística: distribuciones, Teorema de Bayes
- Modelos avanzados: ensables, SVM, Naive Bayes, redes neuronales
- Optimización de parámetros
- Interpretación de modelos
- Ingeniería de features: One Hoy Encoding, LabelEncoding, Scaling, etc.

### [Proyecto: Ingeniería de Features, Modelos Avanzados e Interpretación de Modelos (aprobado <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/Light_green_check.svg/1024px-Light_green_check.svg.png" width="15"> )](https://github.com/gpozzi/acamica-DS/blob/master/DSProyecto02.ipynb)
En este proyecto apliqué ingeniería de features y modelos avanzados para desarrollar con mayor profundidad y mejorar el rendimiento del modelo de Machine Learning obtenido en el Proyecto 1. Si bien con un preprocesamiento más exhaustivo y la aplicación de modelos de regresión más avanzados el rendimiento mejoró notablemente (reduciendo el error en aproximadamente la mitad), debido a la falta de atributos recolectados que correlacionen mejor con el precio no se pudo obtener un modelo cuyo error pueda ser usado por fuera del ámbito experimental.

## Módulo 3: Aplicaciones <img src="https://static.thenounproject.com/png/2985136-200.png" width="30">
### Temas:
- Machine Learning en profundidad
- Fundamentos estadísticos
- Interpretación de modelos
-Estadística: distribuciones, Teorema de Bayes
- Modelos avanzados: ensables, SVM, Naive Bayes, redes neuronales
- Optimización de parámetros
- Interpretación de modelos
- Ingeniería de features: One Hoy Encoding, LabelEncoding, Scaling, etc.

### [Proyecto: Aplicaciones actuales (aprobado <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/Light_green_check.svg/1024px-Light_green_check.svg.png" width="15"> )](https://github.com/gpozzi/acamica-DS/blob/master/DS_Proyecto_03_NLP.ipynb)
En este proyecto, utilicé un dataset de Amazon que contenía reseñas sobre distintos productos para, mediante el procesamiento de lenguaje natural, poder predecir el puntaje que daría un usuario dada una reseña

## Módulo 4: Cierre de carrera <img src="https://cdn2.iconfinder.com/data/icons/ios-7-icons/50/finish_flag-512.png" width="30">
### [Proyecto: Informe final de carrera (aprobado <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/Light_green_check.svg/1024px-Light_green_check.svg.png" width="15"> )](https://github.com/gpozzi/acamica-DS/blob/master/DS_Proyecto_04_NLP-Informe_final.ipynb)
En este proyecto profundicé y amplié los resultados del proyecto anterior mediante el replanteamiento del problema como una clasificación binaria (positiva/negativa). Adicionalmente, clasifiqué a través de esta metodología a las reseñas de 3 estrellas para ver qué categorías de productos presentaban mayor tolerancia por parte de los usuarios.
