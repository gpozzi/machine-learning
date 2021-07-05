**Project at a glance**

- Libraries:
  - Numpy
  - Pandas
  - Seaborn
  - Matplotlib
  - Scikit Learn
- Dataset
- 
The goal of this project is to develop a machine learning algorithm able to predict the prices of potential new properties from the given attributes.
This project is divided into three parts:

- Introduction
- EDA
- Machine Learning
  - Selection of measure
- Conclusions

# Data description
Dataset has been provided by [Properati](https://www.properati.com.ar/data), it contains 1 file: `DS_Proyecto_01_Datos_Properati.csv`
The parameters included are:

- `start_date`: Date of registration of the property publication. (numerical)
- `end_date`: Date in which the publication has been withdrawed. (numerical)
- `created_on`: Date in which the publication has been created. (numerical)
- `lat`: Latitude of the property. (categorical)
- `lon`: Longitude of the property. (categorical)
- `l1`: First administrative level: country. (categorical)
- `l2`: First administrative level: province. (categorical)
- `l3`: First administrative level: city. (categorical)
- `rooms`: Amount of rooms. (numerical)
- `bedrooms`: Amount of bedrooms. (numerical)
- `bathrooms`: Amount of bathrooms. (numerical)
- `surface_total`: Total surface of the property (squared meters). (numerical)
- `surface_covered`: Total covered surface of the property (squared meters). (numerical)
- `price`: Price of the property. (numerical)
- `currency`: Currency in which the price is published. (categorical)
- `title`: Publication title. (categorical)
- `description`: Description of the property. (categorical)
- `property_type`: Type of property (house, apartment, PH). (categorical)
- `operation_type`: Type of operation (buy, rent). (categorical)

# Technical overview
The project has been divided into various steps, which include:
- Data exploration and cleaning
- Supervised learning
- Model evaluation
- Predictions on test data

# Requirements
All the requirements will be given in the requirements.txt file. To install, run pip install -r requirements.txt

# Results
The results have been documented in the [Jupyter Notebook](https://github.com/gpozzi/acamica-DS/blob/master/Project%2001/DSProyecto01.ipynb)
