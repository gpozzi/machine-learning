**Project at a glance**
I have recently joined the Data team of a large real estate company. The first task assigned is to help appraisers value properties, as it's a difficult and sometimes subjective process. To do this, I propose to create a Machine Learning model that, given certain characteristics of the property, predicts its sale price.

- Project tools:
  - `Jupyter Notebook`
  - `Numpy`
  - `Pandas`
  - `Seaborn`
  - `Matplotlib`
  - `Scikit Learn`

This project is divided into five parts:

- Introduction
- EDA and data cleaning
- Feature selection
- Machine Learning
  - Selection of measurement of error
  - Setting benchmark model
  - Comparing benchmark with decision tree and KNN models
  - Optimization of the best performing one
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

# Requirements
All the requirements will be given in the requirements.txt file. To install, run pip install -r requirements.txt

# Motivation
House is one of human life's most essential needs, along with other fundamental needs such as food, water, and much more. Demand for houses grew rapidly over the years as  people's living standards improved. While there are people who make their house as an investment and property, yet most people around the world are buying a house as their shelter or as their livelihood.

An increase in house demand occurs each year, indirectly causing house price increases every year. The problem arises when there are numerous variables that may influence the house price, thus most stakeholders including buyers and developers, house builders and the real estate industry would like to know the exact attributes or the accurate factors  influencing the house price to help investors make decisions and help house builders set the house price.

There  are many  benefits that  home buyers,  property investors, and house builders can reap from the house-price model. A good model would provide a lot of information and knowledge to home buyers, property investors and house builders, such as the valuation of house prices in the present market, which will help them determine house prices.  Meanwhile, this model can help potential buyers decide the characteristics of a house they want according to their budget

# Some exploratory data analysis

[img](https://i.imgur.com/TQUSCsM.png)

[img](https://i.imgur.com/6EJ0IsP.png)

[img](https://i.imgur.com/ZncBweE.png)

[img](https://i.imgur.com/No65L68.png)

[img](https://i.imgur.com/oYR0MF9.png)

[img](https://i.imgur.com/JW7TDxE.png)

# Results
The results have been documented in the [Jupyter Notebook](https://github.com/gpozzi/acamica-DS/blob/master/Project%2001/DSProyecto01.ipynb)
