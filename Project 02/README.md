# Project at a glance

The goal of this project is to deepen what was developed in project 01 [("First Machine Learning model")](https://github.com/gpozzi/machine-learning/tree/master/Project%2001) by applying the incorporated techniques (Data Transformation, Hyperparameter Optimization, Advanced Models, etc.) to generate a model that has a better performance than the model generated in the previous project.

- Project tools:
  - `Jupyter Notebook`
  - `Numpy`
  - `Pandas`
  - `Seaborn`
  - `Matplotlib`
  - `Scipy`
  - `Scikit Learn`

This project is divided into five parts:

- Introduction
- Data transformation
  - Imputation
  - Encoding (`one hot` and `label encoding`)
  - Outliers removal
  - Data scaling (`z-transformation` and `log-transformation`)
- Machine Learning
  - Stating results of previous project
  - Model training and comparing with benchmark of the following models:
   - Decision tree optimized with RandomSearchCV
   - XGBRegressor
   - XGBRegressor optimized with RandomSearchCV
   - RandomForest regressor
   - RandomForest optimized with RandomSearchCV
   - ADABoost optimized with RandomSearchCV
   - Polynomial regression
  - Optimization of the best performing one
- Results interpretation
- Conclusions

# Motivation
House is one of human life's most essential needs, along with other fundamental needs such as food, water, and much more. Demand for houses grew rapidly over the years as  people's living standards improved. While there are people who make their house as an investment and property, yet most people around the world are buying a house as their shelter or as their livelihood.

An increase in house demand occurs each year, indirectly causing house price increases every year. The problem arises when there are numerous variables that may influence the house price, thus most stakeholders including buyers and developers, house builders and the real estate industry would like to know the exact attributes or the accurate factors  influencing the house price to help investors make decisions and help house builders set the house price.

There  are many  benefits that  home buyers,  property investors, and house builders can reap from the house-price model. A good model would provide a lot of information and knowledge to home buyers, property investors and house builders, such as the valuation of house prices in the present market, which will help them determine house prices.  Meanwhile, this model can help potential buyers decide the characteristics of a house they want according to their budget

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

# Some exploratory data analysis

![img](https://i.imgur.com/TQUSCsM.png)

![img](https://i.imgur.com/6EJ0IsP.png)

![img](https://i.imgur.com/ZncBweE.png)

![img](https://i.imgur.com/No65L68.png)

![img](https://i.imgur.com/oYR0MF9.png)

![img](https://i.imgur.com/JW7TDxE.png)

# Conclusions

Although the analysis carried out so far makes it possible to select the best regression model by comparing relative errors, it should be noted that US $ 138,384, which is the value of the RMSE, is also the value of some departments, so an error of this magnitude should be considered inadmissible.

This model presents some opportunities for improvement, such as:

- Better handling of missing values, such as imputation, could be done instead of removing these instances from the dataset
- Categorical variables such as the neighborhood or perhaps the presence of some keywords within the description could be introduced to the analysis
- Data external to the dataset could be used such as the location of some points of interest (such as subway stations, hospitals, schools or bus stops) relative to the properties using the coordinates
- As a future improvement of the dataset and taking into account that the pricing of a property (and more so in a real estate market such as Argentina) can present large variations in a short period of time, the variable "realization of the sale operation" could be added. This categorical variable would be easy to collect, and at the same time penalize excessively high prices, resulting in an improvement in the predictive capacity of the model.
