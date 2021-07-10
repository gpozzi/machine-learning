# Project at a glance: Real estate market analysis 🏡

I have recently joined the Data team of a large real estate company. The first task assigned is to help appraisers value properties, as it's a difficult and sometimes subjective process. To do this, I should develop a Machine Learning model that, given certain characteristics of the property, predicts its sale price.

<img src="https://image.freepik.com/vector-gratis/ilustracion-vector-concepto-abstracto-bienes-raices-agencia-inmobiliaria-mercado-inmobiliario-residencial-industrial-comercial-cartera-inversiones-propiedad-vivienda-metafora-abstracta-valor-propiedad_335657-1967.jpg" width=300>

- Project tools / libraries:
  - `Jupyter Notebook`
  - `Numpy`
  - `Pandas`
  - `Seaborn`
  - `Matplotlib`
  - `Scikit Learn`
  - `Scipy`

- Techniques applied:
  - Exploratory data analysis
  - Data transformation
  - Data cleaning
  - Machine learning technique applied: **regression**

# Motivation
Housing is one of human life's most essential needs, along with other fundamental needs such as food, water, and much more. Demand for houses grew rapidly over the years as  people's living standards improved. While there are people who make their house as an investment and property, yet most people around the world are buying a house as their shelter or as their livelihood.

An increase in house demand occurs each year, indirectly causing house price increases every year. The problem arises when there are numerous variables that may influence the house price, thus most stakeholders including buyers and developers, house builders and the real estate industry would like to know the exact attributes or the accurate factors  influencing the house price to help investors make decisions and help house builders set the house price.

There  are many  benefits that  home buyers,  property investors, and house builders can reap from the house-price model. A good model would provide a lot of information and knowledge to home buyers, property investors and house builders, such as the valuation of house prices in the present market, which will help them determine house prices.  Meanwhile, this model can help potential buyers decide the characteristics of a house they want according to their budget

# Data description
Dataset has been provided by [Properati](https://www.properati.com.ar/data), it contains 1 file: `DS_Proyecto_01_Datos_Properati.csv`
The parameters included are:

- `start_date`: Date of registration of the property publication. (numerical)
- `end_date`: Date in which the publication has been withdrawn. (numerical)
- `created_on`: Date in which the publication has been created. (numerical)
- `lat`: Latitude of the property. (categorical)
- `lon`: Longitude of the property. (categorical)
- `l1`: First administrative level: country. (categorical)
- `l2`: Second administrative level: province. (categorical)
- `l3`: Third administrative level: city. (categorical)
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

<details><summary>Dataset license</summary>

# License

All resources are under the Creative Commons CC BY 3.0 license, meaning that Properati invites everyone to use the data and distribute their products by any medium and format. They only ask in return to mention the source, indicating the changes made and to add a link to their site:

Data provided by <a href='https://www.properati.com.ar'>Properati</a>
</details>

# Project description
This project was made for [Acámica's](https://www.acamica.com/data-science) data science bootcamp and consists of two iterations (click collapsible sections to see details of each):
<details>
<summary>I</summary>

## Iteration one

The first model starts with a comprehensive `EDA` and applying some simple `data cleaning` techniques to end up building a `machine learning regression model` that performs better than a linear regression. Two models have been tried in this project: **decisionTreeRegressor** and **kNeighborsRegressor**

### Index
- Scope
- Factors that influence house price
- EDA and data cleaning
  - Feature selection
  - Additional analysis: correlation between population density and apartment size
- Machine Learning
  - Selection of measurement of error
  - Setting predictor and target variables
  - Setting benchmark model
  - Comparing benchmark with other models
  - Optimization of the best performing one
- Conclusions

### Some visuals

![img](https://i.imgur.com/TQUSCsM.png)

![img](https://i.imgur.com/6EJ0IsP.png)

![img](https://i.imgur.com/ZncBweE.png)

![img](https://i.imgur.com/No65L68.png)

![img](https://i.imgur.com/oYR0MF9.png)

![img](https://i.imgur.com/JW7TDxE.png)

### Conclusions

The value of the RMSE obtained (**US$ 138,384**) is also what some apartments cost, so a model with an error of this magnitude is not recommended to use in real world predictions. It presents some opportunities for improvement, which are best described in the [project's notebook](https://github.com/gpozzi/machine-learning/blob/master/projects/house_price_prediction/DSProject01.ipynb).
</details>

<details>
<summary>II</summary>

## Iteration two
In this version, a more thorough preprocessing has been made, performing `data transformation` (imputation, encoding, outliers removal and data scaling) techniques and finally applying more advanced `machine learning regression models` (**XGBRegressor**, **Decision tree optimized with RandomSearchCV**, **XGBRegressor optimized with RandomSearchCV**, **RandomForest regressor**, **RandomForest optimized with RandomSearchCV**, **ADABoost optimized with RandomSearchCV** and **Polynomial regressor**).

### Index
- Introduction
- Data transformation
  - Imputation
  - Encoding (`one hot` and `label encoding`)
  - Outliers removal
  - Data scaling (`z-transformation` and `log-transformation`)
- Machine Learning
  - Stating results of previous project
  - Model training and comparing benchmark with the following models:
    - Decision tree optimized with RandomSearchCV
    - XGBRegressor
    - XGBRegressor optimized with RandomSearchCV
    - RandomForest regressor
    - RandomForest optimized with RandomSearchCV
    - ADABoost optimized with RandomSearchCV
    - Polynomial regressor
  - Optimization of the best performing one
- Results interpretation
- Conclusions

### Some visuals

![img](https://i.imgur.com/zkMXTXv.png)

![img](https://i.imgur.com/ECrUenO.png)

![img](https://i.imgur.com/ig7yjCW.png)

### Conclusions
Proposed improvements to the first iteration significantly improved the model's performance. However, as in the previous report, the model's error (**USD 59,113**) is still significant and unacceptable considering that it is half the value of 27% of the apartments for sale. There is more room for model improvement, which is also described in the [project's notebook](https://github.com/gpozzi/machine-learning/blob/master/projects/house_price_prediction/DSProyecto02.ipynb)
</details>

# Requirements
All the requirements will be given in the requirements.txt file. To install, run pip install -r requirements.txt
