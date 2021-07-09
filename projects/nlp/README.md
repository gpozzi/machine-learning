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
    - Polynomial regressor
  - Optimization of the best performing one
- Results interpretation
- Conclusions

# Motivation
Housing is one of human life's most essential needs, along with other fundamental needs such as food, water, and much more. Demand for houses grew rapidly over the years as  people's living standards improved. While there are people who make their house as an investment and property, yet most people around the world are buying a house as their shelter or as their livelihood.

An increase in house demand occurs each year, indirectly causing house price increases every year. The problem arises when there are numerous variables that may influence the house price, thus most stakeholders including buyers and developers, house builders and the real estate industry would like to know the exact attributes or the accurate factors  influencing the house price to help investors make decisions and help house builders set the house price.

There  are many  benefits that  home buyers,  property investors, and house builders can reap from the house-price model. A good model would provide a lot of information and knowledge to home buyers, property investors and house builders, such as the valuation of house prices in the present market, which will help them determine house prices.  Meanwhile, this model can help potential buyers decide the characteristics of a house they want according to their budget

# Data description
Dataset has been provided by [Amazon](https://registry.opendata.aws/amazon-reviews-ml/), it contains 3 files: `dataset_es_dev.json`,`dataset_es_test.json` and `dataset_es_train.json`.

The dataset contains reviews in Spanish collected between November 1, 2015 and November 1, 2019. Each record contains the review text, the review title, the star rating, an anonymized reviewer ID, an anonymized product ID and the coarse-grained product category (e.g. ‘books’, ‘appliances’, etc.) The corpus is balanced across stars, so each star rating constitutes 20% of the reviews in each language.

There are **200,000**, **5,000** and **5,000** reviews in the training, development and test sets respectively. The maximum number of reviews per reviewer is 20 and the maximum number of reviews per product is 20. All reviews are truncated after 2,000 characters, and all reviews are at least 20 characters long.

The parameters included are:

- `review_id`:  Identifier of the review. (String - categorical)
- `product_id`: Identifier of the product being reviewed. (String - categorical)
- `created_on`: Identifier of the date review was written in. (Timestamp - numerical)
- `reviewer_id`: Identifier of the reviewer. (String - categorical)
- `stars`: Number of stars given in current review. (Int - numerical)
- `review_body`: Text body of the review. (String - categorical)
- `review_title`: Text title of the review. (String - categorical)
- `language`: Identifier of the review language review was written in. (String - categorical)
- `product_category`: Identifier of the product category. (String - categorical)

# Requirements
All the requirements will be given in the requirements.txt file. To install, run pip install -r requirements.txt

# Some exploratory data analysis

![img](https://i.imgur.com/zkMXTXv.png)

![img](https://i.imgur.com/ECrUenO.png)

![img](https://i.imgur.com/ig7yjCW.png)

# Conclusions
As expected, the proposed improvements, both preprocessing and the use of better predictive models in this report, significantly improved performance compared to the first project. Adding variables such as neighborhood, reducing the noise from outliers, scaling variables or not discarding so many instances just for having missing values greatly enriched the model.

However, as in the previous report, the mean square error of the best model (**USD 59,113**) is still significant and unacceptable considering that it is **half the value of 27% of the apartments for sale**.

I reiterate some of the recommendations from the previous report to improve the performance of this model:

- Data external to the dataset could be used such as the location of some points of interest (such as subway stations, hospitals, schools or bus stops) relative to the properties using the coordinates
- As a future improvement of the dataset and taking into account that the pricing of a property (and more so in a real estate market such as Argentina) can present large variations in a short period of time, the variable "realization of the sale operation could be added ". This categorical variable would be easy to collect, and at the same time penalize excessively high prices, resulting in an improvement in the predictive capacity of the model.
