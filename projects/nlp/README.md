# Project at a glance: Natural Language Processing 💬

The goal of this project is to deepen what was developed in project 01 [("First Machine Learning model")](https://github.com/gpozzi/machine-learning/tree/master/Project%2001) by applying the incorporated techniques (Data Transformation, Hyperparameter Optimization, Advanced Models, etc.) to generate a model that has a better performance than the model generated in the previous project.

<img src="https://image.freepik.com/vector-gratis/ilustracion-concepto-abstracto-inteligencia-artificial-chatbot_335657-3723.jpg" width=100>

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
As online marketplaces have been popular during the past decades, the online sellers and merchants ask their purchasers to share their opinions about the products they have bought. Everyday millions of reviews are generated all over the Internet about different products, services and places. This has made the Internet the most important source of getting ideas and opinions about a product or a service.

However, as the number of reviews available for a product grows, it is becoming more difficult for potential consumers to make a good decision on whether to buy the product. Different opinions about the same product on one hand and ambiguous reviews on the other hand makes customers more confused to get the right decision. Here the need for analyzing this contents seems crucial for all e-commerce businesses.

Sentiment classification is a computational study which attempts to address this problem by extracting subjective information from the given texts in natural language, such as opinions and sentiments. Different approaches have been used to tackle this problem from natural language processing, text analysis, computational linguistics, and biometrics. In recent years, Machine learning methods have got popular in the semantic and review analysis for their simplicity and accuracy.

Amazon is one of the e-commerce giants that people are using every day for online purchases where they can read thousands of reviews written by other customers about their desired products. These reviews provide valuable opinions about a product such as its properties, qualities and recommendations which helps the purchasers to understand almost every detail of a product. This is not only beneficial for consumers but also helps sellers who are manufacturing their own products to understand the consumers and their needs better.

This project is considering the sentiment classification problem for online reviews using a supervised machine learning model to determine the overall semantic of customer reviews by classifying them both by a five-star rating and also by sentiment (positive / negative). The data used in this study has been provided by Amazon.

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

