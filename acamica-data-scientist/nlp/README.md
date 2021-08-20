# Project at a glance: Natural Language Processing 💬

The goal of this project is to be able to predict the score that a user would set given a review by using an Amazon dataset that contained reviews about different products.

<img src="https://image.freepik.com/vector-gratis/ilustracion-concepto-abstracto-inteligencia-artificial-chatbot_335657-3723.jpg" width=300>

- Project tools / libraries:
  - `Jupyter Notebook`
  - `Numpy`
  - `Pandas`
  - `Seaborn`
  - `Matplotlib`
  - `Scipy`
  - `Scikit Learn`
  - `random`
  - `es-core-news-sm`
  - `collections`
  - `itertools`
  - `wordcloud`

- Techniques applied:
  - Exploratory data analysis.
  - Data transformation: stemming and lemmatization.
  - Data cleaning.
  - Machine learning method applied: **classification**.
  - **Supervised learning**

# Motivation
As online marketplaces have been popular during the past decades, the online sellers and merchants ask their purchasers to share their opinions about the products they have bought. Everyday millions of reviews are generated all over the Internet about different products, services and places. This has made the Internet the most important source of getting ideas and opinions about a product or a service.

However, as the number of reviews available for a product grows, it is becoming more difficult for potential consumers to make a good decision on whether to buy the product. Different opinions about the same product on one hand and ambiguous reviews on the other hand makes customers more confused to get the right decision. Here the need for analyzing this contents seems crucial for all e-commerce businesses.

Sentiment classification is a computational study which attempts to address this problem by extracting subjective information from the given texts in natural language, such as opinions and sentiments. Different approaches have been used to tackle this problem from natural language processing, text analysis, computational linguistics, and biometrics. In recent years, Machine learning methods have got popular in the semantic and review analysis for their simplicity and accuracy.

Amazon is one of the e-commerce giants that people are using every day for online purchases where they can read thousands of reviews written by other customers about their desired products. These reviews provide valuable opinions about a product such as its properties, qualities and recommendations which helps the purchasers to understand almost every detail of a product. This is not only beneficial for consumers but also helps sellers who are manufacturing their own products to understand the consumers and their needs better.

This project is considering the sentiment classification problem for online reviews using a supervised machine learning model to determine the overall semantic of customer reviews by classifying them both by a five-star rating and also by sentiment (positive / negative). The data used in this study has been provided by Amazon.

# Data description

<img src="https://image.freepik.com/vector-gratis/investigacion-datos-estadisticos-indicadores-desempeno-empresa-retorno-inversion-razon-porcentual-fluctuacion-indices-cambio-significativo_335657-2552.jpg" width=200>

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

<details> <summary markdown="span">Dataset license</summary>

# LICENSE

By accessing the Multilingual Amazon Reviews Corpus ("Reviews Corpus"), you agree that the Reviews Corpus is an Amazon Service subject to the Amazon.com Conditions of Use (https://www.amazon.com/gp/help/customer/display.html/ref=footer_cou?ie=UTF8&nodeId=508088) and you agree to be bound by them, with the following additional conditions:

In addition to the license rights granted under the Conditions of Use, Amazon or its content providers grant you a limited, non-exclusive, non-transferable, non-sublicensable, revocable license to access and use the Reviews Corpus for purposes of academic research. You may not resell, republish, or make any commercial use of the Reviews Corpus or its contents, including use of the Reviews Corpus for commercial research, such as research related to a funding or consultancy contract, internship, or other relationship in which the results are provided for a fee or delivered to a for-profit organization. You may not (a) link or associate content in the Reviews Corpus with any personal information (including Amazon customer accounts), or (b) attempt to determine the identity of the author of any content in the Reviews Corpus. If you violate any of the foregoing conditions, your license to access and use the Reviews Corpus will automatically terminate without prejudice to any of the other rights or remedies Amazon may have.
</details>

# Project description
This project was made for [Acámica's](https://www.acamica.com/data-science) data science bootcamp and consists of two iterations (click collapsible sections to see details of each):
<details>
<summary>I</summary>

## Iteration one

The first model starts with a comprehensive `EDA` and applying some simple `data cleaning` techniques to end up building a `machine learning regression model` that performs better than a linear regression. Two models have been tried in this project: **decisionTreeRegressor** and **kNeighborsRegressor**

### Index
- Scope
- EDA and data cleaning
- Preprocessing
  - Data wrangling
  - Text normalization: stemming and lemmatization
- Machine Learning
  - Selection of measurement of error
  - Data wrangling
  - Data vectorization
  - Setting benchmark model
  - Model training and comparing benchmark with the following models:
    - LinearSVC
    - RandomForest
  - Optimization of the best performing one
- Conclusions
- Next steps

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
