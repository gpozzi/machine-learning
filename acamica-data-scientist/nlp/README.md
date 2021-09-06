# Project at a glance: Natural Language Processing 💬

The goal of this project is to be able to predict the score that a user would set given a review by using an Amazon dataset that contained reviews about different products.

<img src="https://image.freepik.com/vector-gratis/ilustracion-concepto-abstracto-inteligencia-artificial-chatbot_335657-3723.jpg" width=300>

- Project tools / libraries:
  - `Python`
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
<summary>Iteration one</summary>

## Iteration one

The first model starts with a comprehensive `EDA` and preprocessing, which includes text normalization through SpaCy pipeline (`tok2vec`, `morphologizer`, `parser`, `ner`, `attribute_ruler` and `lemmatizer`), stopwords removal and lemmatization in order to improve model's accuracy. Then, a performance metric has been selected.
  
After that, a benchmark model has been developed and compared with other two other models in order to determine the best performing one. The one that showed the best accuracy has had its hyperparameters optimized and finally some conclusions have been made about the performance and the methodology applied.

### Index
- [Scope](https://render.githubusercontent.com/view/ipynb?color_mode=light&commit=57a0e2af23b7242d0f2ca6bf955e0fb9ce619ae2&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f67706f7a7a692f6d616368696e652d6c6561726e696e672f353761306532616632336237323432643066326361366266393535653066623963653631396165322f6163616d6963612d646174612d736369656e746973742f6e6c702f44535f50726f6a6563745f30335f4e4c502e6970796e62&nwo=gpozzi%2Fmachine-learning&path=acamica-data-scientist%2Fnlp%2FDS_Project_03_NLP.ipynb&repository_id=273610133&repository_type=Repository#1.-Scope)
- [EDA and preprocessing](https://render.githubusercontent.com/view/ipynb?color_mode=light&commit=57a0e2af23b7242d0f2ca6bf955e0fb9ce619ae2&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f67706f7a7a692f6d616368696e652d6c6561726e696e672f353761306532616632336237323432643066326361366266393535653066623963653631396165322f6163616d6963612d646174612d736369656e746973742f6e6c702f44535f50726f6a6563745f30335f4e4c502e6970796e62&nwo=gpozzi%2Fmachine-learning&path=acamica-data-scientist%2Fnlp%2FDS_Project_03_NLP.ipynb&repository_id=273610133&repository_type=Repository#2.-EDA-and-preprocessing)
  - [EDA](https://render.githubusercontent.com/view/ipynb?color_mode=light&commit=57a0e2af23b7242d0f2ca6bf955e0fb9ce619ae2&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f67706f7a7a692f6d616368696e652d6c6561726e696e672f353761306532616632336237323432643066326361366266393535653066623963653631396165322f6163616d6963612d646174612d736369656e746973742f6e6c702f44535f50726f6a6563745f30335f4e4c502e6970796e62&nwo=gpozzi%2Fmachine-learning&path=acamica-data-scientist%2Fnlp%2FDS_Project_03_NLP.ipynb&repository_id=273610133&repository_type=Repository#EDA)
  - [Review preprocessing](https://render.githubusercontent.com/view/ipynb?color_mode=light&commit=57a0e2af23b7242d0f2ca6bf955e0fb9ce619ae2&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f67706f7a7a692f6d616368696e652d6c6561726e696e672f353761306532616632336237323432643066326361366266393535653066623963653631396165322f6163616d6963612d646174612d736369656e746973742f6e6c702f44535f50726f6a6563745f30335f4e4c502e6970796e62&nwo=gpozzi%2Fmachine-learning&path=acamica-data-scientist%2Fnlp%2FDS_Project_03_NLP.ipynb&repository_id=273610133&repository_type=Repository#Review-preprocessing)
- [Machine Learning](https://render.githubusercontent.com/view/ipynb?color_mode=light&commit=57a0e2af23b7242d0f2ca6bf955e0fb9ce619ae2&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f67706f7a7a692f6d616368696e652d6c6561726e696e672f353761306532616632336237323432643066326361366266393535653066623963653631396165322f6163616d6963612d646174612d736369656e746973742f6e6c702f44535f50726f6a6563745f30335f4e4c502e6970796e62&nwo=gpozzi%2Fmachine-learning&path=acamica-data-scientist%2Fnlp%2FDS_Project_03_NLP.ipynb&repository_id=273610133&repository_type=Repository#3.-Machine-Learning)
  - [Metric selection](https://render.githubusercontent.com/view/ipynb?color_mode=light&commit=57a0e2af23b7242d0f2ca6bf955e0fb9ce619ae2&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f67706f7a7a692f6d616368696e652d6c6561726e696e672f353761306532616632336237323432643066326361366266393535653066623963653631396165322f6163616d6963612d646174612d736369656e746973742f6e6c702f44535f50726f6a6563745f30335f4e4c502e6970796e62&nwo=gpozzi%2Fmachine-learning&path=acamica-data-scientist%2Fnlp%2FDS_Project_03_NLP.ipynb&repository_id=273610133&repository_type=Repository#Metric-selection)
  - [Transformations](https://render.githubusercontent.com/view/ipynb?color_mode=light&commit=57a0e2af23b7242d0f2ca6bf955e0fb9ce619ae2&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f67706f7a7a692f6d616368696e652d6c6561726e696e672f353761306532616632336237323432643066326361366266393535653066623963653631396165322f6163616d6963612d646174612d736369656e746973742f6e6c702f44535f50726f6a6563745f30335f4e4c502e6970796e62&nwo=gpozzi%2Fmachine-learning&path=acamica-data-scientist%2Fnlp%2FDS_Project_03_NLP.ipynb&repository_id=273610133&repository_type=Repository#Transformations)
  - Data vectorization
  - Setting benchmark model
  - Model training and comparing benchmark with the following models:
    - LinearSVC
    - RandomForest
  - Optimization of the best performing one
- Conclusions
- Next steps

### Some visuals

![image](https://user-images.githubusercontent.com/52865532/131766302-ce54c9ad-f9a7-469c-8724-0090330990d6.png)
  
![image](https://user-images.githubusercontent.com/52865532/131766441-52b030de-2f9f-44ce-be38-70836e689c69.png)
  
![image](https://user-images.githubusercontent.com/52865532/131766565-362cd79a-2065-4686-b72a-1306c63abed5.png)

![image](https://user-images.githubusercontent.com/52865532/131769133-a7a6558d-83db-4a7a-b342-56a39b3df26a.png)
  
![image](https://user-images.githubusercontent.com/52865532/131769084-afdd0fcb-19cc-490d-9a37-90c0f273ce5a.png)
  
### Conclusions

We can conclude that little can be done to improve the performance of the model, either by adjusting the Tf-idf or optimizing model's hyperparameters. From the confusion matrix we see that with some adjustments the ability to predict 4 stars improved a bit to almost match the performance of 2 stars, as opposed to the non-optimized one. Reviews of intermediate scores will mark the roof of the model's performance, and beyond rigorous optimization the gains from a certain point will be marginal.

There will always be a limit to the ability of a Machine Learning model to classify scores on a scale of 1 to 5. This is due to an inherent limitation of the language due to the lack of distinctive words in intermediate reviews and because they tend to have as many good words as bad ones.

Beyond this, the classification of the reviews in 5 classes, depending on the use of this information, appears to be trivial, since knowing if a product has 2 or 3 stars would not provide valuable and actionable information. Converting this problem to a binary classification (positive / negative) could be more practical and at the same time would greatly improve the performance of the model. This work will be done in the next iteration.
</details>

<details>
<summary>Iteration two</summary>

## Iteration two

  ### Index
- Scope
- Recap: iteration I
  - Results
- Iteration II: model repurposing
  - Motivation: the 5-star scale problem
  - Preprocessing
  - Vectorization
  - Model training and optimization
- Analysis: 3-star reviews
- Conclusions
  
  ### Some visuals
