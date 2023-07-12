# Sentiment Analysis

This project focuses on sentiment analysis using machine learning algorithms to classify text data as positive or negative sentiment. It includes data preprocessing, feature extraction, model training, and evaluation.

## Project Overview

Sentiment analysis is the process of determining the sentiment or emotional tone of a given text. In this project, we aim to build a sentiment analysis model using machine learning techniques. The project utilizes a dataset containing text reviews of movies.

The project follows the following steps:

1. **Data Loading**: The dataset is loaded into the project.

2. **Data Preprocessing**: The text data is preprocessed to clean and transform it into a suitable format for analysis. This includes removing special characters, digits, and stopwords, converting text to lowercase, and handling any missing values.

3. **Exploratory Data Analysis**: Basic exploratory analysis is performed to gain insights into the dataset. Visualizations, such as count plots, are used to understand the distribution of sentiment labels in the data.

4. **Feature Extraction**: Text data is transformed into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) technique. The TF-IDF vectorizer converts the text into a matrix representation, which can be used as input for machine learning models.

5. **Model Training and Evaluation**: Several machine learning models are trained and evaluated for sentiment analysis. The models used in this project are:

   - K-Nearest Neighbors (KNN)
   - Logistic Regression (LR)
   - Naive Bayes (NB)
   - Decision Tree (DT)

   The dataset is split into training and testing sets. Each model is trained on the training set and evaluated on the testing set using various evaluation metrics such as accuracy, precision, recall, and F1-score.

6. **Results and Performance Comparison**: The results of the trained models are compared to determine their performance. Visualizations, such as bar plots, are used to compare the training time, testing time, and accuracy of each model.



## Technologies Used

- Python
- Flask
- Azure
- pandas
- re
- sklearn
- numpy
- nltk
- seaborn
- matplotlib

