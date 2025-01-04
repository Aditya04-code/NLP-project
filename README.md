# Natural Language Processing Project

## Project Overview  
Welcome to the **Natural Language Processing (NLP)** project! In this project, we will be classifying Yelp reviews into **1-star** or **5-star** categories based on the content of the reviews. The objective is to demonstrate the application of NLP techniques to classify reviews, and we will leverage **Pipeline Methods** for this task, which simplifies the implementation of more complex NLP tasks.

We will be using the **Yelp Review Dataset** available on **Kaggle**. The dataset contains various reviews by Yelp users about businesses.

## Dataset Description  
The **Yelp Review Dataset** consists of various columns, including:

- **stars**: The number of stars (ranging from 1 to 5) given by the reviewer to the business. Higher stars indicate better reviews.
- **cool**: The number of "cool" votes this review received from other Yelp users. This column reflects the rating of the review itself, not the business.
- **useful**: The number of "useful" votes this review received, indicating how helpful the review is to others.
- **funny**: The number of "funny" votes the review received, showing how amusing the review is perceived by other users.

The task is to classify reviews into either **1-star** or **5-star** categories based on the text content.

## Objective  
The main goal of this project is to:

- Process Yelp reviews using Natural Language Processing (NLP) techniques.
- Perform classification of reviews into **1-star** or **5-star** categories based on the review text.
- Use machine learning models for text classification and evaluate their performance.

## Methodology  
1. **Data Preprocessing**:  
   The first step is cleaning the text data by removing stopwords, special characters, and performing other text cleaning tasks like stemming and lemmatization.

2. **Feature Extraction**:  
   We will extract useful features from the text data using methods like **TF-IDF** (Term Frequency-Inverse Document Frequency), which transforms the raw text data into numerical features for the model.

3. **Model Building**:  
   A classification model (e.g., **Logistic Regression**, **Naive Bayes**, or **SVM**) will be built using the features extracted from the text. A **Pipeline** will be used to streamline the process of vectorizing the text data, training the model, and making predictions.

4. **Evaluation**:  
   The model will be evaluated using standard classification metrics like **Accuracy**, **Precision**, **Recall**, and **F1-Score**.

## Technologies Used  
- **Python**  
- **Pandas** (for data manipulation)  
- **Scikit-learn** (for machine learning and text processing)  
- **Natural Language Toolkit (NLTK)** or **spaCy** (for text processing)  
- **Matplotlib / Seaborn** (for data visualization)

## How It Works  
1. The dataset is loaded from Kaggle, which contains Yelp reviews and associated metadata.
2. We clean and preprocess the reviews (remove stopwords, punctuation, etc.).
3. We convert the text data into numerical format using **TF-IDF** or **CountVectorizer**.
4. We train a classification model on the preprocessed text data.
5. The model classifies each review into either **1-star** or **5-star** categories.

## Results  
The model achieved a decent level of accuracy in classifying reviews into **1-star** or **5-star** categories. Future improvements could include trying different classification algorithms, tuning hyperparameters, or performing sentiment analysis to gain deeper insights into the reviews.

## How to Run  
1. Clone the repository:
   ```bash
   git clone https://github.com/Aditya04-code/nlp-yelp-reviews.git
