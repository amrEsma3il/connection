#!/usr/bin/env python
# coding: utf-8

# # **Import Libraries**

# In[1]:


# Basic Libraries
import numpy as np
import pandas as pd
import sklearn

# Necessary Libraries for Data Preparation
import string
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Necessary Libraries for ML Models
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Necessary Libraries for Accuracy Measures
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Necessary Libraries for Deployment
import joblib


# **Download NLTK resources**

# In[2]:


# download the Punkt tokenizer models
nltk.download('punkt')

# download a list of common stopwords
nltk.download('stopwords')

# download the WordNet lexical database
nltk.download('wordnet')


# # **Read the Data**

# In[3]:


data = pd.read_csv('Symptom2Disease.csv')


# # **Understand and Clean the Data**

# In[4]:


data


# In[5]:


# Drop the 'Unnamed: 0' column
data.drop(columns = ["Unnamed: 0"], inplace = True)
data


# In[6]:


# Concise summary of the DataFrame's structure and content
data.info()


# In[7]:


data.columns


# In[8]:


data.shape


# In[9]:


# Count the number of unique values in each column
data.nunique()


# In[10]:


data.value_counts().sum()


# In[11]:


# Check and Count null values
data.isnull().sum()


# In[12]:


# Check and Count duplicated values
data.duplicated().sum()


# In[13]:


# Drop duplicated values
data.drop_duplicates(inplace = True)
data


# # **Text Preprocessing--->(NLP)**

# In[14]:


def lowercase_text(text):
    return text.lower()

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    text_without_punct = text.translate(translator).strip()
    return text_without_punct

def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return filtered_tokens

def lemmatize_text(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens


# In[15]:


# Preprocessing Container function

def preprocess_text(text):
    text = lowercase_text(text)
    text = remove_punctuation(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_text(tokens)
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


# In[16]:


# Apply Preprocessing Container function to symptoms

data['text'] = data['text'].apply(preprocess_text)


# In[17]:


# Extract and Count unique dictionary vocabs

def count_unique_vocab(count):
    unique_vocabularies = set()
    for text in count:
        words = text.split()
        for word in words:
            unique_vocabularies.add(word)
    return len(unique_vocabularies)

# Count unique dictionary vocabs
num_unique_vocabs = count_unique_vocab(data['text'])

print("Number of unique dictionary vocabs:", num_unique_vocabs)


# # **Select the features (X) as 'text' column and target (y) as 'label' column**

# In[18]:


X = data['text']
y = data['label']


# In[19]:


X


# In[20]:


y


# In[21]:


# The 'shuffle' function is used to randomly Shuffle/Rearrange the elements of a dataset
from sklearn.utils import shuffle
data = shuffle(data, random_state = 42)
data


# In[22]:


# Charactieristics of the data
info = data.describe().round()
info


# # **Splitting the Dataset into Train set and Test set**

# In[23]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)


# # **Text Feature Extraction**

# In[24]:


# Text feature extraction using TF-IDF vectorizer to transform text data
tfidf_vectorizer = TfidfVectorizer(max_features=2400)

# Transforming training and testing data
X_train = tfidf_vectorizer.fit_transform(X_train).toarray()
X_test = tfidf_vectorizer.transform(X_test).toarray()


# In[25]:


def tfidf_vectorize_text(text_data, max_features=2400):
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(text_data)
    return tfidf_matrix


# 
# 
# ---
# 
# 

# ###############################################
# # ***Machine Learning Models***
# 
# ###############################################

# # **Decision Tree Classifier**
# 

# ## **Applying Grid Search to find the best model version and the best hyperparameters**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Create a Decision Tree Classifier object
dt_classifier = DecisionTreeClassifier(random_state = 42)

# Define the hyperparameters and their possible values to search
parameters = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features' : ['sqrt', 'log2', None]
}

# Create the Grid Search object
grid_search = GridSearchCV(estimator = dt_classifier,
                           param_grid = parameters,
                           cv = 5,
                           scoring = 'accuracy',
                           n_jobs = -1)

# Fit the Grid Search to the train data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters found
best_hyperparameters = grid_search.best_params_
print("Best Hyperparameters:", best_hyperparameters)

# Get the best model version
best_dt_classifier = grid_search.best_estimator_
print(best_dt_classifier)

# Print the best accuracy found
best_accuracy = grid_search.best_score_
print(f'Best Accuracy: {best_accuracy*100:.2f} %')


# ## **Model Evaluation**

# In[ ]:


# Calculate and Compare the Score of train data and test data

train_score = best_dt_classifier.score(X_train, y_train)
test_score = best_dt_classifier.score(X_test, y_test)

# Print the scores
print(f'Training Score: {train_score*100:.2f} %')
print(f'Testing Score: {test_score*100:.2f} %')


# In[ ]:


# Make Predictions on the train data and test data

train_predictions = best_dt_classifier.predict(X_train)
test_predictions = best_dt_classifier.predict(X_test)


# In[ ]:


# Calculate and Compare the Accuracy for training and testing data
from sklearn.metrics import accuracy_score

train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

# Print the accuracies
print(f'Training Accuracy: {train_accuracy*100:.2f} %')
print(f'Testing Accuracy: {test_accuracy*100:.2f} %')


# In[ ]:


# Make the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm_1 = confusion_matrix(y_test, test_predictions)

# Print the Confusion Matrix
print(cm_1)


# ## **Model Validation**

# In[ ]:


# Validation Test #

# text_before = "The skin around my mouth, nose, and eyes is ruddy and kindled. It is regularly bothersome and awkward. There's a recognizable aggravation in my nails."

text_before = "The abdominal pain has been coming and going, and it's been really unpleasant. It's been accompanied by constipation and vomiting. I feel really concerned about my health."

# Cleaning
text_after = preprocess_text(text_before)

print(text_before)
print(text_after)

# Vectorization
tfidf_vectorizer

text_after = tfidf_vectorizer.transform([text_after]).toarray()

print(text_after.reshape(-1,1))

# Prediction
test_predictions = best_dt_classifier.predict(text_after)

print(test_predictions)


# # **Random Forest Classifier**
# 
# Random Forest is an ensemble learning algorithm,
# It works by constructing multiple decision trees.

# ## **Applying Grid Search to find the best model version and the best hyperparameters**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Create a Random Forest Classifier object
rf_classifier = RandomForestClassifier(random_state = 42)

# Define the hyperparameters and their possible values to search
parameters = {
    'n_estimators': [10, 50, 100],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features' : ['sqrt', 'log2', None]
}

# Create the Grid Search object
grid_search = GridSearchCV(estimator = rf_classifier,
                           param_grid = parameters,
                           cv = 5,
                           scoring = 'accuracy',
                           n_jobs = -1)

# Fit the Grid Search to the train data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters found
best_hyperparameters = grid_search.best_params_
print("Best Hyperparameters:", best_hyperparameters)

# Get the best model version
best_rf_classifier = grid_search.best_estimator_
print(best_rf_classifier)

# Get the best accuracy found
best_accuracy = grid_search.best_score_
print(f'Best Accuracy: {best_accuracy*100:.2f} %')


# ## **Model Evaluation**

# In[ ]:


# Calculate and Compare the Score of train data and test data

train_score = best_rf_classifier.score(X_train, y_train)
test_score = best_rf_classifier.score(X_test, y_test)

# Print the scores
print(f'Training Score: {train_score*100:.2f} %')
print(f'Testing Score: {test_score*100:.2f} %')


# In[ ]:


# Make Predictions on the train data and test data

train_predictions = best_rf_classifier.predict(X_train)
test_predictions = best_rf_classifier.predict(X_test)


# In[ ]:


# Calculate and Compare the Accuracy for training and testing data
from sklearn.metrics import accuracy_score

train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

# Print the accuracies
print(f'Training Accuracy: {train_accuracy*100:.2f} %')
print(f'Testing Accuracy: {test_accuracy*100:.2f} %')


# In[ ]:


# Make the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm_2 = confusion_matrix(y_test, test_predictions)

# Print the Confusion Matrix
print(cm_2)


# ## **Model Validation**

# In[ ]:


# Validation Test #

# text_before = "The skin around my mouth, nose, and eyes is ruddy and kindled. It is regularly bothersome and awkward. There's a recognizable aggravation in my nails."

text_before = "The abdominal pain has been coming and going, and it's been really unpleasant. It's been accompanied by constipation and vomiting. I feel really concerned about my health."

# Cleaning
text_after = preprocess_text(text_before)

print(text_before)
print(text_after)

# Vectorization
tfidf_vectorizer

text_after = tfidf_vectorizer.transform([text_after]).toarray()

print(text_after.reshape(-1,1))

# Prediction
test_predictions = best_rf_classifier.predict(text_after)

print(test_predictions)


# # **Support Vector Machine (SVM)**
# 
# Its primary purpose is to find a hyperplane that best separates data points into different classes.

# ## **Applying Grid Search to find the best model version and the best hyperparameters**

# In[26]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Create a Support Vector Machine Classifier object
svm_classifier = SVC(random_state = 42)

# Define the hyperparameters and their possible values to search
parameters = [{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
              {'C': [0.25, 0.5, 0.75, 1],'kernel': ['rbf'], 'gamma': ['scale', 'auto', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}]

# Create the Grid Search object
grid_search = GridSearchCV(estimator = svm_classifier,
                           param_grid = parameters,
                           cv = 5,
                           scoring = 'accuracy',
                           n_jobs = -1)

# Fit the Grid Search to the train data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters found
best_hyperparameters = grid_search.best_params_
print("Best Hyperparameters:", best_hyperparameters)

# Get the best model version
best_svm_classifier = grid_search.best_estimator_
print(best_svm_classifier)

# Print the best accuracy found
best_accuracy = grid_search.best_score_
print(f'Best Accuracy: {best_accuracy*100:.2f} %')


# In[ ]:


best_svm_classifier = SVC(C=1, kernel='linear', random_state=42)
best_svm_classifier.fit(X_train, y_train)


# In[54]:


# Download pretrained Model
import joblib
joblib.dump(best_svm_classifier, "model_SVM.pkl")


# In[55]:


# Load pretrained Model
loaded_model = joblib.load("model_SVM.pkl")
loaded_model


# ## **Model Evaluation**

# In[27]:


# Calculate and Compare the Score of train data and test data

train_score = best_svm_classifier.score(X_train, y_train)
test_score = best_svm_classifier.score(X_test, y_test)

# Print the scores
print(f'Training Score: {train_score*100:.2f} %')
print(f'Testing Score: {test_score*100:.2f} %')


# In[28]:


# Make Predictions on the train data and test data

train_predictions = best_svm_classifier.predict(X_train)
test_predictions = best_svm_classifier.predict(X_test)


# In[29]:


# Calculate and Compare the Accuracy for training and testing data
from sklearn.metrics import accuracy_score

train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

# Print the accuracies
print(f'Training Accuracy: {train_accuracy*100:.2f} %')
print(f'Testing Accuracy: {test_accuracy*100:.2f} %')


# In[30]:


# Make the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm_3 = confusion_matrix(y_test, test_predictions)

# Print the Confusion Matrix
print(cm_3)


# ## **Model Validation**

# In[36]:


# Validation Test #

# text_before = "The skin around my mouth, nose, and eyes is ruddy and kindled. It is regularly bothersome and awkward. There's a recognizable aggravation in my nails."

text_before = "wdwwdsvwsdfjwem;knl,,cod"

# Cleaning
text_after = preprocess_text(text_before)

print(text_before)
print(text_after)

# Vectorization
tfidf_vectorizer

text_after = tfidf_vectorizer.transform([text_after]).toarray()

print(text_after.reshape(-1,1))

# Prediction
test_predictions = best_svm_classifier.predict(text_after)

print(test_predictions)


# # **KNeighborsClassifier**
# In the k-NN algorithm, the "k" represents the number of nearest neighbors considered for making predictions.
# 

# ## **Applying Grid Search to find the best model version and the best hyperparameters**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Create a K-Neighbors Classifier object
knn_classifier = KNeighborsClassifier()

# Define the hyperparameters and their possible values to search
parameters = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [1, 2]
}

# Create the Grid Search object
grid_search = GridSearchCV(estimator = knn_classifier,
                           param_grid = parameters,
                           cv = 5,
                           scoring = 'accuracy',
                           n_jobs = -1)

# Fit the Grid Search to the train data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters found
best_hyperparameters = grid_search.best_params_
print("Best Hyperparameters:", best_hyperparameters)

# Get the best model version
best_knn_classifier = grid_search.best_estimator_
print(best_knn_classifier)

# Print the best accuracy found
best_accuracy = grid_search.best_score_
print(f'Best Accuracy: {best_accuracy*100:.2f} %')


# ## **Model Evaluation**

# In[ ]:


# Calculate and Compare the Score of train data and test data

train_score = best_knn_classifier.score(X_train, y_train)
test_score = best_knn_classifier.score(X_test, y_test)

# Print the scores
print(f'Training Score: {train_score*100:.2f} %')
print(f'Testing Score: {test_score*100:.2f} %')


# In[ ]:


# Make Predictions on the train data and test data

train_predictions = best_knn_classifier.predict(X_train)
test_predictions = best_knn_classifier.predict(X_test)


# In[ ]:


# Calculate and Compare the Accuracy for training and testing data
from sklearn.metrics import accuracy_score

train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

# Print the accuracies
print(f'Training Accuracy: {train_accuracy*100:.2f} %')
print(f'Testing Accuracy: {test_accuracy*100:.2f} %')


# In[ ]:


# Make the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm_5 = confusion_matrix(y_test, test_predictions)

# Print the Confusion Matrix
print(cm_5)


# ## **Model Validation**

# In[ ]:


# Validation Test #

# text_before = "The skin around my mouth, nose, and eyes is ruddy and kindled. It is regularly bothersome and awkward. There's a recognizable aggravation in my nails."

text_before = "The abdominal pain has been coming and going, and it's been really unpleasant. It's been accompanied by constipation and vomiting. I feel really concerned about my health."

# Cleaning
text_after = preprocess_text(text_before)

print(text_before)
print(text_after)

# Vectorization
tfidf_vectorizer

text_after = tfidf_vectorizer.transform([text_after]).toarray()

print(text_after.reshape(-1,1))

# Prediction
test_predictions = best_knn_classifier.predict(text_after)

print(test_predictions)


# In[ ]:




