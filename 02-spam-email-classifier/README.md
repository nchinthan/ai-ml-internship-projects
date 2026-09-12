\# Spam Email Classifier (NLP)



A Natural Language Processing (NLP) project that uses TF-IDF Vectorization and a Multinomial Naive Bayes model to classify text messages and emails as either \*\*Spam\*\* or \*\*Ham\*\* (Legitimate).



\## Project Overview



\- \*\*Dataset:\*\* SMS Spam Collection Dataset (\~5,572 labeled messages)

\- \*\*Techniques Used:\*\* Text Normalization, TF-IDF Vectorization, Multinomial Naive Bayes

\- \*\*Framework:\*\* Python, Scikit-Learn, NLTK, Pandas

\- \*\*Goal:\*\* Build a lightweight classifier with high precision to detect unsolicited spam messages.



\## Pipeline Architecture



1\. \*\*Text Preprocessing:\*\* Lowercasing, removing punctuation, and stripping stopwords.

2\. \*\*Feature Extraction:\*\* Conversion of raw text strings into numerical vectors using `TfidfVectorizer`.

3\. \*\*Model Training:\*\* `MultinomialNB` classifier trained on processed text vectors.

4\. \*\*Evaluation:\*\* Evaluated using Accuracy, Precision, Recall, and Confusion Matrix.



\## How to Run



\### 1. Install Dependencies

```bash

pip install pandas scikit-learn nltk joblib

