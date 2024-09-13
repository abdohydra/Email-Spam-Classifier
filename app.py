from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import nltk
import math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)
nltk.download('punkt')
nltk.download('stopwords')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Preprocess text function
def preprocess_text(text):
    text = text.lower()
    words = word_tokenize(text)
    filtered_words = [stemmer.stem(word) for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(filtered_words)

# Load dataset
spam_data = pd.read_csv('spam12.csv', encoding='latin-1')
# Remove unwanted columns and clean up data
spam_data.columns = spam_data.columns.str.strip()  # Remove leading/trailing spaces from column names
spam_data = spam_data[['label', 'email']]  # Retain only 'label' and 'email' columns
# Preprocess email content
spam_data['processed'] = spam_data['email'].apply(preprocess_text)

# Functions to calculate TF, IDF, and TF-IDF
def tf(datas):
    tflistforechemail = []
    for data in datas:
        sentencesliced = data["processed"].split()
        word_count = {}
        for word in sentencesliced:
            word_count[word] = word_count.get(word, 0) + 1
        dicforeachdata = {word: count / len(sentencesliced) for word, count in word_count.items()}
        dicforeachdata["label"] = data["label"]
        tflistforechemail.append(dicforeachdata)
    return tflistforechemail

def idf(datas):
    document_freq = {}
    num_docs = len(datas)
    for data in datas:
        sentencesliced = set(data["processed"].split())
        for word in sentencesliced:
            document_freq[word] = document_freq.get(word, 0) + 1
    return {word: math.log(num_docs / (1 + freq)) for word, freq in document_freq.items()}

def tf_idf(df, idf_values):
    listofdic = []
    for email in df:
        label = email.pop("label")
        dfidfvalue = {word: tf_val * idf_values[word] for word, tf_val in email.items() if word in idf_values}
        dfidfvalue["label"] = label
        listofdic.append(dfidfvalue)
    return listofdic
# Functions to calculate prior and likelihood probabilities
def prior_probability(datas, key1, key2):
    total = len(datas)
    class_counts = {key1: 0, key2: 0}
    for data in datas:
        class_counts[data["label"]] += 1
    return {key1: class_counts[key1] / total, key2: class_counts[key2] / total}

def likelihoods(tf_idf, key1, key2):
    dic_label_word_freq = {key1: {}, key2: {}}
    total_words_key1 = 0
    total_words_key2 = 0

    for email in tf_idf:
        label = email.pop("label")
        for word, value in email.items():
            if label == key1:
                dic_label_word_freq[key1][word] = dic_label_word_freq[key1].get(word, 0) + value
                total_words_key1 += value
            else:
                dic_label_word_freq[key2][word] = dic_label_word_freq[key2].get(word, 0) + value
                total_words_key2 += value

    likelihoods = {key1: {}, key2: {}}
    for word in dic_label_word_freq[key1]:
        likelihoods[key1][word] = dic_label_word_freq[key1][word] / total_words_key1
    for word in dic_label_word_freq[key2]:
        likelihoods[key2][word] = dic_label_word_freq[key2][word] / total_words_key2

    return likelihoods

def predict(email, prior, likely, key1, key2):
    score1 = math.log(prior[key1])
    score2 = math.log(prior[key2])

    words = email.split()
    for word in words:
        score1 += math.log(likely[key1].get(word, 1e-10))
        score2 += math.log(likely[key2].get(word, 1e-10))

    return key1 if score1 > score2 else key2

# Split the data into training and test sets
train_spam, test_spam = train_test_split(spam_data.to_dict(orient='records'), test_size=0.2, random_state=42)

# Training Naive Bayes classifier
tf_train_spam = tf(train_spam)
idf_train_spam = idf(train_spam)
tfidf_train_spam = tf_idf(tf_train_spam, idf_train_spam)
prior_train_spam = prior_probability(train_spam, "spam", "ham")
likelihood_train_spam = likelihoods(tfidf_train_spam, "spam", "ham")

# Compute metrics
def compute_metrics(test_data):
    y_true = [data['label'] for data in test_data]
    y_pred = [predict(preprocess_text(data['email']), prior_train_spam, likelihood_train_spam, 'spam', 'ham') for data in test_data]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label='spam', average='binary')
    recall = recall_score(y_true, y_pred, pos_label='spam', average='binary')
    f1 = f1_score(y_true, y_pred, pos_label='spam', average='binary')

    return accuracy, precision, recall, f1

accuracy, precision, recall, f1 = compute_metrics(test_spam)


@app.route('/')
def home():
    # Retrieve the prediction from query parameters if it exists
    prediction = request.args.get('prediction')

    # Render the page with the prediction
    # If there's a prediction, redirect to remove it from the URL after rendering
    if prediction:
        return render_template('index.html',
                               prediction=prediction,
                               accuracy=accuracy,
                               precision=precision,
                               recall=recall,
                               f1=f1)
    else:
        # Render normally if no prediction exists
        return render_template('index.html',
                               prediction=None,
                               accuracy=accuracy,
                               precision=precision,
                               recall=recall,
                               f1=f1)


@app.route('/classify', methods=['POST'])
def classify_email():
    email_content = request.form.get('email_content')
    if email_content:
        # Process the email content and classify it
        processed_email = preprocess_text(email_content)
        result = predict(processed_email, prior_train_spam, likelihood_train_spam, 'spam', 'ham')
        return redirect(url_for('home', prediction=result))
    else:
        return redirect(url_for('home'))



if __name__ == '__main__':
    app.run(debug=True)