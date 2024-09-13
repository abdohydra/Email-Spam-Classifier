# Email-Spam-Classifier
Spam Classifier with Naive Bayes and Flask
Overview
This project is a spam email classifier built using Flask and the Naive Bayes algorithm. It allows users to input an email and determine whether it is "spam" or "ham" based on real-time classification. The project implements essential machine learning techniques such as Term Frequency (TF), Inverse Document Frequency (IDF), and TF-IDF.


How It Works
1. Preprocessing
   Converts emails to lowercase.
   Tokenizes words using NLTK.
   Removes stopwords (common words) and applies stemming.
2. TF-IDF Calculation
   Calculates term frequency (TF) for each word in an email.
   Computes inverse document frequency (IDF) based on the entire dataset.
   Generates TF-IDF scores by combining TF and IDF values.
3. Naive Bayes Classification
   Trains the model on labeled email data (spam/ham).
   Uses the trained model to classify incoming emails based on their word frequencies and TF-IDF scores.
4. Web Interface
   Users can enter an email into the web form.
   The app processes the email, applies the classification model, and displays the result in real-time.
   
Docker Usage ( you should have Docker installed in your system )

    To run the application in a Docker container, use the following steps:

    Build the Docker image: docker build -t spam-classifier .
  
    Run the Docker container: run -p 5000:5000 spam-classifier

  Access the web application: Open your web browser and go to http://localhost:5000
 
Future Improvements
  Add more sophisticated preprocessing techniques, such as handling misspellings and using bigrams.
  Improve the user interface and make it more responsive.
  Enhance model performance by using advanced classifiers (e.g., Support Vector Machines or Neural Networks).

