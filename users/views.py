from django.shortcuts import render
import numpy as np
import pandas as pd
import os
from django.conf import settings

import pickle
# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


import re
import string

def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'<.*?>', '', text)  # remove HTML tags
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove punctuation and numbers
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra whitespace
    return text


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})

def DatasetView(request):
    path = settings.MEDIA_ROOT + "//" + 'IMDB_Dataset.csv'
    df = pd.read_csv(path ,nrows=100)
    df = df.to_html
    return render(request, 'users/viewdataset.html', {'data': df})

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
path = settings.MEDIA_ROOT + "//" + 'IMDB_Dataset.csv'
df = pd.read_csv(path)
value_counts = df['sentiment'].value_counts()
# print(value_counts)
# import matplotlib.pyplot as plt
# # Plotting the bar plot
# value_counts.plot(kind='bar')
# # Adding labels and title
# plt.xlabel('Categories')
# plt.ylabel('Counts')
# plt.title('Value Counts of Your Column')
# # Displaying the plot
# plt.show()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)
# Vectorize the tweets using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
# Train a Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

def training1(request):  
    # Adding labels and title
    plt.xlabel('Categories')
    plt.ylabel('Counts')
    plt.title('Value Counts of Your Column')
    # Displaying the plot
    plt.show()    
    predictions = nb_classifier.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")      
    nb = classification_report(y_test,predictions,output_dict=True)
    nb = pd.DataFrame(nb).transpose()    
    nb = pd.DataFrame(nb)
    return render(request,"users/training1.html",{'nb':nb.to_html,'acc':accuracy})





from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from django.conf import settings
from sklearn.preprocessing import LabelEncoder

def training(request):
    # Load dataset
    path = settings.MEDIA_ROOT + "//" + 'IMDB_Dataset.csv'
    df = pd.read_csv(path)
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    df['sentiment'] = label_encoder.fit_transform(df['sentiment'])
    X_train, X_test, y_train, y_test = train_test_split(
        df['review'], df['sentiment'], test_size=0.2, random_state=42
    )

    # TF-IDF vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=10000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Classifier dictionary
    classifiers = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    reports = {}
    accuracies = {}

    # Train and evaluate each classifier
    for name, model in classifiers.items():
        model.fit(X_train_tfidf, y_train)
        predictions = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, predictions)
        accuracies[name] = f"{acc:.2f}"
        report = classification_report(y_test, predictions, output_dict=True)
        reports[name] = pd.DataFrame(report).transpose()

    # Combine all reports into one HTML string
    reports_html = {
        name: df.to_html(classes="table table-bordered", border=0)
        for name, df in reports.items()
    }

    return render(request, "users/training.html", {
        'reports': reports_html,
        'accuracies': accuracies
    })


# def prediction(request):
#     if request.method == 'POST':
#         single_tweet = request.POST.get('tweets') 
#         print(single_tweet)      
#         single_tweet_tfidf = tfidf_vectorizer.transform([single_tweet])
#         print('manohar',single_tweet_tfidf)
#         # Make prediction
#         single_prediction = nb_classifier.predict(single_tweet_tfidf)
#         print(single_prediction)
#         # Print prediction
#         print(f'Tweet: {single_tweet} - Predicted Emotion: {single_prediction[0]}')
#         if single_prediction[0] == 0:
#             single_prediction='positive'
#         elif single_prediction[0] == 1:
#             single_prediction='negative'
#         return render(request, 'users/predictForm.html', {'output':single_prediction})
#     return render(request, 'users/predictForm.html', {})

def prediction(request):
    if request.method == 'POST':
        single_tweet = request.POST.get('tweets')
        cleaned_text = clean_text(single_tweet)  # Apply cleaning
        single_tweet_tfidf = tfidf_vectorizer.transform([cleaned_text])
        single_prediction = nb_classifier.predict(single_tweet_tfidf)

        if single_prediction[0] == 0:
            single_prediction = 'positive'
        elif single_prediction[0] == 1:
            single_prediction = 'negative'

        return render(request, 'users/predictForm.html', {
            'output': single_prediction,
            'user_input': single_tweet,
            'input_text': cleaned_text
        })

    return render(request, 'users/predictForm.html', {})


from admins.models import InputData
from django.shortcuts import render

def show_admin_inputs(request):
    inputs = InputData.objects.all()
    return render(request, 'users/show_admin_data.html', {'inputs': inputs})
from django.shortcuts import render, get_object_or_404
from admins.models import InputData

# def predict_view(request, id):
#     input_data = get_object_or_404(InputData, id=id)
#     return render(request, 'users/predictForm.html', {
#         'input_text': input_data.text,
#         'input_id': input_data.id
#     })
from django.shortcuts import render, get_object_or_404
from admins.models import InputData
import os
from django.conf import settings

from django.shortcuts import render, get_object_or_404
from admins.models import InputData

# def predict_view(request, input_id):
#     input_text = get_object_or_404(InputData, id=input_id).text
#     output = None
#     user_input = None

#     if request.method == 'POST':
#         user_input = request.POST.get('user_input')

#         # === Replace with your actual ML model prediction ===
#         # Example: output = model.predict([user_input])[0]
#         output = "positive" if "good" in user_input.lower() else "negative"

#         # Save to output.txt
#         with open('output.txt', 'a') as file:
#             file.write(f"{output},{user_input},{input_text}\n")

#     return render(request, 'users/predictForm.html', {
#         'input_text': input_text,
#         'user_input': user_input,
#         'output': output
#     })
from django.shortcuts import render, get_object_or_404
from admins.models import InputData

# Import your ML components
# Ensure these are defined earlier in your code
# from your_model_file import tfidf_vectorizer, nb_classifier


from django.shortcuts import render, get_object_or_404
from admins.models import InputData
import joblib

# Load your ML model and vectorizer (only once)
# tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')  # path to your saved vectorizer
# nb_classifier = joblib.load('nb_classifier.pkl')        # path to your trained model

# def predict_view(request, input_id):
#     input_text = get_object_or_404(InputData, id=input_id).text
#     output = None
#     user_input = None
#     debug_info = ""

#     if request.method == 'POST':
#         user_input = request.POST.get('user_input')

#         # Preprocess and predict using the loaded model
#         try:
#             processed_input = tfidf_vectorizer.transform([user_input])
#             prediction_result = nb_classifier.predict(processed_input)[0]
#             debug_info = f"Processed: {processed_input}\nPrediction Code: {prediction_result}"

#             # Convert numeric prediction to readable label
#             output = 'positive' if prediction_result == 0 else 'negative'

#             # Save to output.txt
#             with open('output.txt', 'a') as file:
#                 file.write(f"{output},{user_input},{input_text}\n")

#         except Exception as e:
#             output = "Error"
#             debug_info = f"Exception during prediction: {str(e)}"

#     return render(request, 'users/predictForm.html', {
#     'input_text': input_text,
#     'user_input': user_input,
#     'output': output  # not prediction_result
# })

from django.shortcuts import render, get_object_or_404
from admins.models import InputData  # Adjust import as per your project
from datetime import datetime

def predict_view(request, input_id):
    input_data = get_object_or_404(InputData, id=input_id)
    input_text = input_data.text
    output = None
    user_input = None

    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        try:
            # Transform input using TF-IDF vectorizer
            processed_input = tfidf_vectorizer.transform([user_input])
            print('Processed TF-IDF:', processed_input)

            # Predict using the Naive Bayes classifier
            prediction_result = nb_classifier.predict(processed_input)
            print(f"Prediction Result: {prediction_result}")

            # Convert numeric prediction to label
            if prediction_result[0] == 0:
                output = 'positive'
            elif prediction_result[0] == 1:
                output = 'negative'
            else:
                output = f"{prediction_result[0]}"

            # Get current date and time
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Save to output.txt
            with open('output.txt', 'a') as file:
                file.write(f"{current_time},{output},{user_input},{input_text}\n")

        except Exception as e:
            output = "Error"
            print(f"Exception during prediction: {str(e)}")

    return render(request, 'users/predictForm.html', {
        'input_text': input_text,
        'user_input': user_input,
        'output': output
    })

# from django.shortcuts import render, get_object_or_404
# from admins.models import InputData  # Adjust import as per your project
# from datetime import datetime

# def predict_view(request, input_id):
#     input_data = get_object_or_404(InputData, id=input_id)
#     input_text = input_data.text
#     output = None
#     user_input = None

#     if request.method == 'POST':
#         user_input = request.POST.get('user_input')
#         try:
#             # Transform input using TF-IDF vectorizer
#             processed_input = tfidf_vectorizer.transform([user_input])
#             print('Processed TF-IDF:', processed_input)

#             # Predict using the Naive Bayes classifier
#             prediction_result = nb_classifier.predict(processed_input)
#             print(f"Prediction Result: {prediction_result}")

#             # Convert numeric prediction to label
#             if prediction_result[0] == 0:
#                 output = 'positive'
#             elif prediction_result[0] == 1:
#                 output = 'negative'
#             else:
#                 output = f"{prediction_result[0]}"

#             # Get current date and time
#             current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

#             # Save to output.txt
#             with open('output.txt', 'a') as file:
#                 file.write(f"{current_time},{output},{user_input},{input_text}\n")

#         except Exception as e:
#             output = "Error"
#             print(f"Exception during prediction: {str(e)}")

#     return render(request, 'users/predictForm.html', {
#         'input_text': input_text,
#         'user_input': user_input,
#         'output': output
#     })



from django.shortcuts import render

def show_predictions(request):
    predictions = []

    try:
        with open('output.txt', 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    parts = line.split(',', 4)  # output,user_input,input_text
                    if len(parts) == 4:
                        predictions.append({
                            "timestamp":parts[0],
                            'output': parts[1],
                            'user_input': parts[2],
                            'input_text': parts[3]
                        })
    except FileNotFoundError:
        predictions = []

    return render(request, 'users/show_predictions.html', {'predictions': predictions})




from collections import defaultdict, Counter
import json
from django.shortcuts import render

def show_piechart(request):
    film_sentiments = defaultdict(list)

    try:
        with open("output.txt", "r") as file:
            lines = file.readlines()

        # Parse each line and group sentiment by film name
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) >= 4:
                sentiment = parts[1].strip().lower()  # positive/negative
                film_name = parts[3].strip()          # film name
                film_sentiments[film_name].append(sentiment)

        # Prepare chart data for each film
        chart_data = []
        for film, sentiments in film_sentiments.items():
            counts = Counter(sentiments)
            chart_data.append({
                "film": film,
                "labels": ["Positive", "Negative"],
                "counts": [counts.get("positive", 0), counts.get("negative", 0)]
            })

    except FileNotFoundError:
        chart_data = []

    return render(request, "users/show_piechart.html", {
        "chart_data_json": json.dumps(chart_data)
    })


from django.shortcuts import render

def movie_gallery(request):
    # You can pass dynamic data if needed
    return render(request, 'users/training_results.html')
