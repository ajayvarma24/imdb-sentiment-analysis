IMDb Movie Review Sentiment Analysis
Overview
This project is a Django-based web application that performs sentiment analysis on IMDb movie reviews. Users can register, log in, and predict the sentiment of movie reviews using machine learning models. The system also provides visualization of training results and maintains a history of user predictions.

Features
User registration and login

Sentiment prediction on movie reviews

Display of prediction results with confidence

History of reviewed movies for each user

Admin panel to manage users and data (admin/admin)

Visualization of model training results

Multiple ML models trained: Multinomial Naive Bayes (MNB), XGBoost, SVM, KNN, Logistic Regression, Decision Tree

Why Multinomial Naive Bayes (MNB)?
Although XGBoost shows higher accuracy on the dataset, we use MNB as the default prediction model because:

MNB is faster and more efficient for text classification tasks like sentiment analysis

It requires less computational resources, making the web app more responsive

MNB performs robustly with text data when combined with TF-IDF features

It ensures smoother user experience, especially on shared or limited hosting environments

Users can consider integrating other models like XGBoost for improved accuracy if computational resources allow.

Dataset
The IMDb dataset containing movie reviews is used for training. Due to file size limits on GitHub, the dataset is tracked using Git Large File Storage (Git LFS). Make sure to install Git LFS before cloning the repository to handle the dataset properly.

Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/ajayvarma24/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run migrations:

bash
Copy
Edit
python manage.py migrate
Create a superuser or use admin credentials below to access admin panel.

Start the server:

bash
Copy
Edit
python manage.py runserver
Usage
Visit http://localhost:8000 in your browser

Register or log in

Navigate to the prediction page to enter movie reviews and get sentiment predictions

Admin panel is available at /admin with username: admin and password: admin

Notes
For large files like the dataset, Git LFS is used. Ensure Git LFS is installed for proper cloning and pushing.

Future improvements include enabling model selection for predictions and improving UI responsiveness.

Author
Ajay Varma — GitHub Profile
