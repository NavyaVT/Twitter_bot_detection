import pandas as pd
import numpy as np
import re
import string
import joblib
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from transformers import BertTokenizer, BertModel
from twitter_X import getDetails
# Load the dataset
df = pd.read_csv('bot_detection_data.csv')

# Data Preprocessing Functions
def clean_text(text):
    """ Clean tweet by removing URLs and punctuation """
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def lower(text):
    """ Convert text to lowercase """
    return str(text).lower()

# Apply text preprocessing
df['Tweet'] = df['Tweet'].apply(clean_text)
df['TweetLower'] = df['Tweet'].apply(lower)

# Convert Verified column to int (if not already)
df['Verified'] = df['Verified'].astype(int)

# Sentence Embeddings using BERT
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # You can use a better model like BERT or RoBERTa for better performance
X_text_embeddings = np.array(embedder.encode(df['Tweet'].tolist()))

# Add additional features: Tweet length, Hashtag count, Uppercase count
df['Tweet_Length'] = df['Tweet'].apply(len)
df['Hashtag_Count'] = df['Hashtags'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
df['Uppercase_Count'] = df['Tweet'].apply(lambda x: sum(1 for c in x if c.isupper()))

# Numeric Features
X_numerical = df[['Retweet Count', 'Mention Count', 'Follower Count', 'Verified', 'Tweet_Length', 'Hashtag_Count', 'Uppercase_Count']].astype(float).values

# Combine text embeddings with numerical features
X = np.hstack((X_text_embeddings, X_numerical))
y = df['Bot Label']  # Label: 1 = Bot, 0 = Human

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE for class imbalance if necessary
smote = SMOTE()
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Hyperparameter tuning using GridSearchCV for XGBoost
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.05, 0.1]
}

grid_search = GridSearchCV(XGBClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train_res, y_train_res)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Save the model and sentence transformer for future use
joblib.dump(best_model, "bot_detector.pkl")
joblib.dump(embedder, "text_embedder.pkl")

# Evaluate the model on the test set
y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Predicting new tweet data
def predict_bot(tweet, retweet_count, mention_count, follower_count, verified,hashes):
    """ Predict if a tweet is from a bot or human """
    # Load the models
    embedder = joblib.load("text_embedder.pkl")
    best_model = joblib.load("bot_detector.pkl")
    
    # Preprocess the tweet
    tweet_cleaned = clean_text(tweet)
    tweet_embedding = np.array(embedder.encode([tweet_cleaned.lower()]))
    
    # Combine features
    features = np.hstack((tweet_embedding, [[retweet_count, mention_count, follower_count, int(verified), len(tweet), len(str(hashes).split()), sum(1 for c in tweet if c.isupper())]]))
    
    # Predict using the trained model
    prediction = best_model.predict(features)
    return "Bot" if prediction[0] == 1 else "Human"

if __name__ == "__main__":
    example_tweet = "This is an automated message, please ignore."
    print("Prediction:", predict_bot(example_tweet, 10, 2, 1000, False,"anyone respond perhaps market run"))
