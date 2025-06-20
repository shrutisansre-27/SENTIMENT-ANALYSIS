import pandas as pd
import numpy as np
import re
import string
from collections import Counter

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# --- 1. NLTK Downloads (Run this once if you don't have them) ---
print("--- Downloading NLTK resources (if not already present) ---")

# Download necessary NLTK data. This will check if already present.
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4') # Open Multilingual Wordnet, often needed for WordNetLemmatizer

print("--- NLTK downloads complete ---")

# --- Rest of your script follows from here ---

# --- 2. Data Loading ---
print("\n--- Loading Dataset ---")
# --- Configuration: SET YOUR DATASET PATH AND COLUMNS HERE ---
DATASET_PATH = 'your_dataset.csv' # e.g., 'tweets.csv' or 'reviews.csv'
TEXT_COLUMN = 'text'             # Name of the column containing the text
SENTIMENT_COLUMN = 'sentiment'   # Name of the column containing the sentiment labels

# Dummy dataset for demonstration if your_dataset.csv isn't found
try:
    # Attempt to load your specified dataset
    df = pd.read_csv(DATASET_PATH, encoding='utf-8')
    # Rename columns if they don't match your TEXT_COLUMN and SENTIMENT_COLUMN variables
    # For example, if your CSV has 'tweet_text' and 'label', you might do:
    # df = df.rename(columns={'tweet_text': TEXT_COLUMN, 'label': SENTIMENT_COLUMN})

    # For common datasets like Sentiment140, special handling might be needed:
    # If using 'training.1600000.processed.noemoticon.csv', uncomment and adjust below:
    # df = pd.read_csv(DATASET_PATH, encoding='ISO-8859-1', header=None)
    # df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
    # df = df[[TEXT_COLUMN, 'target']].copy() # Select only necessary columns
    # df.rename(columns={'target': SENTIMENT_COLUMN}, inplace=True)
    # df[SENTIMENT_COLUMN] = df[SENTIMENT_COLUMN].replace({0: 'negative', 4: 'positive'}) # Map 0 to negative, 4 to positive

    print(f"Successfully loaded '{DATASET_PATH}'")

except FileNotFoundError:
    print(f"'{DATASET_PATH}' not found. Using a dummy dataset for demonstration.")
    data = {
        'text': [
            "This is an amazing product! I absolutely love it, highly recommend.",
            "I hate this, it's terrible and a complete waste of my money.",
            "The movie was just okay, nothing special, very average.",
            "Great customer service, they were so helpful and quick to respond.",
            "Absolutely horrible experience, will never buy from them again, worst ever.",
            "It's good, but could definitely be better with some improvements.",
            "The food was delicious and the ambiance was perfect for a date.",
            "Disappointing performance by the team, they need to improve significantly.",
            "Neutral comment here. Just stating facts, no strong feelings.",
            "Happy with the purchase! It arrived on time and works as expected."
        ],
        'sentiment': [
            'positive', 'negative', 'neutral', 'positive', 'negative',
            'neutral', 'positive', 'negative', 'neutral', 'positive'
        ]
    }
    df = pd.DataFrame(data)
    # Ensure column names match configuration
    df.rename(columns={'text': TEXT_COLUMN, 'sentiment': SENTIMENT_COLUMN}, inplace=True)

print("\nOriginal Dataset Head:")
print(df.head())
print("\nDataset Info:")
df.info()
print(f"\n{SENTIMENT_COLUMN} Distribution:")
print(df[SENTIMENT_COLUMN].value_counts())

# Add text length for EDA
df['text_length'] = df[TEXT_COLUMN].apply(len)


# --- 3. Exploratory Data Analysis (EDA) ---
print("\n--- Performing EDA ---")
plt.figure(figsize=(8, 6))
sns.countplot(x=SENTIMENT_COLUMN, data=df, palette='viridis')
plt.title(f'Distribution of {SENTIMENT_COLUMN.capitalize()}')
plt.xlabel(SENTIMENT_COLUMN.capitalize())
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['text_length'], bins=50, kde=True)
plt.title('Distribution of Text Lengths')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.show()

# --- 4. Data Preprocessing ---
print("\n--- Preprocessing Text Data ---")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Convert to lowercase
    text = str(text).lower() # Ensure text is string
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions (@) and hashtags (#)
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stop words and lemmatize
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(processed_tokens)

df['cleaned_text'] = df[TEXT_COLUMN].apply(preprocess_text)

print("\nOriginal vs. Cleaned Text Examples:")
print(df[[TEXT_COLUMN, 'cleaned_text']].head())

# Word clouds for cleaned text
print("\n--- Generating Word Clouds ---")
# Make sure sentiment column contains 'positive' and 'negative'
positive_texts = ' '.join(df[df[SENTIMENT_COLUMN] == 'positive']['cleaned_text'])
negative_texts = ' '.join(df[df[SENTIMENT_COLUMN] == 'negative']['cleaned_text'])

if positive_texts:
    wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_texts)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_positive, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Positive Sentiment')
    plt.show()
else:
    print("No positive sentiment texts found for word cloud.")

if negative_texts:
    wordcloud_negative = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(negative_texts)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_negative, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Negative Sentiment')
    plt.show()
else:
    print("No negative sentiment texts found for word cloud.")


# --- 5. Feature Extraction ---
print("\n--- Performing TF-IDF Feature Extraction ---")
X = df['cleaned_text']
y = df[SENTIMENT_COLUMN]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

tfidf_vectorizer = TfidfVectorizer(max_features=5000) # You can adjust max_features
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"TF-IDF Vectorized Training Data Shape: {X_train_tfidf.shape}")
print(f"TF-IDF Vectorized Testing Data Shape: {X_test_tfidf.shape}")


# --- 6. Model Training (Logistic Regression) ---
print("\n--- Training Logistic Regression Model ---")
model = LogisticRegression(max_iter=1000, solver='liblinear') # 'liblinear' often good for smaller datasets
model.fit(X_train_tfidf, y_train)
print("Model training complete.")


# --- 7. Model Evaluation ---
print("\n--- Evaluating Model Performance ---")
y_pred = model.predict(X_test_tfidf)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# --- 8. Prediction on New Text ---
print("\n--- Predicting Sentiment on New Text Examples ---")

def predict_sentiment(text, vectorizer, trained_model):
    """
    Predicts the sentiment of a given text using the trained model.
    """
    cleaned_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = trained_model.predict(vectorized_text)
    return prediction[0]

new_texts_to_predict = [
    "This software update is fantastic! It fixed all the bugs and runs smoothly.",
    "The customer support was terrible; I waited forever and they weren't helpful.",
    "It's an interesting concept, but the execution needs some work.",
    "Just received my order, everything seems fine.",
    "Worst pizza I've ever had, absolutely disgusting."
]

for text in new_texts_to_predict:
    sentiment = predict_sentiment(text, tfidf_vectorizer, model)
    print(f"Text: '{text}'")
    print(f"Predicted Sentiment: {sentiment}\n")

print("\n--- Sentiment Analysis Script Complete ---")
