import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

import nltk
import string
import re
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

# ---------------------- Enhanced Preprocessing Function ----------------------

def preprocess(text):
    """Enhanced preprocessing with better text cleaning"""
    text = str(text).lower()
    
    # Remove URLs, mentions, hashtags
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#\w+', '', text)
    
    # Handle contractions
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'d", " would", text)
    
    # Remove excessive punctuation but keep some for context
    text = re.sub(r'[!]{2,}', ' EXCLAMATION ', text)
    text = re.sub(r'[?]{2,}', ' QUESTION ', text)
    text = re.sub(r'[.]{3,}', ' DOTS ', text)
    
    # Remove extra whitespace and special characters
    text = re.sub(r'\s+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove stopwords but keep negation words
    negation_words = {'not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither', 'nor'}
    words = text.split()
    words = [word for word in words if word not in stop_words or word in negation_words]
    
    return ' '.join(words)

def extract_features(text):
    """Extract additional features from text"""
    features = {}
    
    # Basic text statistics
    features['length'] = len(text)
    features['word_count'] = len(text.split())
    # features['char_count'] = len(text)
    features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
    
    # Punctuation features
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    
    # Sentiment features using VADER
    try:
        sentiment_scores = sia.polarity_scores(text)
        features['sentiment_pos'] = sentiment_scores['pos']
        features['sentiment_neg'] = sentiment_scores['neg']
        features['sentiment_neu'] = sentiment_scores['neu']
        features['sentiment_compound'] = sentiment_scores['compound']
    except:
        features['sentiment_pos'] = 0
        features['sentiment_neg'] = 0
        features['sentiment_neu'] = 0
        features['sentiment_compound'] = 0
    
    # Offensive word patterns (basic indicators)
    offensive_patterns = ['hate', 'stupid', 'idiot', 'kill', 'die', 'ugly', 'fat', 'loser', 
                          'offensive', 'abuse', 'harass', 'bully', 'threat', 'slur', 'racist', 
                          'racism', 'feminazi', 'sexist', 'misogynist', 'homophobic', 'transphobic', 
                          'transphobia', 'islamophobic', 'xenophobic', 'ableist', 'bodyshaming', 
                          'cyberbully', 'cyberbullying']
    features['offensive_word_count'] = sum(1 for word in offensive_patterns if word in text.lower())
    
    return features

# ---------------------- Load and Prepare Dataset ----------------------

df = pd.read_csv("cyberbullying_tweets.csv")
df.columns = ['text', 'label']

# Dataset overview
print("Dataset Shape:", df.shape)

print("\nLabel Distribution:")
print(df['label'].value_counts())

# Label Distribution Pie Chart
plt.figure(figsize=(8, 8))

df['label'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    startangle=90
)

plt.title("Label Distribution of Cyberbullying Dataset")
plt.ylabel("")
plt.show()
plt.close()

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove rows with missing values
df.dropna(inplace=True)


# Enhanced preprocessing
df['cleaned_text'] = df['text'].apply(preprocess)
df = df[df['cleaned_text'].str.strip() != '']

# Extract additional features
feature_data = []
for text in df['text']:
    feature_data.append(extract_features(text))

feature_df = pd.DataFrame(feature_data)
df = pd.concat([df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)

# Convert multiclass labels into binary labels
# 0 = Not Cyberbullying
# 1 = Cyberbullying
df['binary_label'] = df['label'].apply(lambda x: 0 if x == 'not_cyberbullying' else 1)

# Enhanced balancing with stratification
from sklearn.utils import resample

# Separate classes
cyber_df = df[df['binary_label'] == 1]
not_cyber_df = df[df['binary_label'] == 0]

# Balance dataset with better sampling

# Find the size of the smaller class
# This helps prevent one class from dominating the dataset
min_len = min(len(cyber_df), len(not_cyber_df))

# Set a maximum limit for samples to reduce training time
# and avoid excessive memory usage
max_samples = min(min_len * 2, 10000)

# If the cyberbullying class has too many samples,
# randomly select only max_samples records
if len(cyber_df) > max_samples:
    cyber_df = resample(
        cyber_df,
        n_samples=max_samples,
        random_state=42
    )

print("\nCyberbullying Class Distribution After Sampling:")
print(cyber_df['label'].value_counts())

# Pie Chart for Cyberbullying Categories After Sampling

cyber_counts = cyber_df['label'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(
    cyber_counts,
    labels=cyber_counts.index,
    autopct='%1.1f%%',
    startangle=90
)

plt.title('Cyberbullying Categories After Sampling')
plt.axis('equal')
plt.show()
plt.close()

# If the non-cyberbullying class has too many samples,
# randomly select only max_samples records
if len(not_cyber_df) > max_samples:
    not_cyber_df = resample(
        not_cyber_df,
        n_samples=max_samples,
        random_state=42
    )

# Combine both classes into a single balanced dataset
df_balanced = pd.concat([cyber_df, not_cyber_df])

# Shuffle the dataset to mix both classes randomly
# Reset index to create a clean sequential index
df_balanced = df_balanced.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print(f"Dataset size after balancing: {len(df_balanced)}")
print(f"Cyberbullying samples: {sum(df_balanced['binary_label'])}")
print(f"Non-cyberbullying samples: {len(df_balanced) - sum(df_balanced['binary_label'])}")


# ---------------------- Dataset Distribution Visualization ----------------------

plt.figure(figsize=(6,4))

sns.countplot(
    x='binary_label',
    data=df_balanced
)

plt.title("Class Distribution After Balancing")
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.xticks([0,1], ['Not Cyberbullying', 'Cyberbullying'])

plt.show()
plt.close()


# ---------------------- Enhanced Feature Engineering ----------------------

# TF-IDF with optimized parameters
tfidf_vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 3),  # Include bigrams and trigrams
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents='unicode'
)

# Fit TF-IDF on cleaned text
X_text = tfidf_vectorizer.fit_transform(df_balanced['cleaned_text'])

# Display TF-IDF matrix size
print("\nTF-IDF Matrix Shape:", X_text.shape)
print("Number of TF-IDF Features:", X_text.shape[1])

# Additional numerical features
feature_columns = ['length', 'word_count', 'avg_word_length', 
                  'exclamation_count', 'question_count', 'caps_ratio',
                  'sentiment_pos', 'sentiment_neg', 'sentiment_neu', 
                  'sentiment_compound', 'offensive_word_count']

X_features = df_balanced[feature_columns].fillna(0)

# Scale features to [0,1] range for MultinomialNB compatibility
scaler = MinMaxScaler()
X_features_scaled = scaler.fit_transform(X_features)

# Combine TF-IDF and scaled additional features
from scipy.sparse import hstack, csr_matrix
X_features_sparse = csr_matrix(X_features_scaled)
X_combined = hstack([X_text, X_features_sparse])

y = df_balanced['binary_label']
# Final dataset summary
print("\nFinal Dataset Summary")
print("Feature Matrix Shape:", X_combined.shape)
print("Total Features:", X_combined.shape[1])
print("Total Samples:", X_combined.shape[0])

print("\nTF-IDF Feature Size:", X_text.shape)

# ---------------------- Train / Validation / Test Split ----------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X_combined,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nDataset Split:")
print("Training Set:", X_train.shape[0])
print("Validation Set:", X_val.shape[0])
print("Test Set:", X_test.shape[0])

plt.figure(figsize=(7,7))

sizes = [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
labels = ['Training', 'Validation', 'Test']

plt.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    startangle=90
)

plt.title('Train / Validation / Test Split')
plt.axis('equal')
plt.show()
plt.close()

# ---------------------- Enhanced Model with Ensemble ----------------------

# Define individual models
svm_model = LinearSVC(C=1.0, random_state=42, max_iter=2000)
lr_model = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
nb_model = MultinomialNB(alpha=0.1)  # Now works with scaled features
knn_model = KNeighborsClassifier(n_neighbors=5)

# Create ensemble model
ensemble_model = VotingClassifier(
    estimators=[
        ('svm', svm_model),
        ('lr', lr_model),
        ('rf', rf_model),
        ('nb', nb_model),
        ('knn', knn_model)
    ],
    voting='hard'  # Use hard voting for binary classification
)

# Train the ensemble model
print("Training ensemble model...")
ensemble_model.fit(X_train, y_train)

# Validation Accuracy
y_val_ensemble = ensemble_model.predict(X_val)
ensemble_val_acc = accuracy_score(y_val, y_val_ensemble)

print(f"\nEnsemble Validation Accuracy: {ensemble_val_acc * 100:.2f}%")

# Test Accuracy
y_pred_ensemble = ensemble_model.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)

# Evaluate ensemble model
print(f"\nEnsemble Test Accuracy: {ensemble_accuracy * 100:.2f}%")

print("\nEnsemble Classification Report:")
print(classification_report(
    y_test,
    y_pred_ensemble,
    target_names=['Not Cyberbullying', 'Cyberbullying']
))

# Convert report to dictionary
report = classification_report(
    y_test,
    y_pred_ensemble,
    target_names=['Not Cyberbullying', 'Cyberbullying'],
    output_dict=True
)

# Convert to DataFrame
df = pd.DataFrame(report).transpose()

# Select only class metrics
plot_df = df.loc[
    ['Not Cyberbullying', 'Cyberbullying'],
    ['precision', 'recall', 'f1-score']
]

# Plot
plot_df.plot(kind='bar', figsize=(8,5))

plt.title("Ensemble Classification Report")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(loc='lower right')

plt.show()
plt.close()

# Also train individual models for comparison

models = {'SVM': svm_model,
          'Logistic Regression': lr_model,
          'Random Forest': rf_model,
          'Naive Bayes': nb_model,
          'KNN': knn_model}

individual_accuracies = {}
validation_accuracies = {}

for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)

    # Validation Accuracy
    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)

    # Test Accuracy
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    validation_accuracies[name] = val_acc
    individual_accuracies[name] = test_acc

    print(f"\n{name}")
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Use the best performing model (ensemble or individual)
best_model = ensemble_model
best_accuracy = ensemble_accuracy

# Cross-validation for more robust evaluation
cv_scores = cross_val_score(ensemble_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"\nCross-validation scores: {cv_scores}")
print(f"Average CV accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 2 * 100:.2f}%)")

individual_accuracies['Ensemble'] = ensemble_accuracy
validation_accuracies['Ensemble'] = ensemble_val_acc

plt.figure(figsize=(10, 6))

sns.barplot(
    x=list(individual_accuracies.keys()),
    y=[acc * 100 for acc in individual_accuracies.values()]
)

plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Test Accuracy (%)")
plt.show()
plt.close()

# ---------------------- Confusion Matrix ----------------------

cm = confusion_matrix(y_test, y_pred_ensemble)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Not Cyberbullying', 'Cyberbullying'],
    yticklabels=['Not Cyberbullying', 'Cyberbullying']
)

plt.title("Confusion Matrix - Ensemble Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
plt.close()


# ---------------------- LIVE PREDICTION SYSTEM ----------------------

def predict_cyberbullying(text):

    # Step 1: preprocess text
    cleaned_text = preprocess(text)

    # Step 2: extract features
    features = extract_features(text)
    feature_df = pd.DataFrame([features])

    # Step 3: TF-IDF transform
    text_tfidf = tfidf_vectorizer.transform([cleaned_text])

    # Step 4: scale numerical features
    features_scaled = scaler.transform(feature_df)

    # Step 5: combine features
    from scipy.sparse import hstack, csr_matrix
    features_sparse = csr_matrix(features_scaled)
    combined_input = hstack([text_tfidf, features_sparse])

    # Step 6: prediction
    prediction = ensemble_model.predict(combined_input)[0]

    # Step 7: output label
    if prediction == 1:
        return "🚨 Cyberbullying Detected"
    else:
        return "✅ Not Cyberbullying"


# ---------------------- TESTING INPUT ----------------------

print("\n--- LIVE CYBERBULLYING DETECTOR ---")
user_input = input("Enter a sentence: ")

result = predict_cyberbullying(user_input)
print("Result:", result)