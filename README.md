# Cyberbullying Detection Using a Hard-Voting Ensemble

A machine learning project that classifies social media text as cyberbullying or non-cyberbullying using a hard-voting ensemble of five traditional machine learning classifiers.

## Project Overview

The system combines TF-IDF text features with linguistic, sentiment, and offensive-word indicators.

### Models

- Linear Support Vector Machine (LinearSVC)
- Logistic Regression
- Random Forest
- Multinomial Naive Bayes
- K-Nearest Neighbors (KNN)

The final ensemble uses majority voting to combine predictions.

## Dataset

The project uses the [Cyberbullying Tweets dataset on Kaggle](https://www.kaggle.com/datasets/pradeepjswl/cyberbullying-tweets/data). The dataset contains tweets from six original categories, which are converted into two classes:

- `0` — Not Cyberbullying
- `1` — Cyberbullying

Download the dataset from Kaggle and place the CSV file named `cyberbullying_tweets.csv` in the project root before running the script. The dataset is not included in this repository by default.

## Feature Engineering

The implementation combines:

- 8,000 TF-IDF features using unigrams, bigrams, and trigrams
- 6 linguistic features
- 4 VADER sentiment features
- 1 offensive-word indicator

## Reported Results

According to the accompanying technical report:

- Test accuracy: 83.08%
- Cyberbullying-class precision: 0.88
- Cyberbullying-class recall: 0.81
- Cyberbullying-class F1-score: 0.84
- Mean five-fold cross-validation accuracy: 82.75%

These are the reported project results and have not been independently reproduced.

## Installation

Requirements: Python and the dependencies listed in `requirements.txt`.

Clone the repository:

```bash
git clone https://github.com/FahimAhmed121/cyberbullying-detection-hard-voting-ensemble.git
cd cyberbullying-detection-hard-voting-ensemble
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Place `cyberbullying_tweets.csv` in the project root, then run:

```bash
python cyberbullying_detection.py
```

The script trains the models, displays evaluation results and visualizations, and prompts for text to classify.

## Repository Structure

```text
cyberbullying-detection-hard-voting-ensemble/
├── cyberbullying_detection.py
├── requirements.txt
├── README.md
├── .gitignore
└── docs/
    ├── Cyberbullying_Detection_Technical_Report.pdf
    └── Cyberbullying_Final_Presentation.pptx
```

## Documentation

- [Technical Report](docs/Cyberbullying_Detection_Technical_Report.pdf)
- [Final Presentation](docs/Cyberbullying_Final_Presentation.pptx)

## Limitations

The implementation uses an English Twitter dataset and traditional machine learning models. Sarcasm, indirect insults, and context-dependent abuse may be difficult to detect. Predictions should be treated as a classification aid rather than definitive judgments.

## Author

Md. Fahim Ahmed

United International University  
Department of Computer Science and Engineering

Course: Machine Learning (CSE 4889)