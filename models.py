# %%
import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import (
    BernoulliNB,
    MultinomialNB,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

# %%
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')
nltk.download('stopwords')

# %%
pidgin_sentiment_df = pd.read_csv('data/processed/PidginSenti.csv')
resampled_pidgin_sentiment_df = pd.read_csv('data/processed/ResampledPidginSenti.csv')

# %%
sia = SentimentIntensityAnalyzer()
sia.polarity_scores('how you dey')

# %%
features = []
stop_words = nltk.corpus.stopwords.words("english")
for text in tqdm(resampled_pidgin_sentiment_df['text']):
    t = ' '.join([word for word in text.split() if word not in stop_words])
    text_length = len(t)
    polarities = sia.polarity_scores(t)
    feature = polarities
    feature['text_length'] = text_length
    features.append(feature)

features[0]

# %%
target = []
for s in resampled_pidgin_sentiment_df['sentiment']:
    if s == "Positive":
        target.append("pos")
    elif s == "Neutral":
        target.append("neu")
    elif s == "Negative":
        target.append("neg")

X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.25,
    random_state=42
)
# %%
train_dataset = [(features, sentiment) for features, sentiment in zip(X_train, y_train)]

# %%
classifiers = {
    "BernoulliNB": BernoulliNB(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(),
    "MLPClassifier": MLPClassifier(max_iter=10000),
    "AdaBoostClassifier": AdaBoostClassifier(),
}
results = {}

def compute_metrics(classifier):
    pred = [classifier.classify(features) for features in X_test]
    return classification_report(y_test, pred, target_names=["pos", "neu", "neg"], output_dict=True)

for name, sklearn_classifier in tqdm(classifiers.items()):
    classifier = nltk.classify.SklearnClassifier(sklearn_classifier)
    classifier = classifier.train(train_dataset)
    results[name] = compute_metrics(classifier)
    
# %%
for name, report in results.items():
    print(f'{name} - {report["accuracy"]}') 
