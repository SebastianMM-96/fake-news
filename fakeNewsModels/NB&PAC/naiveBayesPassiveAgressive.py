# Include libraries
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
from matplotlib import pyplot as plt
import itertools
from seaborn import palettes

# Import scikit learn's library
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Model's import
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import KFold

# Metric's
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# Pipeline module
from sklearn.pipeline import Pipeline

# ---------------------
# Dataset import
# ---------------------

print('Loading the dataset...')
df = pd.read_csv('data/fake_or_real_news.csv')
print('Done\n')
input("\nPress Enter to continue...\n")

# ---------------------
# Data overview
# Inspecting the pandas dataframe
# Print some lines from the dataframe
# ---------------------

print('Shape of dataframe: {}'.format(df.shape))
print(df.head(5))
input("\nPress Enter to continue...\n")

# ---------------------
# Plot distribution of classes for prediction
# ---------------------
print('Plotting...')
sns.countplot(x='label', data=df, palette='hls')
plt.show()

# ---------------------
# Missing values?
# ---------------------


def missingValues():
    print('\nVerify data qualities...')
    df.isnull().sum()
    df.info()
    print('Done\n')


# Call the function
missingValues()
input("Press Enter to continue...")


# ---------------------
# Get the labels from the dataset
# ---------------------

# Get access to the label column
y = df.label
print('Some labels collected from the dataframe: \n{}'.format(y.head()))
input("\nPress Enter to continue...\n")

# ---------------------
# Drop the label column from the dataset
# ---------------------
print('Dropping label column from the dataset...\n')
df.drop('label', axis=1)
print(df.head(5))
input("\nPress Enter to continue...\n")

# ---------------------
# Trainning and test sets from the dataset
# ---------------------

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], y, test_size=0.33, random_state=53)

print('Trainning and test datasets\n')
print('Train dataset: \n{}'.format(X_train))
print('\nTest dataset: \n{}'.format(X_test))
input("\nPress Enter to continue...\n")

# ---------------------
# Feature extraction
# ---------------------

countVectorizer = CountVectorizer(stop_words='english')

# Fit and transform the trainning data
countTrain = countVectorizer.fit_transform(X_train)

# Learn the vocabulary dictionary
# Return the term-document matrix
print(countVectorizer)
print(countTrain)
input("\nPress Enter to continue...\n")


def getCountVectorizerStats():
    # Size
    print('Vocabulary size: {}'.format(countTrain.shape))

    # Check the vocabulary
    print(countVectorizer.vocabulary_)


# Use function
getCountVectorizerStats()
input("\nPress Enter to continue...\n")

# ---------------------
# Transform the test set
# ---------------------
countTest = countVectorizer.transform(X_test)

# ---------------------
# Tf-idf frequency features
# Steps:
# a. initialize tf-idf vectorizer
# b. initialize the variable
# About:
# This removes the words wich appear in more than the 70%
# of the articles
# ---------------------

tfidfVectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the trainning set and transform the test set
tfidfTrain = tfidfVectorizer.fit_transform(X_train)


def getTfidfStats():
    print('Shape of Tf-idf using train set: {}\n'.format(tfidfTrain.shape))
    print('\nMatrix values: \n{}'.format(tfidfTrain.A[:10]))


# Use function
getTfidfStats()
input("\nPress Enter to continue...\n")

# Transform the test set
tfidfTest = tfidfVectorizer.transform(X_test)

# Get the features names of tfidfVectorizer
print('Features names: \n{}'.format(tfidfVectorizer.get_feature_names()[-10:]))
# Get the features names of the countVectorizer
print('\nFeatures names: \n{}'.format(
    countVectorizer.get_feature_names()[-10:]))

input("\nPress Enter to continue...\n")

countDf = pd.DataFrame(
    countTrain.A, columns=countVectorizer.get_feature_names())
tfidfDf = pd.DataFrame(
    tfidfTrain.A, columns=tfidfVectorizer.get_feature_names())
diff = set(countDf.columns) - set(tfidfDf.columns)

print(diff)
input("\nPress Enter to continue...\n")

# ---------------------
# Check whether the dataframes are equal
# ---------------------

print('Dataframes are equal?')
print(countDf.equals(tfidfDf))
input("\nPress Enter to continue...\n")

# ---------------------
# Confusion matrix plot function
# ---------------------
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#---------------------
# Pipeline classifier using naïve bayes | Tf-idf Vectorizer
#---------------------

nbPipeline = Pipeline([
    ('NBTV', tfidfVectorizer),
    ('nbClf', MultinomialNB())
])

# Fit the Naïve bayes classifier
print('Fit the Naïve Bayes using Tf-idf Vectorizer...\n')
nbPipeline.fit(X_train, y_train)
input("\nPress Enter to continue...\n")

#---------------------
# Perform classification
#---------------------
predictedNB = nbPipeline.predict(X_test)

# Show score
accuracy = metrics.accuracy_score(y_test, predictedNB)
print('Accuracy: {}%'.format(round(accuracy*100, 2)))
input("\nPress Enter to continue...\n")

# Plot confusion matrix
cm = metrics.confusion_matrix(y_test, predictedNB, labels=['fake', 'real'])
plot_confusion_matrix(cm, classes=['fake', 'real'])

#---------------------
# Pipeline classifier using naïve bayes | CountVectorizer
#---------------------

nbcvPipeline = Pipeline([
    ('NBCV', countVectorizer),
    ('nbClf', MultinomialNB())
])

# Fit the Naïve bayes classifier
print('Fit the Naïve Bayes using Countectorizer...\n')
nbcvPipeline.fit(X_train, y_train)
input("\nPress Enter to continue...\n")

#---------------------
# Perform classification
#---------------------
predictedNBCV = nbcvPipeline.predict(X_test)

# Show score
accuracy = metrics.accuracy_score(y_test, predictedNBCV)
print('Accuracy: {}%'.format(round(accuracy*100, 2)))
input("\nPress Enter to continue...\n")

# Plot confusion matrix
cm = metrics.confusion_matrix(y_test, predictedNBCV, labels=['fake', 'real'])
plot_confusion_matrix(cm, classes=['fake', 'real'])


#---------------------
# Classification report
#---------------------

print('Tf-idf Vectorizer\n')
print(metrics.classification_report(y_test, predictedNB))
input("\nPress Enter to continue...\n")

print('Count Vectorizer\n')
print(metrics.classification_report(y_test, predictedNBCV))
input("\nPress Enter to continue...\n")


#---------------------
# Pipeline classifier using 
# Passive Agressive Classifier | Tf-idf Vectorizer
#---------------------

# Initialize the Passive Agressive Classifier
linearClf = Pipeline([
    ('linear', tfidfVectorizer),
    ('paClf', PassiveAggressiveClassifier(max_iter = 50))
])

# Fit the classifier
print('Fit the Passive Agressive Classifier using Tf-idf Vectorizer...\n')
linearClf.fit(X_train, y_train)
input("\nPress Enter to continue...\n")

# Predict on the test set 
predictedPAC = linearClf.predict(X_test)

# Show score
accuracyPAC = metrics.accuracy_score(y_test, predictedPAC)
print('Accuracy: {}%'.format(round(accuracyPAC*100, 2)))
input("\nPress Enter to continue...\n")

# Plot confusion matrix
cm = metrics.confusion_matrix(y_test, predictedPAC, labels=['fake', 'real'])
plot_confusion_matrix(cm, classes=['fake', 'real'])

# Classification report
print('Passive agressive report\n')
print(metrics.classification_report(y_test, predictedPAC))
input("\nPress Enter to continue...\n")

#---------------------
# Saving the model
#---------------------
modelFile = 'model_01.sav'
pickle.dump(linearClf, open(modelFile, 'wb'))