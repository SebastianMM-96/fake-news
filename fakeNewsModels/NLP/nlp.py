# Import the libraries
from os.path import split
import pandas as pd
import numpy as np 
import re
import nltk
import matplotlib.pyplot as plt
from pandas.core.dtypes.missing import isnull
from sklearn import pipeline
from wordcloud import WordCloud
from wordcloud.wordcloud import STOPWORDS


#-------------------
# Import the dataset's
#-------------------

print('Loading dataset\'s')
trainDf = pd.read_csv('data/train.csv')
testDf = pd.read_csv('data/test.csv')
print('Done\n')
input('\nPress enter to continue...\n')

#-------------------
# Shape of the dataset's
#-------------------

print('\nDataset\'s shape')
print('Train: {}\n'.format(trainDf))
print('\nTest: {}\n'.format(testDf))
input('\nPress enter to continue...\n')

#-------------------
# Info about the dataset's
#-------------------

print('Loading info...\n')
def missingValues():
    isNullTrain = trainDf.isnull().sum()
    isNullTest = testDf.isnull().sum()
    print('Train info: {}'.format(isNullTrain))
    print('\nTest info: {}'.format(isNullTest))

# Call the function
missingValues()
input('\nPress enter to continue...\n')

#-------------------
# Fill the NaN values inside the
# dataset's
#-------------------

test = testDf.fillna(' ')
train = trainDf.fillna(' ')

test['total'] = test['title'] + ' ' + test['author'] + ' ' + test['text']
train['total'] = train['title'] + ' ' + train['author'] + ' ' + train['text']


#-------------------
# WordCloud
#-------------------

realWords = ''
fakeWords = ''
stopwords = set(STOPWORDS)

for aux in train[train['label'] == 1].total:
    # split
    token = aux.split()

    # to lower case
    for i in range(len(token)):
        token[i] = token[i].lower()
    realWords += ' '.join(token) + ' '

for aux_ in train[train['label'] == 0].total:
    # split
    token_ = aux_.split()

    # to lower case
    for i in range(len(token_)):
        token_[i] = token_[i].lower()
    fakeWords += ' '.join(token_) + ' '

# Call the wordCloud for Real Words
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(realWords)

# Plot the Real Words
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.title('Real Word\'s')
plt.tight_layout(pad = 0) 
plt.show()

# Call the wordCloud for Fake Words
wordcloud_ = WordCloud(width = 800, height = 800, 
                background_color ='black', 
                stopwords = stopwords, 
                min_font_size = 10).generate(fakeWords)

# Plot the Fake Words
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud_)
plt.axis("off")
plt.title('Fake Word\'s')
plt.tight_layout(pad = 0) 
plt.show()

#-------------------
# Cleaning & Preprocessing
#-------------------

# Regex
## Removing punctuations
print('Removing punctuation from a string...')
punctuations = "!</> hello please$$ </>^v!!!i%%s&&%$i@@@t^^^&&!& </>*my@# git&&\hub@@@##%^^& repo!@# %%$"
punctuations = re.sub(r'[^\w\s]', '', punctuations)
print('\nRemoved punctuations in string: {}'.format(punctuations))
input('\nPress enter to continue...\n')


#-------------------
# Tokenization
#-------------------

# Download data from nltk
print('Download nltk data...')
nltk.download('punkt')
nltk.download('wordnet')

# Test the data
print('\nTesting...\n')
nltk.word_tokenize("Hello how are you")
input('\nPress enter to continue...\n')

#-------------------
# Stop Words
#-------------------

from nltk.corpus import stopwords

print('Loading stopword\'s from english language...')
stopwords_ = stopwords.words('english')
print('Done')
input('\nPress enter to continue...\n')

#-------------------
# Lemmatization
#-------------------

from nltk.stem import WordNetLemmatizer

print('Loading the lemmatization module...')
lemmatizer = WordNetLemmatizer()
print('Done')
input('\nPress enter to continue...\n')

#-------------------
# Apply Lemmatization and StopWords
#-------------------

for idx, row in train.iterrows():
    filter = ''
    sentence = row['total']
    # Cleaning
    sentence = re.sub(r'[^\w\s]', '', sentence)
    # Lemmatization
    words = nltk.word_tokenize(sentence)
    # Remove the stopwords
    words = [aux_ for aux_ in words if not aux_ in stopwords_]

    for word in words:
        filter = filter + ' ' + str(lemmatizer.lemmatize(word)).lower()
    
    train.loc[idx, 'total'] = filter

train = train[['total', 'label']]

#-------------------
# Apply NLP
#-------------------

# Import's
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

X_train = train['total']
Y_train = train['label']

# Feature extraction using count vectorization and tfidf
countVectorizer = CountVectorizer()
countVectorizer.fit_transform(X_train)
freqTermMatrix = countVectorizer.transform(X_train)

tfidf = TfidfTransformer(norm="l2")
tfidf.fit(freqTermMatrix)
tfidfMatrix = tfidf.transform(freqTermMatrix)

# Show the matrix
print('Tf-idf matrix\n')
print(tfidfMatrix)
input('\nPress enter to continue...\n')

#-------------------
# Pipieline
#-------------------

from sklearn.pipeline import Pipeline
from sklearn import linear_model


pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer(norm='l2')),
    ('clf', linear_model.LogisticRegression()),
])

pipeline.fit_transform(X_train, Y_train)

#-------------------
# Test a prediction
#-------------------

print('Test the pipeline prediction\n')
textNews = input('Enter the text of a news: ')

pipeline.predict([textNews])