# Import the librarie's

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re

# Import gensim for preprocessing
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, \
    strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short
# Import the Word2Vec
from gensim.models import Word2Vec

# Import scikit learn
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#-------------------------
# Import the Kmeans function from file
#-------------------------
from fakeNewsModels.KMeans.myKmeans import _KMeans_

#-------------------------
# EDA and cleanning
#-------------------------

print('Loading the dataset\'s')
fake = pd.read_csv('data/Fake.csv')
real = pd.read_csv('data/Real.csv')
print('Done\n')
input('Press enter to continue...\n')

# Print some lines from the dataset
print('Fake news dataset...\n')
print(fake.head(5))
print('\nReal news dataset...\n')
print(real.head(5))
input('\nPress enter to continue...\n')

# Removing some @ from the text

# Merging the cols: Title & Text
fake['Sentences'] = fake['title'] + ' ' + fake['text']
real['Sentences'] = real['title'] + ' ' + real['text']

# Add some extra col to the dataset's
fake['Label'] = 0
real['Label'] = 1

# Concat both dataframes
data = pd.concat([fake, real])

# Delete some cols. 
data = data.drop(['title', 'text', 'subject', 'date'], axis=1)

print('Final data...\n')
print(data.head(5))
input('\nPress enter to continue...\n')

#-------------------------
# Preprocess the sentences
#-------------------------

def removingUrl(sentence):
    regex = re.compile(r'https?://\S+|www\.\S+|bit\.ly\S+')
    return regex.sub(r'', sentence)

# Preprocessing functions to remove: 
# lowercase, 
# links, 
# whitespace, 
# tags, 
# numbers, 
# punctuation, 
# strip word

customFilter = [lambda x: x.lower(), strip_tags, removingUrl, strip_punctuation, \
    strip_multiple_whitespaces, strip_numeric, remove_stopwords, strip_short]

# Variables to store data
processedData = []
processedLabel = []

for index, row in data.iterrows():
    words = preprocess_string(row['Sentences'], customFilter)
    if len(words) > 0:
        processedData.append(words)
        processedLabel.append(row['Label'])

#-------------------------
# Word2Vec
#-------------------------

model2vec = Word2Vec(processedData, min_count=1)
print('Show Word 2 vec by Country...\n')
print(model2vec.wv.most_similar('country'))
input('\nPress enter to continue...\n')

#-------------------------
# Sentence Vectors
#-------------------------

def returnVector(x):
    try:
        return model2vec[x]
    except:
        return np.zeros(100)

def sentenceVector(sentence):
    wordVectors = list(map(lambda x: returnVector(x), sentence))
    return np.average(wordVectors, axis=0).tolist()

X = []
for dataX in processedData:
    X.append(sentenceVector(dataX))

xNp = np.array(X)
print('Shape of the new dataset: {}'.format(xNp.shape))
input('\nPress enter to continue...\n')

#-------------------------
# Using the KMeans
#-------------------------

# Train the _KMeans_ for 2 clusters (fake or real)
result, clusters = _KMeans_(xNp, 2, is_kmeans=False, is_random=False)

# Testing
testingDf = {'Sentence': processedData, 'Labels': processedLabel, 'Prediction': clusters}
testingDf = pd.DataFrame(data=testingDf)

testingDf.head(10)

correct = 0
incorrect = 0
for index, row in testingDf.iterrows():
    if row['Labels'] == row['Prediction']:
        correct += 1
    else:
        incorrect += 1
        
print("Correctly clustered news: " + str((correct*100)/(correct+incorrect)) + "%")

#-------------------------
# Visualization
#-------------------------

# PCA
pca = PCA(n_components=2)
pcaResult = pca.fit_transform(xNp)

PCAdf = pd.DataFrame(pcaResult)
PCAdf['cluster'] = clusters
PCAdf.columns = ['x1','x2','cluster']

# T-SNE
tsne = TSNE(n_components=2)
tsneResult = tsne.fit_transform(pcaResult)

TSNEdf = pd.DataFrame(tsneResult)
TSNEdf['cluster'] = clusters
TSNEdf.columns = ['x1','x2','cluster']


# Plot
fig, ax = plt.subplots(1, 2, figsize=(12,6))
sns.scatterplot(data=PCAdf, x='x1', y='x2', hue='cluster', legend="full", alpha=0.5, ax=ax[1])
sns.scatterplot(data=TSNEdf, x='x1', y='x2', hue='cluster', legend="full", alpha=0.5, ax=ax[0])
ax[0].set_title('Visualized on TSNE')
ax[1].set_title('Visualized on PCA')

