# imports
import pandas as pd
import nltk
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords, words
from matplotlib import pyplot as plt
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
from keras import models, layers
from collections import Counter

# Function for build the one hot matrix
def build_corpus(tweets):
    # corpora downloads for nltk
    nltk.download('stopwords')
    # dictionary size
    dictionarySize = 7000
    # tokenize and get frequency
    topUniqueWordsFiltered = []
    tok = Tokenizer(filters='!"#$%&()*+,-./:;<=>?[\\]^_`{|}~\t\n')
    tok.fit_on_texts(tweets)
    # print(tok.__dict__.items())
    topUniqueWords = sorted(tok.word_counts.items(), key=lambda x: x[1], reverse=True)
    # print(topUniqueWords)
    for word,_ in topUniqueWords:
          if len(word)> 3 and ('@') not in word and 'http' not in word and word not in stopwords.words('spanish') and not word in topUniqueWordsFiltered:
                topUniqueWordsFiltered.append(word)
    return topUniqueWordsFiltered[:dictionarySize]

# Function to represent tweets as numerical vectors considering the corpus of TASS and candidate datsets as reference
def buildWordVectorMatrix(tweetsVect, corpusW):
    # empty numpy matrix of necesary size
    wordVectorMatrix = np.zeros((len(tweetsVect), len(corpusW)))
    # fill matrix, with binary representation of words
    for pos, tweetInPos in enumerate (tweetsVect):
        # split each tweet into a list of its words
        tweetWords = tweetInPos.lower().split()
        # assign value of 1 in matrix position corresponding to the current tweet
        # and the id, each of its contained words is located on the previously built dictionary
        for word in tweetWords:
          # only consider words that are part of the built dictionary
          if word in corpusDictionary:
              wordVectorMatrix[pos, corpusDictionary.index(word)] = 1
    return wordVectorMatrix

# Load joined TASS dataset
tassDf = pd.read_csv("Datasets/ALL_TassDF.csv", encoding='utf8').reset_index(drop=True)[['Text', 'Tag']]

# Select tweets with tag of positive, neative or neutral
tassDf = tassDf.loc[(tassDf.Tag == 'P') | (tassDf.Tag == 'N') | (tassDf.Tag == 'NEU')]

# Verify the final dataset - 57454 tweets
print(tassDf, '\n\n', tassDf.shape)
print(tassDf.columns.values)
print(tassDf.values)

# Load replies to tweets from two Ecuadorian presidential candidates
candidDf = pd.read_csv( 'Datasets/ALL_candidates.csv', encoding='utf8')

# Merge TASS and canidiate datsets to create the corpus
joinedDfTexts = candidDf['text'].append(tassDf['Text'], ignore_index=True)  # continuous idxs
print(joinedDfTexts, '\n\n', joinedDfTexts.shape)
print(joinedDfTexts.values)

# Build the one hot matrix considering TASS and candidate datasets
corpusDictionary = build_corpus(joinedDfTexts)
print (len(corpusDictionary), corpusDictionary)

# Observe Tass dataset balance, to see if we apply any method for balancing
# histogram to see dataset balance
plt.hist(tassDf['Tag'])

# Transform target names (P, N, NEU) of TASS dataset into integer representation
tassDf.loc[tassDf['Tag'] == 'N', 'Tag'] = 0
tassDf.loc[tassDf['Tag'] == 'NEU', 'Tag'] = 1
tassDf.loc[tassDf['Tag'] == 'P', 'Tag'] = 2

# Visualize transformation to numerical classes
plt.hist(tassDf['Tag'])

# Reprresent TASS dataset as vectors considering the corpus created before
X_data = np.array(buildWordVectorMatrix(tassDf['Text'], corpusDictionary))

#  Select the y (target) of all data
y_data = np.array(tassDf['Tag'])

# Divide TASS dataset into training (70%), validation (15%) and test (15%)
# Training and validation-test
X_train, X_other, y_train, y_other = train_test_split(X_data, y_data, test_size=0.50, random_state=1, stratify=y_data)

# divide into validation and test datasets
X_validation, X_test, y_validation, y_test = train_test_split(X_other, y_other, test_size=0.50, random_state=1, stratify=y_other)

# Convert tags to binnary representation using to categorical function from Keras
y_train_bin = to_categorical(y_train, num_classes=3)
y_test_bin = to_categorical(y_test, num_classes=3)
y_validation_bin = to_categorical(y_validation, num_classes=3)

# Create the neuronal network model with keras Sequential class and add layer with given activation functions
model = models.Sequential()
model.add(layers.Dense(1200, activation='relu', input_shape=(len(corpusDictionary),)))
model.add(layers.Dense(300, activation='relu'))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

# Associate the metrics to the model
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['acc', 'AUC'])

# Run the model with processed data and selected metrics
print('xTest test', X_train.sum(axis=1))
print('X_test', X_train, X_train.shape)
print('y_train_bin', y_train_bin)
train_log = model.fit(X_train, y_train_bin,
                     epochs=10, batch_size=512,
                     validation_data=(X_validation, y_validation_bin))

# Model evaluation considering the accuracy of training and validations datasets
acc = train_log.history['acc']
val_acc = train_log.history['val_acc']
loss = train_log.history['loss']
val_loss = train_log.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# Calculate the accuracy considering test dataset
test_accuracy = model.evaluate(X_test, y_test_bin)
print(test_accuracy)

# Import data from replies to tweets of ecuadorian candidates
lassoPath = 'Datasets/replies_random1000_lasso_Diciembre.csv'
arauzPath = 'Datasets/replies_random1000_arauz_Diciembre.csv'
lassoDf = pd.read_csv(lassoPath, encoding='utf8').reset_index(drop=True)
arauzDf = pd.read_csv(arauzPath, encoding='utf8').reset_index(drop=True)


# Transform candidate tweets into numercial values by hot encoding
X_lasso = np.array(buildWordVectorMatrix(lassoDf['text'], corpusDictionary))
X_arauz = np.array(buildWordVectorMatrix(arauzDf['text'], corpusDictionary))
# Predict sentiment analysis of candidate tweets using outr trained model
# predict classes function will be deprecated for multi class distribution use (model.predict(X_lasso) > 0.5).astype("int32") in the future
# or numpy argmax, could be good option
classResultsLasso = model.predict_classes(X_lasso)
lassoPredDistribution = Counter(classResultsLasso).most_common()
print(lassoPredDistribution)

classResultsArauz = model.predict_classes(X_arauz) # predict classes function will be deprecated
arauzPredDistribution = Counter(classResultsArauz).most_common()
print(arauzPredDistribution)

# Candidate results
# (0 Negative, 1 neutral, 2 postive)
print('\nReplies to Lasso tweets results: ')
print(f'NEGATIVES: {lassoPredDistribution[0][1]}')
print(f'POSITIVES: {lassoPredDistribution[1][1]}')
print(f'NEUTRAL: {lassoPredDistribution[2][1]}')
print('\nReplies to Arauz tweets results:')
print(f'NEGATIVES: {arauzPredDistribution[0][1]}')
print(f'POSITIVES: {arauzPredDistribution[1][1]}')
print(f'NEUTRAL: {arauzPredDistribution[2][1]}')

# Pie charts
# Lasso
valuesLasso = [freq for word, freq in lassoPredDistribution]
labelsLasso = ['NEGATIVE', 'POSITIVE', 'NEUTRAL']
plt.figure('Guillermo Lasso')
figLasso1, axLasso1 = plt.subplots()
axLasso1.set_title('Guillermo Laso')
axLasso1.pie(valuesLasso, labels=labelsLasso, autopct='%1.1f%%', shadow=True, startangle=90)
plt.show()

# Arauz
valuesArauz = [freq for word, freq in arauzPredDistribution]
labelsArauz= ['NEGATIVE', 'POSITIVE', 'NEUTRAL']
plt.figure('Andres Arauz')
figArauz1, axArauz1 = plt.subplots()
axArauz1.set_title('Andres Arauz')
axArauz1.pie(valuesArauz, labels=labelsArauz, autopct='%1.1f%%', shadow=True, startangle=90)
plt.show()
