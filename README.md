# Sentiment analysis of Ecuadorian political tweets from 2021 elections using Natural Language Processing

## Authors: Sebastián Ayala, David Mena, Sebastián Lucero 
### 21th December 2021

## About this project 
In this project, we captured sentiments of replies to tweets from two Ecuadorian presidential candidates in the 2021 elections (@LassoGuillermo and @ecuarauz). A model of neural networks was used to classify tweets considering their sentiment as positive, negative, or neutral. The training dataset was obtained from the    [Workshop on Semantic Analysis at SEPLN (TASS)](http://tass.sepln.org/) of 2020, 2019, and 2012. Thus, we joined together data of the three editions in one dataset that has information about the id, text, and sentiment associated with tweets. Due to privacy politcs we can't share the datasets here, but you can register in this [page]  (http://tass.sepln.org/2020/?page_id=74) and yo will access all datasets. 

## Data extraction 
We extracted all replies to tweets of both candidates between 01/12/2021 and 18/12/2021. For this task, we used the Python package [*Tweepy*](https://pypi.org/project/tweepy/) and the R package [*rtweet*](https://www.rdocumentation.org/packages/rtweet/versions/0.7.0). We obtained better results with *rtweet*, so this software was chosen for this task. Then, a sample of 1000 tweets for each candidate was selected. Both scripts in Python and R are available in the **Data_extraction** folder. We can't share tweets information because of privacy politics of tweeter development account, but you can access to this information applyin for tweeter development account. 

In addition, we obtained TASS datasets in xml and csv files. Then, we applied the script **Merge_TASS_data.py** to join all data in one file.

## Preprocessing, Feature extraction, Model and Results (PFMR)
All these processes were done using Python. We applied two preprocessing steps, tokenization and stop words deletion, using *Keras* and *nltk* tools. For the feature extraction, we used a simple one hot encoding method to get the text representation into numerical data for the model, obtaining a corpus from all the words in the TASS dataset and candidate tweets. The dataset was divided in 70-15-15 proportion of training, validation, and test respectively. The model applied was neural network using *Keras*. Then, replies to tweets from presidential candidates were classified using our model. The script of this section was performed using Google Colaboratory servers, and the code is available in Jupyter notebook format as **Ecuadorian_candidates_tweets_sentiment_an.ipynb** and as Python script as **Ecuadorian_candidates_tweets_sentiment_an.py**.  


## Python libraries to run the script of PFMR section 
1. pandas
2. nltk
3. keras
4. matplotlib
5. numpy
6. sklearn





