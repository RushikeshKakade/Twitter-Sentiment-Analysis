# Twitter-Sentiment-Analysis
- Twitter Sentiment Analysis It is a Natural Language Processing Problem where Sentiment Analysis is done by Classifying the Positive tweets from negative tweets by machine learning models for classification, text mining, text analysis, data analysis and data visualization.
  *In this project we have built a model which takes a dataset as an input and as an output gives the percentage of Positive ,Negative,Neutral and Irrelevant tweets in the given dataset. It is done using natural language processing library using scikit learn machine learning libraries.
  
## Sentiment Analysis
- Sentiment analysis refers to a process which is used for determining if the given information is positive , negative , neutral or irrelevant using computational methods .

## Introduction
- Natural Language Processing (NLP) is a research in data science these days and one of the most common applications of NLP is sentiment analysis. From opinion polls to creating entire marketing strategies, this domain has completely reshaped the way businesses work, which is why this is an area every data scientist must be familiar with.

- Thousands of text documents can be processed for sentiment (and other features including named entities, topics, themes, etc.) in seconds, compared to the hours it would take a team of people to manually complete the same task.

- We will do so by following a sequence of steps needed to solve a general sentiment analysis problem. We will start with preprocessing and cleaning of the raw text of the tweets. Then we will explore the cleaned text and try to get some intuition about the context of the tweets. After that, we will extract numerical features from the data and finally use these feature sets to train models and identify the sentiments of the tweets.

- This is one of the most interesting challenges in NLP so I’m very excited to take this journey with you!

## Problem Statement
Let’s go through the problem statement once as it is very crucial to understand the objective before working on the dataset. The problem statement is as follows:

The objective of this task is to detect different types of sentiment in tweets. For the sake of simplicity, we say a tweet contains Positive,Negative,Neutral,Itterative sentiment if it has tweet associated with it. So, the task is to classify Positive,Negative,Neutral,Itterative sentiment tweets from other tweets so our objective is to predict the labels on the given test dataset.

## Characteristic features of Tweets
From the perspective of Sentiment Analysis, we discuss a few characteristics of Twitter:

#### Length of a Tweet 
The maximum length of a Twitter message is 140 characters. This means that we can practically consider a tweet to be a single sentence, void of complex grammatical constructs. This is a vast difference from traditional subjects of Sentiment Analysis, such as movie reviews.

#### Language used 
Twitter is used via a variety of media including SMS and mobile phone apps. Because of this and the 140-character limit, language used in Tweets tend be more colloquial, and filled with slang and misspellings. Use of hashtags also gained popularity on Twitter and is a primary feature in any given tweet. Our analysis shows that there are approximately 1-2 hashtags per tweet.

#### Data availability
Another difference is the magnitude of data available. With the Twitter API, it is easy to collect millions of tweets for training. There also exist a few datasets that have automatically and manually labelled the tweets.

#### Domain of topics 
People often post about their likes and dislikes on social media. These are not al concentrated around one topic. This makes twitter a unique place to model a generic classifier as opposed to domain specific classifiers that could be build datasets such as movie reviews.

## Dataset
- One of the major challenges in Sentiment Analysis of Twitter is to collect a labelled dataset. Researchers have made public the following datasets for training and testing classifiers.
- Dataset Link :- https://www.kaggle.com/jp797498e/twitter-entity-sentiment-analysis

## Steps for the project

  1.Import Libraries
  
  2.Tweets Mining
  
  3.Data Cleaning
  
  4.Location Geocoding
  
  5.Tweets Processing
  
  6.Data Exploration
  
  7.Sentiment Analysis

## Tweets Processing Steps
#### 1: Segmentation of the Sentence
- The first step in the given text is to break the text apart into separate sentences. We can assume that each sentence in English is a separate thought or idea. It will be a lot easier to write a program to understand a single sentence than to understand a whole paragraph. Coding a Sentence Segmentation model can be as simple as splitting apart sentences whenever you see a punctuation mark. But modern NLP pipelines often use more complex techniques that work even when a document isn’t formatted cleanly.

#### 2: Tokenizing 
- The words in the text Now that we’ve split our document into sentences, we can process them one at a time. The next step in our pipeline is to break this sentence into separate words or tokens. This is called tokenization. Tokenization is easy to do in English. We’ll just split apart words whenever there’s a space between them. And we’ll also treat punctuation marks as separate tokens since punctuation also has meaning.

#### 3:Hashtags
- A hashtag is a word or an un-spaced phrase prefixed with the hash symbol (#). These are used to both naming subjects and phrases that are currently in trending topics. For example, #iPad, #news

Regular Expression: #(\w+)

Replace Expression: HASH_\1

#### 4:Handles
- Every Twitter user has a unique username. Any thing directed towards that user can be indicated be writing their username preceded by ‘@’. Thus, these are like proper nouns. For example, @Apple

Regular Expression: @(\w+)

Replace Expression: HNDL_\1

#### 5:URLs
- Users often share hyperlinks in their tweets. Twitter shortens them using its in-house URL shortening service, like http://t.co/FCWXoUd8 - such links also enables Twitter to alert users if the link leads out of its domain. From the point of view of text classification, a particular URL is not important. However, presence of a URL can be an important feature. Regular expression for detecting a URL is fairly complex because of different types of URLs that can be there, but because of Twitter’s shortening service, we can use a relatively simple regular expression.

Regular Expression: (http|https|ftp)://[a-zA-Z0-9\\./]+

Replace Expression: URL

#### 6:Emoticons
- Use of emoticons is very prevalent throughout the web, more so on micro- blogging sites. We identify the following emoticons and replace them with a single word lists the emoticons we are currently detecting. All other emoticons would be ignored.

#### 7:Punctuations
- Although not all Punctuations are important from the point of view of classification but some of these, like question mark, exclamation mark can also provide information about the sentiments of the text. We replace every word boundary by a list of relevant punctuations present at that point  lists the punctuations currently identified. We also remove any single quotes that might exist in the text.

#### 8:Repeating Characters
- People often use repeating characters while using colloquial language, like "I’m in a hurrryyyyy", "We won, yaaayyyyy!" As our final pre-processing step, we replace characters repeating more than twice as two characters.

Regular Expression: (.)\1{1,}

Replace Expression: \1\1

## Stemming Algorithms
- All stemming algorithms are of the following major types – affix removing, statistical and mixed. The first kind, Affix removal stemmer, is the most basic one. These apply a set of transformation rules to each word in an attempt to cut off commonly known prefixes and / or suffixes. A trivial stemming algorithm would be to truncate words at N-th symbol. But this obviously is not well suited for practical purposes.

#### 1.Porter Stemmer
- Martin Porter wrote a stemmer that was published in July 1980. This stemmer was very widely used and became and remains the de facto standard algorithm used for English stemming. It offers excellent trade-off between speed, readability, and accuracy. It uses a set of around 60 rules applied in 6 successive steps. An important feature to note is that it doesn’t involve recursion. 

#### 2.Lemmatization
Lemmatization is the process of normalizing a word rather than just finding its stem. In the process, a suffix may not only be removed, but may also be substituted with a different one. It may also involve first determining the part-of-speech for a word and then applying normalization rules. It might also involve dictionary look-up. For example, verb ‘saw’ would be lemmatized to ‘see’ and the noun ‘saw’ will remain ‘saw’. For our purpose of classifying text, stemming should suffice.

## Stop words 
- Stop words  are a set of commonly used words in a language. Examples of stop words in English are “a”, “the”, “is”, “are” and etc. Stop words are commonly used in Text Mining and Natural Language Processing (NLP) to eliminate words that are so commonly used that they carry very little useful information.

## Word Cloud Generation
- To get the most common words used to describe 2020, I made use of the POS-tag (Parts of Speech tagging) module in the NLTK library. Using the WordCloud library, one can generate a Word Cloud based on word frequency and superimpose these words on any image. In this case, I used the Twitter logo and Matplotlib to display the image. The Word Cloud shows the words with higher frequency in bigger text size while the "not-so" common words are in smaller text sizes.

## Vectorization 
- Vectorization is the process of converting an algorithm from operating on individual matrix elements one at a time, to operating on a batch of values in a single operation.Text Vectorization is the process of converting text into numerical representation.

#### 1.TF-IDF or ( Term Frequency(TF) Inverse Dense Frequency(IDF) ) 
— It is a technique which is used to find meaning of sentences consisting of words and cancels out the incapabilities of Bag of Words technique which is good for text classification or for helping a machine read words in numbers.

#### 2. Bag of words
- A bag-of-words model, or BoW for short, is a way of extracting features from text for use in modeling, such as with machine learning algorithms. The approach is very simple and flexible, and can be used in a myriad of ways for extracting features from documents.

## Story Generation and Visualization from Tweets
- In this section, we will explore the cleaned tweets text. Exploring and visualizing data, no matter whether its text or any other data, is an essential step in gaining insights. Do not limit yourself to only these methods told in this tutorial, feel free to explore the data as much as possible.

- Before we begin exploration, we must think and ask questions related to the data in hand. A few probable questions are as follows:

- What are the most common words in the entire dataset? What are the most common words in the dataset for negative and positive tweets, respectively? How many hashtags are there in a tweet? Which trends are associated with my dataset? Which trends are associated with either of the sentiments? Are they compatible with the sentiments?

## Conclusion
We create a sentiment classifier for twitter using labelled data sets. We also investigate the relevance of using a RandomForestClassifier,Logistic Regression and Decision Tree Classifier machine learning algorithm and negation detection for the purpose of sentiment analysis.

- In this project our accuracy result is follows :
   ML Algorithm                         =             Accuracy in percentage
1. Random_forest_Tfidf_Accuracy	        =             90.066667
2. Logistic_Regression_Tfidf_Accuracy	  =             76.026667
3. Decision_tree_tfidf_Accuracy	        =             77.960000
4. Random_forest_Bow_Accuracy	          =             89.740000
5. Logistic_Regression_Bow_Accuracy	    =             80.046667
6. Decision_tree_Bow_Accuracy	          =             80.54666

- From above we can concluded that the Best Model for the "Twitter Sentiment Analysis NLP Project" is 'Random Forest Classifier' and the best vectorization method is "TFIDF Vectorizer" in which the model gives the 90.06 % acuuracy which is higher rhan other Machine learning models there we have use Random Forest Classifier as a best model.









