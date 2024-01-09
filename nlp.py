import re
from textblob import TextBlob #, Word, Blobber
import pandas as pd
import csv


# function: https://ipullrank.com/step-step-twitter-sentiment-analysis-visualizing-united-airlines-pr-crisis/
def cleanTxt(tweet):
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to empty str
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
    #Convert @username to empty str
    tweet = re.sub('@[^\s]+','',tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet

# Mood Function
def mood_function(tweet_text):
    # print(1, tweet_text)
    # print(2, cleanTxt(tweet_text))
    # print(3, cleanData(tweet_text))
    # print()
    # preprocess text and input it into textblob
    text_obj = TextBlob(cleanTxt(tweet_text))
    polarity = text_obj.polarity
    subjectivity = text_obj.subjectivity

    # We can determine the thresholds for tweet mood
    mood = ""
    if polarity < -0.01:
        mood = "negative"
    elif polarity >= -0.01 and polarity <= 0.01:
        mood = "neutral"
    else:
        mood = "positive"

    subj_level = ""
    if subjectivity <= 0.25 and subjectivity >= 0:
        subj_level = "very objective"
    elif subjectivity <= 0.5 and subjectivity > 0.25:
        subj_level = "objective"
    elif subjectivity <= 0.75 and subjectivity > 0.5:
        subj_level = "subjective"
    else:
        subj_level = "very subjective"

    return [mood, polarity, subjectivity, subj_level] 

# Subjectivity and Polarity
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity # or t.subjectivity (the call to sentiment is really not necessary)

def getPolarity(text):
    return TextBlob(text).sentiment.polarity # or t.polarity

for month in range(1,12):
    df = pd.read_csv('2020_'+str(month)+'_result.csv')
    csvFile = open('2020_'+str(month)+'_result_score.csv', 'w', newline='', encoding='utf8')
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow(['id','date','tweet','sentiment','sentiment_score','subjectivity_score','subjectivity']) 

    for index,row in df.iterrows():
        # storeData(tweet): add function
        retweet_status = ""
        retweet_sentiment = ""
        retweet_sentiment_score = ""
        retweet_subjectivity = ""
        retweet_subjectivity_score = ""

        retweet_status = row['tweet']
        # retweet_exists = True
        retweet_list = mood_function(retweet_status)
        retweet_sentiment = retweet_list[0]
        retweet_sentiment_score = retweet_list[1]
        retweet_subjectivity_score = retweet_list[2]
        retweet_subjectivity = retweet_list[3]

        #Use csv writer
        csvWriter.writerow([row['id'],row['date'],row['tweet'],retweet_sentiment,retweet_sentiment_score,retweet_subjectivity_score,retweet_subjectivity])