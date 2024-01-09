import re
from textblob import TextBlob #, Word, Blobber
import pandas as pd
import csv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from google.cloud import language_v1
import numpy as np


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

client = language_v1.LanguageServiceClient.from_service_account_json("/Users/limeitang/Desktop/twitternlp-317323-af4ef73bfb22.json")

for year in [2019,2020,2021]:
    df = pd.read_csv(str(year)+'_result.csv')
    csvFile = open(str(year)+'_result_score.csv', 'w', newline='', encoding='utf8')
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow(['id','date','tweet','TextBlob_Sentiment','TextBlob_Score','Vader_Sentiment','Vader_Score','Google_Score','Google_Magnitude','Avg_Score']) 
    analyzer = SentimentIntensityAnalyzer()

    for index,row in df.iterrows():
        # storeData(tweet): add function
        retweet_status = ""
        retweet_sentiment = ""
        retweet_sentiment_score = ""

        retweet_status = row['tweet']
        # retweet_exists = True
        retweet_list = mood_function(retweet_status)
        retweet_sentiment = retweet_list[0]
        retweet_sentiment_score = retweet_list[1]

        cleaned_tweet = cleanTxt(row['tweet'])
        vs = analyzer.polarity_scores(cleaned_tweet)
        v_score = vs['compound']
        if v_score >= 0.05:
            v_sentiment = 'positive'
        elif v_score <= -0.05:
            v_sentiment = 'negative'
        else:
            v_sentiment = 'neutral'
        
        # The text to analyze
        encoded_string = cleaned_tweet.encode("ascii", "ignore")
        decode_string = encoded_string.decode()
        text = decode_string
        try:
            document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
            # Detects the sentiment of the text
            sentiment = client.analyze_sentiment(request={'document': document}).document_sentiment
        except:
            continue
        g_score = sentiment.score
        g_magnitude = sentiment.magnitude
        #g_score = 0
        #g_magnitude = 0

        avg = np.mean([retweet_sentiment_score,v_score,g_score])
        #avg = np.mean([retweet_sentiment_score,v_score])
        #Use csv writer
        csvWriter.writerow([row['id'],row['date'],row['tweet'],retweet_sentiment,retweet_sentiment_score,v_sentiment,v_score,g_score,g_magnitude,avg])