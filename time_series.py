import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import gensim.corpora as corpora
from nltk.corpus import stopwords
import nltk
import gensim
from gensim.utils import simple_preprocess

search_keywords = ['ADAS'
,'advanced driving assistance system'
,'FCW'
,'Crash warning'
,'forward crash warning'
,'LDW'
,'lane departure warning'
,'LKA'
,'Lane Keeping assist'
,'ACC'
,'Adaptive Cruise Control'
,'Autopilot'
,'Super cruise'
,'lane centering']

def cleanTxt(tweet):
    tweet = tweet.strip('\'"')
    encoded_string = tweet.encode("ascii", "ignore")
    tweet = encoded_string.decode()
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to empty str
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
    #Convert @username to empty str
    tweet = re.sub('@[^\s]+','',tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #Remove additional white spaces
    tweet = re.sub('[,\.!?:;()]', ' ', tweet)
    tweet = re.sub('&amp', ' ', tweet)
    tweet = re.sub('[\s]+', ' ', tweet)
    #trim
    return tweet

stop_words = stopwords.words('english')
stop_words.extend(['adas'
        ,'advanced'
        ,'assistance'
        ,'system'
        ,'fcw'
        ,'crash'
        ,'warning'
        ,'forward'
        ,'lane'
        ,'departure'
        ,'keeping'
        ,'assist'
        ,'adaptive'
        ,'cruise'
        ,'control'
        ,'autopilot'
        ,'super'
        ,'centering'
        ,'driving'
        ,'ldw'
        ,'lka'
        ,'acc'])
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]

def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

def remove_irrelevant(df):
    idx = []
    for i, row in df.iterrows():
        found = False
        for phrase in search_keywords:
            if phrase in row.tweet:
                found = True
                break
        if not found:
            idx.append(i)
    df.drop(idx, inplace=True)
    return


years = ['2019','2020','2021']

for year in years:
    filepath = os.path.join('./Content',year+'_result_score.csv')
    df = pd.read_csv(filepath,usecols=['date','tweet','Avg_Score'])
    remove_irrelevant(df)
    df = df[df.date!='date']
    df.date = df.date.map(lambda x: x.split(' ')[0])
    dates = df.date.unique()
    dates = np.sort(dates)
    pos_tweets = []
    pos_score = []
    neg_tweets = []
    neg_score = []
    for date in dates:
        temp_df = df[df.date==date]
        pos = temp_df[temp_df.Avg_Score>0.05]
        neg = temp_df[temp_df.Avg_Score<-0.05]
        pos_tweets.append(len(pos))
        neg_tweets.append(len(neg))
        pos_score.append(np.mean(pos.Avg_Score))
        neg_score.append(np.mean(neg.Avg_Score))
    
    pos_indices = np.argsort(pos_tweets)[-5:]
    neg_indices = np.argsort(neg_tweets)[-5:]
    pos_tweets = np.array(pos_tweets)
    neg_tweets = np.array(neg_tweets)

    fig, ax = plt.subplots(2,constrained_layout=True)
    dates = np.array([x.split(year+'-')[1] for x in dates])
    ax[0].plot(dates,pos_tweets,label='Positive Tweets Number')
    ax[0].plot(dates,neg_tweets,label='Negative Tweets Number')
    ax[1].plot(dates,pos_score,label='Positive Tweets Score')
    ax[1].plot(dates,neg_score,label='Negative Tweets Score')
    ax[0].set_xticks(np.arange(0, len(dates)+1, 30))
    ax[1].set_xticks(np.arange(0, len(dates)+1, 30))
    ax[0].scatter(dates[pos_indices],pos_tweets[pos_indices])
    ax[0].scatter(dates[neg_indices],neg_tweets[neg_indices])
    for tick_1, tick_2 in zip(ax[0].get_xticklabels(),ax[1].get_xticklabels()):
        tick_1.set_rotation(45)
        tick_2.set_rotation(45)
    ax[0].legend()
    ax[1].legend()
    ax[0].set_title('Number of Tweets')
    ax[1].set_title('Average Score')
    fig.suptitle(year+' ADAS Tweets Result')
    filename = './TimeSeries/'+year+'_1.jpg'
    plt.show()
    
    fig, ax = plt.subplots(2,constrained_layout=True)
    ax[0].plot(dates,pos_tweets,label='Positive Tweets Number')
    ax[0].plot(dates,neg_tweets,label='Negative Tweets Number')
    ax[1].plot(dates,pos_tweets,label='Positive Tweets Number')
    ax[1].plot(dates,neg_tweets,label='Negative Tweets Number')
    ax[0].set_xticks(np.arange(0, len(dates)+1, 30))
    ax[1].set_xticks(np.arange(0, len(dates)+1, 30))
    for idx in pos_indices:
        ax[0].annotate(str(dates[idx])+' '+str(pos_tweets[idx]), (dates[idx],pos_tweets[idx]))
    for idx in neg_indices:
        ax[1].annotate(str(dates[idx])+' '+str(neg_tweets[idx]), (dates[idx],neg_tweets[idx]))
    ax[0].scatter(dates[pos_indices],pos_tweets[pos_indices])
    ax[1].scatter(dates[neg_indices],neg_tweets[neg_indices],c='#ff7f0e')
    for tick_1, tick_2 in zip(ax[0].get_xticklabels(),ax[1].get_xticklabels()):
        tick_1.set_rotation(45)
        tick_2.set_rotation(45)
    ax[0].legend()
    ax[1].legend()
    ax[0].set_title('Peaks of Positive Tweets')
    ax[1].set_title('Peaks of Negative Tweets')
    fig.suptitle(year+' ADAS Tweets Result')
    filename = './TimeSeries/'+year+'_2.jpg'
    plt.show()

    i = 0
    path = './TimeSeries/' + year + '_lda/'

    for date in np.concatenate([dates[pos_indices], dates[neg_indices]]):
        i += 1
        date = year + '-' + date
        if i <= 5:
            temp_df = df[(df.date==date) & (df.Avg_Score>0.05)]
        else:
            temp_df = df[(df.date==date) & (df.Avg_Score<-0.05)]
        tweets = []
        repeat = {}
        data = []

        for index,row in temp_df.iterrows():
            tweet = cleanTxt(row.tweet)
            if tweet not in repeat:
                tweets.append(tweet)
                data.append(tweet)
                repeat[tweet] = 1

        data_words = list(sent_to_words(tweets))
        # remove stop words
        data_words = remove_stopwords(data_words)
        # Create Dictionary
        id2word = corpora.Dictionary(data_words)
        # Create Corpus
        texts = data_words
        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]

        lda_model = gensim.models.LdaModel(corpus=corpus,id2word=id2word,num_topics=10,random_state=0)
        # Print the Keyword in the 10 topics
        doc_lda = lda_model[corpus]
        model_topics = lda_model.show_topics(formatted=False)
        df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data)

        # Format
        df_dominant_topic = df_topic_sents_keywords.reset_index()
        df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

        

        # Number of Documents for Each Topic
        topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

        # Percentage of Documents for Each Topic
        topic_contribution = round(topic_counts/topic_counts.sum(), 4)

        # Topic Number and Keywords
        topic_num_keywords = pd.DataFrame(model_topics)

        # Concatenate Column wise
        df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

        # Change Column names
        df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']
        if i <= 5:
            df_dominant_topics.to_csv(path+date+'_pos_topics.csv')
            df_dominant_topic.to_csv(path+date+'_pos_result.csv')
        else:
            df_dominant_topics.to_csv(path+date+'_neg_topics.csv')
            df_dominant_topic.to_csv(path+date+'_neg_result.csv')
    