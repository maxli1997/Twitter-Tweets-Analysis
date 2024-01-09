# Importing modules
import numpy as np
import pandas as pd
import os
import re
import gensim
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
import gensim.corpora as corpora
import pyLDAvis.gensim
import pickle 
import pyLDAvis
from gensim.models import CoherenceModel
import tqdm



# Remove punctuation
#df = pd.read_csv('./Content/2021_result.csv',usecols=['tweet'])
df = pd.read_csv('./result.csv',usecols=['Dominant_Topic','Text'])
df = df[df['Dominant_Topic']==0]
df.columns = ['Dominant_Topic','tweet']


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

def compute_coherence_values(corpus, dictionary, k, a, b):
    print(a)
    lda_model = gensim.models.LdaModel(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b)
    
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words, dictionary=id2word, coherence='c_v',processes=1)
    print(b)
    return coherence_model_lda.get_coherence()

nltk.download('stopwords')

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



tweets = []
repeat = {}
data = []
for index,row in df.iterrows():
    tweet = cleanTxt(row['tweet'])
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

'''grid = {}
grid['Validation_Set'] = {}
# Topics range
min_topics = 2
max_topics = 11
step_size = 1
topics_range = range(min_topics, max_topics, step_size)
# Alpha parameter
alpha = list(np.arange(0.01, 1, 0.3))
alpha.append('symmetric')
alpha.append('asymmetric')
# Beta parameter
beta = list(np.arange(0.01, 1, 0.3))
beta.append('symmetric')
# Validation sets
num_of_docs = len(corpus)
corpus_sets = [corpus]
corpus_title = ['100% Corpus']
model_results = {'Validation_Set': [],
                 'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': []
                }
# Can take a long time to run
if 1 == 1:
    pbar = tqdm.tqdm(total=540)
    
    # iterate through validation corpuses
    for i in range(len(corpus_sets)):
        # iterate through number of topics
        for k in topics_range:
            # iterate through alpha values
            for a in alpha:
                # iterare through beta values
                for b in beta:
                    # get the coherence score for the given parameters
                    cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=id2word, 
                                                  k=k, a=a, b=b)
                    # Save the model results
                    model_results['Validation_Set'].append(corpus_title[i])
                    model_results['Topics'].append(k)
                    model_results['Alpha'].append(a)
                    model_results['Beta'].append(b)
                    model_results['Coherence'].append(cv)
                    
                    pbar.update(1)
    pd.DataFrame(model_results).to_csv('lda_tuning_results.csv', index=False)
    pbar.close()
quit()'''

# number of topics
a = 0.31
b = 0.91
num_topics = 8
# Build LDA model
lda_model = gensim.models.LdaModel(corpus=corpus,id2word=id2word,num_topics=num_topics,random_state=0,alpha=a,eta=b)
# Print the Keyword in the 10 topics
doc_lda = lda_model[corpus]
model_topics = lda_model.show_topics(formatted=False)

def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=texts):
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


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

df_dominant_topic.to_csv('result.csv')

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

df_dominant_topics.to_csv('topics.csv')
#LDAvis_data_filepath = os.path.join('./results/ldavis_prepared_'+str(num_topics))
# # this is a bit time consuming - make the if statement True
# # if you want to execute visualization prep yourself
#LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
#pyLDAvis.show(LDAvis_prepared)
