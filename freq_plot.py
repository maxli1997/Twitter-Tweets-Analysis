import pandas as pd
import csv
from nltk.corpus import stopwords
from nltk import FreqDist
import re
from collections import Counter
import matplotlib.pyplot as plt
import string


stop_words = set(stopwords.words('english'))

word_list = {}
stopwords = []

gist_file = open("gist_stopwords.txt", "r")
try:
    content = gist_file.read()
    stopwords = content.split(",")
    stop_words=[i.replace('"',"").strip() for i in stopwords]
finally:
    gist_file.close()

for year in [2019]:
    df = pd.read_csv(str(year)+'_result_score.csv')
    df = df[df['TextBlob_Sentiment']=='negative']
    for text in df["tweet"]:
        text = str(text).lower()
        # Removals
        translator = re.compile('[%s]' % re.escape(string.punctuation))
        text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))','',text) # Removes hyperlink
        text = translator.sub(' ', text)    

        text = re.sub(r'#([^\s]+)', r'\1', text) #Replace #word with word
        
        text = re.sub(r'(\bNaN\b)|(\bnan\b)','',text) # Removes NaN values 
        emoji_patterns = re.compile(
            "(["
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002702-\U000027B0"  # Dingbats
            "])"
        )
        text = emoji_patterns.sub('', text)
        
        # Cleanup
        text = re.sub(r'[\s]+', ' ', text)  # Removes additional white spaces
        text = text.strip('\'"').lstrip().rstrip() # Trim
        sep_word = text.split()
        for word in sep_word:
            word = word.lower()
            if word in word_list:
                word_list[word] += 1
            else:
                word_list[word] = 1
    #f = open("result.txt","w+")
    stopwords = ['adas'
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
        ,'acc']
    for word in stopwords:
        word_list[word] = 0
    for word in stop_words:
        word_list[word] = 0

    counter_dict = Counter(word_list)
    print(counter_dict.most_common(50))
    freqDf = pd.DataFrame(counter_dict.most_common(20), columns=["words", "count"])
    freqDf = freqDf.sort_values(by='count')
    freqDict = freqDf.to_dict() # used in 'pie'

    # Init Params
    fig, ax = plt.subplots(figsize=(12,8))

    # Plot
    freqDf.plot.barh(
        x='words', 
        y='count',
        ax=ax,
        color='green'
    )
    gtitle='Freq Graph'
    # Save and Close Graph
    ax.set_title(gtitle.replace('**', str(20)))
    plt.savefig(str(year)+'_result.png')
    plt.clf()

    # End

