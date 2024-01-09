import pandas as pd
import numpy as np

for year in [2019,2020,2021]:
    df = pd.read_csv(str(year)+'_result_score.csv')
    t_score = df['TextBlob_Score']
    v_score = df['Vader_Score']
    g_score = df['Google_Score']
    avg_score = df['Avg_Score']
    '''print(df.count())
    print(np.mean(t_score))
    print(np.mean(v_score))
    print(np.mean(g_score))    
    print(np.mean(avg_score))'''
    t = df['TextBlob_Sentiment']
    v = df['Vader_Sentiment']
    
    '''print(df[df['Vader_Sentiment']=='positive'].count())
    print(df[df['Vader_Sentiment']=='negative'].count())
    print(df[df['Vader_Sentiment']=='neutral'].count())'''
    print(df[df['TextBlob_Sentiment']=='positive'].count())
    print(df[df['TextBlob_Sentiment']=='negative'].count())
    print(df[df['TextBlob_Sentiment']=='neutral'].count())