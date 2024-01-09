Snscrape -- a open source social network scraper downloaded from Github

Setup.py -- part of snscrape no need to run unless snscrape fails to work

scrape_tweets.py -- scrape tweets and write into csv files month by month can change Social networks from Twitter to other shown under snscrape/modules and change month and year by changing the codes here

frep_plot -- given results from scrape_tweets can generate frequency plot, need to manually change escape word dictionary in the code to ignore not useful words

Nlp.py -- given results from scrape_tweets can generate subjectivity score and sentiment score on each tweets

twitter_watson -- codes from last summer's students. A much complicated nlp library which can give score in detailed emotions like joy, sadness. Need to modify input and output file path before use. I did not use this for my result so I am not sure if it can work properly.