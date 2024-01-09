import snscrape.modules.twitter as sntwitter
import csv

#maxTweets = 20000

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

for year in [2019,2020,2021]:
#Open/create a file to append data to
    csvFile = open(str(year)+'_result.csv', 'a', newline='', encoding='utf8')
    #Use csv writer
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow(['id','date','tweet',]) 
    for search_keyword in search_keywords:
        new_keyword = search_keyword+' driving'
        print(new_keyword)
        #for i,tweet in enumerate(sntwitter.TwitterSearchScraper(search_keyword+' + since:2020-1-01 until:2020-11-01 -filter:links -filter:replies').get_items()):
        for i,tweet in enumerate(sntwitter.TwitterSearchScraper(new_keyword+' + since:'+str(year)+'-01-01 until:'+str(year)+'-12-31').get_items()):
            if tweet.lang == 'en':
                csvWriter.writerow([tweet.id, tweet.date, tweet.content])
        print(i)
    csvFile.close()