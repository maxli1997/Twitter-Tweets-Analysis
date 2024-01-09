import snscrape.modules.twitter as sntwitter
import csv

#maxTweets = 20000

'''search_keywords = ['pedestrian crash','pedestrian accident','pedestrian injury',
                        'pedestrian crash distraction','pedestrian accident distraction','car accident involving pedestrians',
                        'car crash involving pedestrians','SUV accident involving pedestrians','SUV crash involving pedestrians',
                        'truck accident involving pedestrians','truck crash involving pedestrians']'''

search_keywords = ['autonomous vehicle','autonomous driving','autonomous car','driverless','self driving']

for month in range(1,12):
#Open/create a file to append data to
    csvFile = open('2020_'+str(month)+'_result.csv', 'a', newline='', encoding='utf8')
    #Use csv writer
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow(['id','date','tweet',]) 
    for search_keyword in search_keywords:
        print(search_keyword)
        #for i,tweet in enumerate(sntwitter.TwitterSearchScraper(search_keyword+' + since:2020-1-01 until:2020-11-01 -filter:links -filter:replies').get_items()):
        if month in [1,3,5,7,8,10,12]:
            for i,tweet in enumerate(sntwitter.TwitterSearchScraper(search_keyword+' + since:2020-'+str(month)+'-01 until:2020-'+str(month)+'-31').get_items()):
                #if i > maxTweets :
                #    print (i)
                #    break  
                csvWriter.writerow([tweet.id, tweet.date, tweet.content])
            print(i)
        elif month in [4,6,9,11]:
            for i,tweet in enumerate(sntwitter.TwitterSearchScraper(search_keyword+' + since:2020-'+str(month)+'-01 until:2020-'+str(month)+'-30').get_items()):
                csvWriter.writerow([tweet.id, tweet.date, tweet.content])
            print(i)
        else:
            for i,tweet in enumerate(sntwitter.TwitterSearchScraper(search_keyword+' + since:2020-2-01 until:2020-2-29').get_items()):
                csvWriter.writerow([tweet.id, tweet.date, tweet.content])
            print(i)
    csvFile.close()