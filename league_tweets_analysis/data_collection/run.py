import tweepy
from preprocess import TweetData
import json

secret = json.loads(open('secret.json').read())
auth = tweepy.OAuthHandler(secret['o1'], secret['o2'])
auth.set_access_token(secret['t1'], secret['t2'])

f = open('data.json', 'ab', 0)

class MyStreamListener(tweepy.StreamListener):
	def on_data(self, tweet):
		tweet = TweetData(tweet)
		data = tweet.getInfoToStore()
		print data
		f.write(data)
		f.write('\n')

myStream = tweepy.Stream(auth = auth, listener=MyStreamListener())
myStream.filter(track=['league of legends'])
