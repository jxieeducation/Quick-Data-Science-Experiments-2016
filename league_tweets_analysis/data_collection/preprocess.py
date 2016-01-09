import json
from datetime import datetime
from dateutil.tz import *

class TweetData:
	def __init__(self, jsonstr):
		self.jsonstr = jsonstr

	def getInfoToStore(self):
		# print self.jsonstr.replace('\\', '')
		parsed_json = json.loads(self.jsonstr)
		text = parsed_json['text']
		source = parsed_json['source']
		ts = int(parsed_json['timestamp_ms'])
		utc_time = datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M:%S")
		if parsed_json['user']['utc_offset']:
			user_time = datetime.fromtimestamp(ts / 1000, tzoffset("UTC", parsed_json['user']['utc_offset'])).strftime("%Y-%m-%d %H:%M:%S")
		else:
			user_time = None
		screen_name = parsed_json['user']['screen_name']
		followers_count = parsed_json['user']['followers_count']
		friends_count = parsed_json['user']['friends_count']
		language = parsed_json['lang']
		urls = [url['expanded_url'] for url in parsed_json['entities']['urls']]

		return_dict = {'text': text, 'source': source, 'utc_time': utc_time, 'user_time': user_time, 'language': language, 'urls': urls, "screen_name": screen_name, "followers_count": followers_count, "friends_count": friends_count}
		return json.dumps(return_dict)
