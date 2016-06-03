import gensim
from gensim import corpora, models, similarities
import sys
from random import shuffle

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

word2vec_sentiment_dir = "/Users/jason.xie/Downloads/word2vec-sentiments/"

def get_lines(myDir): 
	lines = open(word2vec_sentiment_dir + "test-pos.txt").readlines()
	return [line.split() for line in lines]

get_lines(word2vec_sentiment_dir + "test-pos.txt")

pos_sentences = get_lines(word2vec_sentiment_dir + "test-pos.txt") + get_lines(word2vec_sentiment_dir + "train-pos.txt")
neg_sentences = get_lines(word2vec_sentiment_dir + "test-neg.txt") + get_lines(word2vec_sentiment_dir + "train-neg.txt")

dictionary = corpora.Dictionary(pos_sentences + neg_sentences)
dictionary.filter_extremes(no_below=25, no_above=0.5)
dictionary.save('data/imdb.dict')

UNKNOWN = len(dictionary.token2id.keys())

counter = 0 # lol bad code
def reformat(line, posOrNeg=0):
	global counter
	if counter % 100 == 0:
		print counter
	counter += 1
	myStr = str(posOrNeg) + "\t"
	vectorLine = [dictionary.token2id[word] if word in dictionary.token2id.keys() else UNKNOWN for word in line]
	myStr += ",".join([str(i) for i in vectorLine])
	return myStr

print "\nformatting the corpus!!\n"
lines = []
lines += [reformat(line, posOrNeg=1) for line in pos_sentences]
lines += [reformat(line, posOrNeg=0) for line in neg_sentences]
shuffle(lines)

train_file = open("data/train.txt", "wb")
test_file = open("data/test.txt", "wb")

for index, line in enumerate(lines):
	myFile = train_file
	if index > 40000:
		myFile = test_file
	myFile.write(line)
	myFile.write("\n")
