'''
python -i train_random_indexing.py ../../EmbeddingMapper/exp1/0corpus/test-pos.txt
'''
from gensim import corpora, models, similarities
from pprint import pprint
from gensim.models.tfidfmodel import TfidfModel
from gensim.models.rpmodel import RpModel
import sys

documents = open(sys.argv[1]).read().split('\n')
texts = [[word for word in document.lower().split()] for document in documents]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

tfidf = TfidfModel(corpus)
rp = RpModel(corpus)

exp = "imdb"
rp.save('models/' + exp + '.rp_model')
dictionary.save('models/' + exp + '.dict')

def mostSimilarWord(w):
	score_dict = {}
	theId = dictionary.token2id[w]
	w_vector = rp.projection[theId]
	for oneId in range(rp.projection.shape[0]):
		score_dict[oneId] = -cosine(w_vector, rp.projection[oneId])
	sorted_list = sorted(score_dict.items(), key=lambda kv: kv[1], reverse=True)
	return sorted_list
