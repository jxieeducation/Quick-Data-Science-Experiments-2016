import numpy as np 
from gensim import corpora, models, similarities
from pprint import pprint
from gensim.models.tfidfmodel import TfidfModel
from scipy.spatial.distance import cosine

class RandomIndexing:
	def __init__(self, documents):
		self.documents = documents
		self.texts = [[word for word in document.lower().split()] for document in documents]
		self.dictionary = corpora.Dictionary(self.texts)
		self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
		self.tfidf = TfidfModel(self.corpus)
		self._make_random_indexing()
		print "initialized!"

	def _make_random_indexing(self, rows=1, cols=512, sparsity=4):
		rows = len(self.dictionary.keys())
		self.RImatrix = np.zeros((rows, cols), dtype=np.int8)
		for i in range(rows):
			pos = np.random.choice(cols, int(sparsity / 2))
			neg = np.random.choice(cols, int(sparsity / 2))
			self.RImatrix[i][pos] = 1
			self.RImatrix[i][neg] = -1
		self.matrix = np.zeros((rows, cols), dtype=np.float32)

	def train(self, corpusTFIDF=None):
		if not corpusTFIDF:
			corpusTFIDF = self.tfidf[self.corpus]
		total = len(corpusTFIDF)
		for count, doc in enumerate(corpusTFIDF):
			if count % 100 == 0:
				print "at " + str(count) + "/" + str(total)
			for w_i, w_i_weight in doc:
				for w_j, w_j_weight in doc:
					self.matrix[w_i] += self.RImatrix[w_j] * w_j_weight

	def most_similar_word(self, word, topn=5):
		if word not in self.dictionary.token2id.keys():
			print "Don't know the word!"
			return None
		word_vec = self.matrix[self.dictionary.token2id[word]]
		score_dict = {}
		for oneId in range(self.matrix.shape[0]):
			score_dict[oneId] = -cosine(word_vec, self.matrix[oneId])
		sorted_list = sorted(score_dict.items(), key=lambda kv: kv[1], reverse=True)
		sorted_list = sorted_list[:topn]
		sorted_list = [self.dictionary[indx] for indx, score in sorted_list]
		return 	sorted_list

	def get_word_vec(self, word, original=False):
		indx = self.dictionary.token2id[word]
		if original:
			return self.RImatrix[indx]
		else:
			return self.matrix[indx]

documents = [line for line in open('small_imdb.txt').read().split('\n')]
r = RandomIndexing(documents)
r.train()
print r.most_similar_word("good", 100)
print r.get_word_vec("good")
print r.get_word_vec("good", True)
