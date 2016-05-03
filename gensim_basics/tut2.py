import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities
dictionary = corpora.Dictionary.load('temp/deerwester.dict')
corpus = corpora.MmCorpus('temp/deerwester.mm')
print(corpus)

tfidf = models.TfidfModel(corpus)

doc_bow = [(0, 1), (1, 1)]
print(tfidf[doc_bow])

corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
	print(doc)

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
corpus_lsi = lsi[corpus_tfidf]
lsi.print_topics(2)

'''
model = models.[](corpus)
transformed_corpus = model[corpus]
'''
