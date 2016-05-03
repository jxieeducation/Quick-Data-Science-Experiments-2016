import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities
dictionary = corpora.Dictionary.load('temp/deerwester.dict')
corpus = corpora.MmCorpus('temp/deerwester.mm')
print(corpus)

lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
doc = "Human computer interaction"
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow]
print(vec_lsi)

index = similarities.MatrixSimilarity(lsi[corpus])
sims = index[vec_lsi] 
print(list(enumerate(sims)))




