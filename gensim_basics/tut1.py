import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities
documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
	for token in text:
		frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1] for text in texts]
from pprint import pprint
# pprint(texts)

dictionary = corpora.Dictionary(texts)
# dictionary.save('temp/deerwester.dict')

new_doc = "Human computer interaction"
print dictionary.doc2bow(new_doc.lower().split())
corpus = [dictionary.doc2bow(text) for text in texts]
# corpora.MmCorpus.serialize('temp/deerwester.mm', corpus)

class MyCorpus(object):
	def __iter__(self):
		for line in documents:
			yield dictionary.doc2bow(line.lower().split())
corpus_memory_friendly = MyCorpus()
print corpus_memory_friendly.__iter__().next()


# corpora.SvmLightCorpus.serialize('temp/corpus.svmlight', corpus)
# corpora.BleiCorpus.serialize('temp/corpus.lda-c', corpus)
# corpora.LowCorpus.serialize('temp/corpus.low', corpus)

