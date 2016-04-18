from gensim import corpora, models, similarities
from pprint import pprint
from gensim.models.tfidfmodel import TfidfModel
from gensim.models.rpmodel import RpModel

documents = ["Human machine interface for lab abc computer applications",
              "A survey of user opinion of computer system response time",
              "The EPS user interface management system",
              "System and human system engineering testing of EPS",
              "Relation of user perceived response time to error measurement",
              "The generation of random binary unordered trees",
              "The intersection graph of paths in trees",
              "Graph minors IV Widths of trees and well quasi ordering",
              "Graph minors A survey"]

texts = [[word for word in document.lower().split()] for document in documents]
# pprint(texts)

dictionary = corpora.Dictionary(texts)
# dictionary.save('/tmp/deerwester.dict')
# new_doc = "Human computer interaction"
# print(dictionary.token2id)
# print dictionary.doc2bow(new_doc.lower().split())

corpus = [dictionary.doc2bow(text) for text in texts]
# corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)
# print(corpus)

tfidf = TfidfModel(corpus)
rp = RpModel(corpus)
# print rp[corpus[0]]





