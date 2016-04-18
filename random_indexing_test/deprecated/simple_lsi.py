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
dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = TfidfModel(corpus)

lsi = models.LsiModel(tfidf[corpus], id2word=dictionary, num_topics=2)

print lsi.print_topics(2)

print lsi[tfidf[corpus[0]]]

# corpus_lsi = lsi[tfidf]
