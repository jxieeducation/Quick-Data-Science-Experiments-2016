from gensim.models import Doc2Vec

model = Doc2Vec.load('./imdb.d2v')

print "\n\n--> boxing"
print model.docvecs.most_similar([model['boxing']])

print "\n\n--> naruto"
print model.docvecs.most_similar([model['naruto']])

