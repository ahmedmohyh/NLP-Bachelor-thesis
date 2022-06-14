from gensim.models import Word2Vec

sentences = [['this', 'is', 'the', 'good', 'machine', 'learning', 'book'],
             ['this', 'is', 'another', 'book'],
             ['one', 'more', 'book'],
             ['this', 'is', 'the', 'new', 'post'],
             ['this', 'is', 'about', 'machine', 'learning', 'post'],
             ['and', 'this', 'is', 'the', 'last', 'post']
             ]

model = Word2Vec(sentences, min_count=1)

print(model.wv.similarity('this', 'is'))
print(model.wv.similarity('post', 'book'))
# output -0.0198180344218
# output -0.079446731287
print(model.wv.most_similar(positive=['machine'], negative=[], topn=2))
# output: [('new', 0.24608060717582703), ('is', 0.06899910420179367)]
# print(model['the'])
# output [-0.00217354 -0.00237131  0.00296396 ...,  0.00138597  0.00291924  0.00409528]
print(list(model.wv.index_to_key))
print(len(list(model.wv.index_to_key)))

X = list(model.wv.index_to_key)

from nltk.cluster import KMeansClusterer
import nltk

NUM_CLUSTERS = 3
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
print(assigned_clusters)
