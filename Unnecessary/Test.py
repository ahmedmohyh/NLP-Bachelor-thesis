# import nltk
# from gensim.models import Word2Vec
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn import cluster
# from sklearn import metrics
# from sklearn.decomposition import PCA
# from scipy.cluster import hierarchy
# from sklearn.cluster import AgglomerativeClustering
# import matplotlib.pyplot as plt
#
# # creating Own dataset
# sentences = [['this', 'is', 'learning', 'good', 'deep', 'good', 'book'],
#              ['this', 'is', 'another', 'book'],
#              ['one ', 'more', 'book'],
#              ['train', 'railway', 'station'],
#              ['time', 'train', 'station'],
#              ['time', 'railway', 'station', 'train'],
#              ['this', 'is', 'the', 'new', 'post'],
#              ['this', 'is', 'about', 'more', 'deep', 'learning', 'post'],
#              ['and', 'this', 'is', 'the', 'one'],
#              ]
# m = Word2Vec(sentences, vector_size=50, min_count=1, sg=1)
# print(m)
#
#
#
#
#
#
#
# def vectorizer(sent, m2):
#     vec = []
#     numw = 0
#     for w in sent:
#         try:
#             if numw == 0:
#                 vec = m2[w]
#             else:
#                 vec = np.add(vec, m2[w])
#             numw += 1
#         except:
#             pass
#     print(np.asarray(vec) / numw)
#     return np.asarray(vec) / numw
#
#
# l = []
# for i in sentences:
#     print(i)
#     l.append(vectorizer(i, m))
# X = np.array(1)
#
# wcss = []
# # for i in range(1, 4):
# #     kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
# #     kmeans.fit(X)
# #     wcss.append(kmeans.inertia_)
# # plt.plot(range(1, 4), wcss)
# # plt.title('the Elbow Method')
# # plt.xlabel('Number of clusters')
# # plt.ylabel('WCSS')
# # plt.show()
