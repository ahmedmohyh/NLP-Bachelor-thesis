# import os
# import string
# from nltk import tokenize
# from gensim.models import Word2Vec
# from nltk import word_tokenize
# import numpy as np
# from sklearn.cluster import MiniBatchKMeans
# from sklearn.metrics import silhouette_score
# from sklearn.metrics import silhouette_samples
# import pandas as pd
# import gensim
# from nltk.corpus import stopwords
# import sentence_transformers
# from sentence_transformers import SentenceTransformer,util
# from sklearn.cluster import KMeans
#
#
# model = SentenceTransformer('all-mpnet-base-v2', device="cuda")
# # embedder = SentenceTransformer('distilbert-base-nli-mean-tokens')
#
#
#
# sentences = []
# for filename in os.listdir(r"D:\UDE\6th Semester\MEMS\MEWS Data\MEWS_Essays\MEWS_Essays\Essays_all\Only 100"):
#    with open(os.path.join(r"D:\UDE\6th Semester\MEMS\MEWS Data\MEWS_Essays\MEWS_Essays\Essays_all\Only 100", filename)) as f:
#        text = f.read()
#        text = text.replace("ï»¿","")
#        sents = tokenize.sent_tokenize(text)
#        for s in sents:
#            #s = s.lower()
#            #s = s.translate(str.maketrans('', '', string.punctuation))
#            sentences.append(s)
#
#
# corpus_embeddings = model.encode(sentences, show_progress_bar =True, device="cuda")
#
# corpus_embeddings.shape
#
#
# def mbkmeans_clusters(X, k, mb=500, print_silhouette_values=False):
#     """Generate clusters.
#
#     Args:
#         X: Matrix of features.
#         k: Number of clusters.
#         mb: Size of mini-batches. Defaults to 500.
#         print_silhouette_values: Print silhouette values per cluster.
#
#     Returns:
#         Trained clustering model and labels based on X.
#     """
#     km = MiniBatchKMeans(n_clusters=k, batch_size=mb).fit(X)
#     print(f"For n_clusters = {k}")
#     print(f"Silhouette coefficient: {silhouette_score(X, km.labels_):0.2f}")
#     print(f"Inertia:{km.inertia_}")
#
#     if print_silhouette_values:
#         sample_silhouette_values = silhouette_samples(X, km.labels_)
#         print(f"Silhouette values:")
#         silhouette_values = []
#         for i in range(k):
#             cluster_silhouette_values = sample_silhouette_values[km.labels_ == i]
#             silhouette_values.append(
#                 (
#                     i,
#                     cluster_silhouette_values.shape[0],
#                     cluster_silhouette_values.mean(),
#                     cluster_silhouette_values.min(),
#                     cluster_silhouette_values.max(),
#                 )
#             )
#         silhouette_values = sorted(
#             silhouette_values, key=lambda tup: tup[2], reverse=True
#         )
#         for s in silhouette_values:
#             print(
#                 f"    Cluster {s[0]}: Size:{s[1]} | Avg:{s[2]:.2f} | Min:{s[3]:.2f} | Max: {s[4]:.2f}"
#             )
#     return km, km.labels_
#
# clustering, cluster_labels = mbkmeans_clusters(X=corpus_embeddings, k=50, print_silhouette_values=True)
#
# df_clusters = pd.DataFrame({
#     "text": sentences,
#     "cluster": cluster_labels
# })
#
#
# test_cluster = 1
# most_representative_docs = np.argsort(
#     np.linalg.norm(corpus_embeddings - clustering.cluster_centers_[test_cluster], axis=1)
# )
# # print(most_representative_docs[0])
# for d in most_representative_docs[:10]:
#     print(d)
#     print(sentences[d])
#     print("-------------")
#
#
# #df_clusters.to_csv('SbertClustering100Essays.csv')
