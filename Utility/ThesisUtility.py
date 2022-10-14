import gensim
import numpy as np
import pandas as pd
import scipy
from nltk import tokenize
from nltk.corpus import stopwords
from scipy import sparse
from scipy.sparse import vstack, hstack, csr_matrix
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score, confusion_matrix, cohen_kappa_score
from pathlib import Path
from collections import defaultdict
import os
import re

from sentence_transformers import util


'''
This function calculates the average feature vector of each sentence 
@:parameter list_of_docs the
@:parameter model of all sentences. 
@:parameter strategy min-max or average.  
'''
def vectorize(list_of_docs, model, strategy):
    """Generate vectors for list of documents using a Word Emx`bedding.

    Args:
        list_of_docs: List of documents.
        model: Gensim Word Embedding.
        strategy: Aggregation strategy ("average", or "min-max".)

    Raises:
        ValueError: If the strategy is other than "average" or "min-max".

    Returns:
        List of vectors.
    """
    features = []
    size_output = 100
    embedding_dict = model

    #Word2Vec setups
    if hasattr(model, "wv"):
        size_output = model.vector_size
        embedding_dict = model.wv

    if strategy == "min-max":
        size_output *= 2

    for tokens in list_of_docs:
        zero_vector = np.zeros(size_output)
        vectors = []
        for token in tokens:
            if token in embedding_dict:
                try:
                    vectors.append(embedding_dict[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            if strategy == "min-max":
                min_vec = vectors.min(axis=0)
                max_vec = vectors.max(axis=0)
                features.append(np.concatenate((min_vec, max_vec)))
            elif strategy == "average":
                avg_vec = vectors.mean(axis=0)
                features.append(avg_vec)
            else:
                raise ValueError(f"Aggregation strategy {strategy} does not exist!")
        else:
            features.append(zero_vector)
    return features

'''
This function return the clustering results with the assigned 
'''
def mbkmeans_clusters(X, k, mb=500, print_silhouette_values=False):
    """Generate clusters.

    Args:
        X: Matrix of features.
        k: Number of clusters.
        mb: Size of mini-batches. Defaults to 500.
        print_silhouette_values: Print silhouette values per cluster.

    Returns:
        Trained clustering model and labels based on X.
    """
    km = MiniBatchKMeans(n_clusters=k, batch_size=mb).fit(X)
    print(f"For n_clusters = {k}")
    print(f"Silhouette coefficient: {silhouette_score(X, km.labels_):0.2f}")
    print(f"Inertia:{km.inertia_}")

    if print_silhouette_values:
        sample_silhouette_values = silhouette_samples(X, km.labels_)
        print(f"Silhouette values:")
        silhouette_values = []
        for i in range(k):
            cluster_silhouette_values = sample_silhouette_values[km.labels_ == i]
            silhouette_values.append(
                (
                    i,
                    cluster_silhouette_values.shape[0],
                    cluster_silhouette_values.mean(),
                    cluster_silhouette_values.min(),
                    cluster_silhouette_values.max(),
                )
            )
        silhouette_values = sorted(
            silhouette_values, key=lambda tup: tup[2], reverse=True
        )
        for s in silhouette_values:
            print(
                f"    Cluster {s[0]}: Size:{s[1]} | Avg:{s[2]:.2f} | Min:{s[3]:.2f} | Max: {s[4]:.2f}"
            )
    return km, km.labels_

'''
This function computes the adjacent accuracy given the values of the y_predicted and y_true. 
'''
def custom_adjacent_accuracy_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum(np.abs(y_pred - y_true) <= 1) / len(y_pred)

'''
This function prints the results of the classification
'''
def printResults (score, y_test, pred):
    print("Accuracy:", score)
    print("ِAdjecent Accuracy:", custom_adjacent_accuracy_score(y_test, pred))
    print("pearson correaltion 1 ", scipy.stats.pearsonr(y_test, pred))
    print("weighted kappa: ", cohen_kappa_score(y_test, pred, weights="quadratic"))
    print(confusion_matrix(y_test, pred))

'''
This funciton reads all content of essays given their right names and the path to the folder where those essays are located
@:return array of texts of the essays that their names are in the listNames @parameter. 
'''
def readTextFromFileName(listNames, essaysPath):
    returnList= []
    for i in range (len(listNames)):
        my_file = Path(essaysPath, listNames[i])
        if my_file.is_file():
            text = my_file.read_text(encoding='utf-8-sig')
            returnList.append(text)
    return returnList

'''
This funciton appends the clusters' feature vector to the feature vector of each array.
@:return new vector of size tfIDFVectorsArray + 19 after adding the clusters feature vector
'''
def appendClustersToVector(essaysNamesList , tfIDFVectorsArray, vectorizer,finalResultsPath):

    df_goldStandarsdClusters = pd.read_excel(finalResultsPath)
    clusterFeatureVectormulti = []

    for i in range (len(essaysNamesList)):
        clusterVectorFeatureSingle =  [0 for col in range(19)]
        listEssayClusers = df_goldStandarsdClusters[df_goldStandarsdClusters['fileName'].str.contains(essaysNamesList[i])]['cluster']

        if (len(listEssayClusers) ==0):
            print("error")

        for cluster in listEssayClusers:
            clusterVectorFeatureSingle[cluster] = clusterVectorFeatureSingle[cluster] + 1

        clusterFeatureVectormulti.append(clusterVectorFeatureSingle)

    if (vectorizer == "tfidf"):
        #print("tfIDf")
        ### append logic ############
        Xa = tfIDFVectorsArray[0]
        Xb = sparse.csr_matrix(clusterFeatureVectormulti[0])
        diff_n_rows = Xa.shape[0] - Xb.shape[0]
        Xb_new = vstack((Xb, csr_matrix((diff_n_rows, Xb.shape[1]))))
        X_final = hstack((Xa, Xb_new))

        for i in range(1,len(essaysNamesList)):
            Xa = tfIDFVectorsArray[i]
            Xb = sparse.csr_matrix(clusterFeatureVectormulti[i])
            diff_n_rows = Xa.shape[0] - Xb.shape[0]
            Xb_new = vstack((Xb, csr_matrix((diff_n_rows, Xb.shape[1]))))
            X_semiFinal = hstack((Xa, Xb_new))
            X_final = vstack((X_final, X_semiFinal))
        return X_final
    else:
        #print("sbert")
        finalX_Train = []
        for i in range(len(tfIDFVectorsArray)):
            arr = np.append(tfIDFVectorsArray[i],clusterFeatureVectormulti[i])
            finalX_Train.append(arr)
        npa = np.asarray(finalX_Train, dtype=np.ndarray)
        return npa

'''
This function writes the clustering results in 3 excel sheets: 
1. the results of the clustering 
2. the count of items in each cluster ID
3. The 20 most representative clusters  -> the nearst 20 items to the centriods of the clusters. 
'''
def writeClustersToExcel (clusterExcelSheetName,cluster_labels,k,sentences,embeddings,clusteringResutls) :
    df_clusters = pd.DataFrame({
        "text": sentences,
        "cluster": cluster_labels
    })
    df_clusters.to_excel(clusterExcelSheetName + str(k) + ".xlsx")

    df_mostRepresentative = pd.DataFrame({})
    df_ClsuteringSentencesCount = pd.DataFrame({})

    for i in range(k):
        listSentencesMostRep = []
        most_representative_docs_Save = np.argsort(
            np.linalg.norm(embeddings - clusteringResutls.cluster_centers_[i], axis=1)
        )
        queryMostCount = len(df_clusters.query(f"cluster == {i}"))
        df_ClsuteringSentencesCount = df_ClsuteringSentencesCount.append({
            "cluster": i,
            "Sentences Count": queryMostCount
        }, ignore_index=True)
        if (queryMostCount > 20):
            for d in most_representative_docs_Save[:20]:
                listSentencesMostRep.append(sentences[d])
        else:
            for d in most_representative_docs_Save[:queryMostCount]:
                listSentencesMostRep.append(sentences[d])

        for sent in listSentencesMostRep:
            df_mostRepresentative = df_mostRepresentative.append({
                "text": sent,
                "cluster": i
            }, ignore_index=True)

    df_mostRepresentative.to_excel(clusterExcelSheetName + str(k) + " MostRepresentative.xlsx")
    df_ClsuteringSentencesCount.to_excel(clusterExcelSheetName + str(k) + " Count.xlsx")

'''
this funciton plots the graph between the different threshold and the number of missed count
'''
def plotGraph (thresholds, countMissed):
    import matplotlib.pyplot as plt

    # x axis values
    x = thresholds
    # corresponding y axis values
    y = countMissed

    # plotting the points
    plt.plot(x, y, color='green', linestyle='dashed', linewidth=3,
             marker='o', markerfacecolor='blue', markersize=12)

    # setting x and y axis range
    plt.xlim(0.1, 1)
    plt.ylim(50, 70000)

    # naming the x axis
    plt.xlabel('x - axis - value of the threshold')
    # naming the y axis
    plt.ylabel('y - axis - Number of igonred items')

    # giving a title to my graph
    plt.title('Graph between the value of threshold vs number of Igonred items - Teachers')

    # function to show the plot
    plt.show()

'''
this function prints the value of the accuracy, recall and precision using Pairs method
@:parameter path to the csv
@:parameter fileType csv or excel
'''
def printPairsEvaluation(path,fileType) :
    tpCount = 0
    tnCount = 0

    fnCount = 0
    fpcount = 0

    if (fileType == "csv"):
        dfPairEvaluation = pd.read_csv(path)

    else:
        dfPairEvaluation = pd.read_excel(path)

    iteration = dfPairEvaluation.count()['ID']

    for i in range(0, iteration):
        for j in range(i + 1, iteration):

            clusterSystem1 = dfPairEvaluation.query(f"ID == {i}")['cluster'].iloc[0]
            clusterGold1 = dfPairEvaluation.query(f"ID == {i}")['GOLDStandards'].iloc[0]
            clusterSystem2 = dfPairEvaluation.query(f"ID == {j}")['cluster'].iloc[0]
            clusterGold2 = dfPairEvaluation.query(f"ID == {j}")['GOLDStandards'].iloc[0]

            if (clusterSystem1 == clusterSystem2):

                if (clusterGold1 == clusterGold2):  # it is a true positive
                    tpCount = tpCount + 1

                else:
                    fnCount = fnCount + 1  # it is a false negative
            else:
                if (clusterGold1 != clusterGold2):
                    tnCount = tnCount + 1  # it is a true negative
                else:
                    fpcount = fpcount + 1  # it is a false positive.

    print("the value of tp is", tpCount)
    print("the value of tn is", tnCount)
    print("the value of fn is", fnCount)
    print("the value of fb is", fpcount)

    accuracy = (tpCount + tnCount) / (tpCount + tnCount + fpcount + fnCount)

    print("The accuracy is ", accuracy * 100)  # the value is higher than the other accuracy.

    percision = tpCount / (tpCount + fpcount)

    print("The percision is ", percision * 100)

    recall = tpCount / (tpCount + fnCount)

    print("The recall is ", recall * 100)

'''
this function prints the value of the purity 
@:parameter path to the csv
@:parameter fileType csv or excel
'''
def printPurity (path,fileType):

    if (fileType == "csv"):
        dfPurity = pd.read_csv(path)

    else:
        dfPurity = pd.read_excel(path)

    d = defaultdict(list)

    for j in range(0, 19):
        mylist = dfPurity.query(f"cluster == {j}")['GOLDStandards']
        d[j].extend(mylist)

    rightCounter = 0

    for key, value in d.items():
        mostFreq = max(set(value), key=value.count,default=0)
        rightCounter = rightCounter + value.count(mostFreq)

    total = dfPurity.count()['ID']

    print("purity = ", (rightCounter / total) * 100)

'''
this funciton reads the text from the given path and parse them in form of sentences using the structring words
'''
def getSentenceUsingStructringWords(path):
    listStructureWords = []

    dfSrtucturingWords = pd.read_excel("../CSV Files/structureWords.xlsx")
    listStructureWords = dfSrtucturingWords.iloc[:, 0].tolist()
    regex = r"\b(?:{})\b".format("|".join(listStructureWords))
    sentences = []

    for filename in os.listdir(path):
        with open(os.path.join(path, filename)) as f:
            text = f.read()
            text = text.replace("ï»¿", "")
            sents = re.split(regex, text)
            sents = tokenize.sent_tokenize(text)
            for s in sents:
                sentses = tokenize.sent_tokenize(s)
                if (s.isspace() or len(s) == 0):
                    continue
                s = s.lower()
                if (len(s.split()) < 5):
                    continue
                for ss in sentses:
                    if (len(ss.split()) > 5):
                        sentences.append(ss)
    return  sentences

'''
This functions reads the texts from the given path and parses them into normal sentences and tokens
'''
def getSentcesAndTokens(path, withTokens=True):
    sentences = []  # sentences that would be clusters
    for filename in os.listdir(path):
        with open(os.path.join(path, filename)) as f:
            text = f.read()
            text = text.replace("ï»¿", "")
            sents = tokenize.sent_tokenize(text)
            for s in sents:
                sentences.append(s)

    ### list of tokens that would be fed
    tokensSentenceslist = []

    for s in sentences:
        wordsList = gensim.utils.simple_preprocess(s)  # removing the punction and so on....
        filtered_words = [word for word in wordsList if word not in stopwords.words('english')]
        tokensSentenceslist.append(filtered_words)
    if (withTokens):
        return sentences,tokensSentenceslist
    else:
        return sentences

'''
this function iterates over the sentenecs and assigns for each sentence the gold standard id with the highest cosine similairty
@:parameter  embeddings : vectors of all sentences of our dataset
@:parameter sentences, the acutal text of the dataset
@:parameter embRefs vectors of the gold standards 
@:parameter fileName of the excel file
@:parameter tfidf boolean parameter indicates wether the passed array are parse arrays or not. 
@:return allCosinSimiliarityValues list of the value of the highest cosin similairty for each sentence 
'''
def iterateAndAssignGoldStandard (embeddings,sentences,embRefs,fileName,tfIdf= True):
    allCosinSimiliarityValues = []  # contains the value of the highest cosine similairty between a sentence and all gold standards
    df_test = pd.DataFrame({})
    mm = 0
    # looping over all the sentences
    for emb in embeddings:
        listMax = []
        ## looping over all the gold standards to calculate the cos_sim between them and the sentences.
        for embref in embRefs:
            if(tfIdf):
                cos_sim = util.cos_sim(emb.toarray(), embref.toarray())  # cos_sim is the value of the cosine similairty
            else:
                cos_sim = util.cos_sim(emb, embref)
            listMax.append(cos_sim)
        index_max = np.argmax(listMax)  # picking up the index (Gold standard Id)  of the max cosine similairty
        allCosinSimiliarityValues.append(listMax[index_max]) # the value of the highest cosine similairty with this sentence
        df_test = df_test.append({
            "ID": mm,
            "text": sentences[mm],
            "AssignedSimilairty": index_max,
            "Similairty": listMax[index_max]
        }, ignore_index=True)
        mm = mm + 1

    df_test.to_excel(fileName);
    return  allCosinSimiliarityValues