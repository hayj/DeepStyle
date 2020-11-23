import pandas as pd
from deepstyle.experiment.metrics import *
import numpy as np

def pairwiseCosineSimilarity(data):
    """
        This function compute the pairwise cosine similary of a vector.
        n is the dimension of the input vector, n*n is the size of the output matrix.
    """
    return 1 - pairwise_distances(data, metric='cosine', n_jobs=multiprocessing.cpu_count())

def similarityNDCG(vectors, labels, returnSimMatrix=False, logger=None, verbose=True):
    """
        This fonction take vector representation of documents (so a matrix).
        vectors[0] is the first document and is a vector, for example [1.2, 5.3, -2.4, ..]
        labels are class identifiers of each dimension.
        Can be strings, for example the author of an article:
        ["author1", "author1", "author2", "author3", "author3", "author3", ...]
        This function return the averaged ndcg score over all column of the
        cosine similarity matrix of vectors: for a better understanding of this
        see functions `pairwiseCosineSimilarity` and `pairwiseSimNDCG`.
        Set returnSimMatrix as True to get both the generated matrix and the score in a tuple.
    """
    mtx = pairwiseCosineSimilarity(vectors)
    score = pairwiseSimNDCG(mtx, labels, logger=logger, verbose=verbose)
    if returnSimMatrix:
        return (mtx, score)
    else:
        return score

def pairwiseSimNDCG(simMatrix, labels, logger=None, verbose=True, useNNDCG=True):
    """
        This function take a similary matrix n*n (which is a symmetric matrix)
        with 1.0 on the diagonal.
        It take labels which are class identifiers of each dimension.
        Can be strings, for example the author of an article:
        ["author1", "author1", "author2", "author3", "author3", "author3"]
        It returns the nDCG at k (with k = n) averaged over all columns.
    """
    if useNNDCG:
        ndcgFunct = nndcg
    else:
        ndcgFunct = ndcg
    labels = pd.factorize(labels)[0]
    def rankLabels(col):
        # col = np.array([row[0] + row[1], row[2] + row[3]])
        col = np.vstack([col, labels])
        # https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
        # argsort() to get indexes and sort, [::-1] for reverse
        col = col[:, col[0,:].argsort()[::-1]]
        col = col[1]
        return col
    simMatrix = np.apply_along_axis(rankLabels, 0, simMatrix)
    nDCGs = []
    for x in range(simMatrix.shape[0]):
        col = simMatrix[:, x]
        label = labels[x]
        for y in range(len(col)):
            col[y] = col[y] == label
        nDCGs.append(ndcgFunct(col))
    return np.average(nDCGs)



def ndcg(r, method=0):
    """
        This function return the nDCG at k with k = len(r)
    """
    return ndcg_at_k(r, len(r), method=method)

def nndcg(r, method=0):
    k = len(r)
    idcg = dcg_at_k(sorted(r, reverse=True), k, method)
    if not idcg:
        return 0.
    wdcg = dcg_at_k(sorted(r, reverse=False), k, method)
    if not wdcg:
        return 0.
    dcg = dcg_at_k(r, k, method)
    return (dcg - wdcg) / (idcg - wdcg)

