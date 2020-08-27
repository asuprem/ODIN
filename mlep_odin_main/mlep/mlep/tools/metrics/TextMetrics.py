from sklearn.metrics.pairwise import cosine_similarity


def inverted_cosine_similarity(a,b):
    """ Inverts cosine similarity. Designed for MLEPModelDriftAdaptor and associated classes """
    return 1.0 - cosine_similarity(a.reshape(1,-1), b.reshape(1,-1)).mean()