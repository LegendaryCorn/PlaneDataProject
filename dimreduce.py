####################################################################
# dimreduce.py
####################################################################
# Stores all of the dimensionality techniques.
####################################################################

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap

class DimReduce:
    def __init__(self, type, dims):
        self.type = type
        self.red = None

        if(type == "pca"):
            self.red = PCA(n_components=dims)
        elif(type == "lda"):
            self.red = LinearDiscriminantAnalysis(n_components=dims)
        elif(type == "isomap"):
            self.red = Isomap(n_components=dims, n_neighbors=5)
        else:
            1

    
    def fit(self, X, y):
        if self.red:
            self.red.fit(X, y)


    def reduce(self, X):
        if self.red:
            dX = self.red.transform(X)
            return dX
        else:
            return X