####################################################################
# dimreduce.py
####################################################################
# Stores all of the dimensionality techniques.
####################################################################

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap

import tensorflow as tf
from tensorflow.keras import layers, losses # Ignore warnings, this works for some reason
from tensorflow.keras.models import Model

class Autoencoder(Model):
  def __init__(self, latent_dim, shape):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.shape = shape
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(600, activation='sigmoid'),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(600, activation='sigmoid'),
      layers.Dense(tf.math.reduce_prod(shape).numpy(), activation='sigmoid'),
      layers.Reshape(shape)
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded



class DimReduce:
    def __init__(self, type, dims):
        self.type = type
        self.red = None

        if(type == "pca"):
            self.red = PCA(n_components=dims)
        elif(type == "lda"):
            self.red = LinearDiscriminantAnalysis(n_components=dims)
        elif(type == "isomap"):
            self.red = Isomap(n_components=dims, n_neighbors=500, max_iter=1, n_jobs=-1)
        elif(type == "autoencoder"):
            self.red = Autoencoder(latent_dim=dims, shape=[3200,])
            self.red.compile(optimizer='adam', loss=losses.MeanSquaredError())
        else:
            1

    
    def fit(self, X, y):
        if type(self.red) == Autoencoder:
            self.red.fit(X, X, epochs=10, shuffle=True,)
        elif self.red:
            self.red.fit(X, y)


    def reduce(self, X):
        if type(self.red) == Autoencoder:
            dX = self.red.encoder(X).numpy()
            return dX
        elif self.red:
            dX = self.red.transform(X)
            return dX
        else:
            return X