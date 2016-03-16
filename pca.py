import logging, random, pdb
from time import time

from numpy.random import RandomState
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import MiniBatchKMeans
from sklearn import decomposition



## A lot of code taken from http://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html
## Written by 
# Authors: Vlad Niculae, Alexandre Gramfort
# License: BSD 3 clause

n_row, n_col = 3, 4
n_components = n_row * n_col
image_shape = (64, 64)
rng = RandomState(0)


num_calls = 0
def plot_gallery(images, title="", n_col=n_col, n_row=n_row, image_shape=image_shape, num_calls=[]):
    num_calls.append(1)
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    if not title:
      title = "image " + str(len(num_calls))
    plt.suptitle(title)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)

###############################################################################
# Load faces data
dataset = fetch_olivetti_faces(shuffle=True, random_state=rng)
faces = dataset.data

n_samples, n_features = faces.shape

# global centering
faces_centered = faces - faces.mean(axis=0)

# local centering
faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)
# print faces_centered.shape

pca = decomposition.PCA(n_components=4096)

pca.fit(faces_centered)

rands = []
raw = []
n = 5
## Show n random faces
for _ in range(n):
	rand = random.randrange(0,faces_centered.shape[0])
	rands.append(rand)
	raw.append(faces[rand,:])
# plot_gallery(raw, title="Randomly selected faces")

## Show the top 10 components
comps = []
num_var = 0.0
for i in range(10):
  comps.append(pca.components_[i,:])
  num_var = num_var + pca.explained_variance_ratio_[i]
plot_gallery(comps, title="top 10 components.\n Explain %.2f of variance" % num_var)


## Project the first n faces onto a subspace spanned by all 400 components
comp = pca.components_[0,:]
raw = []
for i in rands:
  img = faces[i,:]
  total = np.zeros((4096,))
  for j in range(400):
    comp = pca.components_[j,:] 
    t = np.dot(img,comp)/np.linalg.norm(comp)
    t = t*comp/np.linalg.norm(comp)
    total += t
  raw.append(faces[i,:])
  raw.append(total)
# plot_gallery(raw, title="random faces next to their projection onto span(all components)")

m = 400
#Show the sum of the m top components. Is this meaningful?
total = np.zeros((4096,))
for i in range(m):
  total+= pca.components_[i]
total /= np.linalg.norm(total)
# plt.figure()
# plt.title("sum of the top %d components" % m)
# plt.imshow(total.reshape((64,64)))


## Again project the first n faces onto a subspace spanned by the first m components
## But we saw before that if you use all 400 components, you get an almost perfect image back
## If we only used 1, we pretty much got the first component but with some areas brighter (glasses, no glasses)
## Can we strike a balance here?
## What if we use the components which account for c% of the variance

cs = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
for c in cs:
  num_c = 0
  perc = 0.0
  for i in pca.explained_variance_ratio_:
    num_c+=1
    perc += i
    if perc >= c:
      break
  print "We will need %d components" %num_c

  raw = []
  for i in rands:
    img = faces[i,:]
    total = np.zeros((4096,))
    for j in range(num_c):
      comp = pca.components_[j,:] 
      t = np.dot(img,comp)/np.linalg.norm(comp)
      t = t*comp/np.linalg.norm(comp)
      total += t
    raw.append(faces[i,:])
    raw.append(total)

  plot_gallery(raw, title="projection onto top %d components necessary \nto explain %f of variance" % (num_c,c))

plt.show()