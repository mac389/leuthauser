import csv 

import numpy as np
import matplotlib.pyplot as plt

from awesome_print import ap

def unique_words(aStr):
	return ' '.join([word for word in set(aStr.split())])

TEXT = 1
basis_vectors = [unique_words(line.split(':')[TEXT]) for line in open('lda-topics.txt','rb').read().splitlines()]
stopwords = set(open('stopwords.txt').read().splitlines())
data = filter(lambda x: x != 'none',[shift['Student Comment'].lower().strip() for shift in csv.DictReader(open('comments.csv'))])
data = [' '.join([word for word in set(string.split()) if word not in stopwords]) for string in data]

ap(basis_vectors)

def jaccard_similarity(a,b):
	a = set(a)
	b = set(b)

	return len(a & b)/float(len(a | b))

def gs(X, row_vecs=True, norm = True):
	if not row_vecs:
		X = X.T
	Y = X[0:1,:].copy()
	for i in range(1, X.shape[0]):
		proj = np.diag((X[i,:].dot(Y.T)/np.linalg.norm(Y,axis=1)**2).flat).dot(Y)
		Y = np.vstack((Y, X[i,:] - proj.sum(0)))
	if norm:
		Y = np.diag(1/np.linalg.norm(Y,axis=1)).dot(Y)
	if row_vecs:
		return Y
	else:
		return Y.T

similarity_matrix = np.array([[jaccard_similarity(vector,entry) for vector in basis_vectors] for entry in data])

#Orthogonalize basis_vectors, easier to use PCA of JS of basis_vectors
basis_vector_correlation_matrix = np.array([[jaccard_similarity(one,two) for one in basis_vectors] for two in basis_vectors])

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.imshow(basis_vector_correlation_matrix,interpolation='nearest',aspect='auto')
cbar = plt.colorbar(cax)
cbar.set_label('Jaccard similarity')
ax.set_ylabel('Student Comment')
ax.set_xlabel('LDA topics')
plt.tight_layout()
plt.show()

