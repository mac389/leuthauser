import csv, nltk, string

import numpy as np 
import matplotlib.cm as cm
import matplotlib.pyplot as plt 
import Graphics as artist
import operator as o
import pandas as pd 

from awesome_print import ap
from nltk.tokenize import word_tokenize
from matplotlib import rcParams
from sklearn.decomposition import PCA

rcParams['text.usetex'] = True 

def process(text):
	return ' '.join([nltk.stem.WordNetLemmatizer().lemmatize(word) for word in word_tokenize(text.decode('utf-8').encode('ascii','ignore').lower()) 
			if word not in stopwords and word not in ['x','na'] and word not in set(string.punctuation)])

def project(text,axis):
	weights,words = zip(*[token.split(' * ') for token in axis.split(' : ')])

	#How much of each student comment is in the first four axes? Should save this information. 
	weights = np.array(weights).astype(float)
	weights /= np.sqrt(weights.dot(weights))

	words_weights = dict(zip(words,weights))

	projection = 0 
	for word in words:
		if word in text.split():
			projection += words_weights[word]
	
	return projection

READ = 'rb'
stopwords = open('stopwords.txt').read().splitlines()

with open('comments.csv',READ) as csvfile:
	comments = [row for row in csv.DictReader(csvfile)]	

students =  {entry['Name']:"" for entry in comments}

axes = open('sklearn-lda-topics-w-weights').read().splitlines()

for student in students:
	students[student] = process(' '.join([entry['Student Comment'] for entry in comments
								if entry['Name']==student]))

#ap(students)
keys = [student for student in students.iterkeys() if len(students[student]) > 0]

projections = {student: [project(students[student],axis) for axis in axes] 
	for student in keys}

panel_count = len(axes)
'''
fig,axs = plt.subplots(nrows=panel_count-1,ncols=panel_count-1)
for i in xrange(panel_count):
	for j in xrange(i):
		ax = axs[i-1,j]

		x = [projections[student][i] for student in keys]
		y = [projections[student][j] for student in keys]

		ax.scatter(x,y,c='k')
		artist.adjust_spines(ax)

		ax.set_xlabel(artist.format('Topic %d'%i))
		ax.set_ylabel(artist.format('Topic %d'%j))

plt.tight_layout()
plt.show()
'''
#No clear clustering usin pairs of LDA as axes. What about orthogonalizing?

#No clear clustering after orthogonalization :-(
'''
matrix = np.array([projections[student] for student in keys]) #Each row is normalized. 
matrix /= np.linalg.norm(matrix,axis=0) #Now each column is normalized.

print matrix.mean(axis=0)

pca = PCA(n_components = 5)
X = pca.fit_transform(matrix)
print pca.explained_variance_ratio_


fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X[:,2],X[:,1],c=X[:,0],cmap=plt.cm.bone_r)
plt.show()
'''

#Are different topics prevalence in different groups
projections_width_ratings = {}
for student in keys:
	rating = [entry['Physician Comment'] for entry in comments if entry['Name'] == student][0]
	if rating not in  ['NA','None','Manager-Superior']:	
		projections_width_ratings[student] = {'Rating':rating,'projection':projections[student]}

ratings = ['Reporter','Interpreter','Manager','Superior']


dpoints = np.array([[projections_width_ratings[student]['Rating'],
	'Topic %d'%idx,float(projections_width_ratings[student]['projection'][idx])] for idx in xrange(len(axes)) for student in projections_width_ratings.keys()])

df = pd.DataFrame([[projections_width_ratings[student]['Rating']] + projections[student] for student in projections_width_ratings.keys()], 
	columns = ['Rating'] + ['Topic %d'%idx for idx in xrange(len(axes))])

fig = plt.figure()
ax = fig.add_subplot(111)


df.groupby('Rating').aggregate(np.average).plot.bar(ax=ax)
plt.savefig('draft-LDA-by-student.png')