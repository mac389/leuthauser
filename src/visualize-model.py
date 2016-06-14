import json, itertools

import scipy.stats as stats
import numpy as np 
import matplotlib.pyplot as plt 
import Graphics as artist

from matplotlib import rcParams

def simplify(rating):
	if '-' not in rating:
		if not any([level in rating for level in ['Reporter','Interpreter','Manager','Superior']]):
			return None
		else: 
			return rating
	else:
		return rating.split('-')[0]

rcParams['text.usetex'] = True
'''
data = np.loadtxt('../data/model/model/k20.pw_z')
data[data>0.01] = 0

#--- Visualize words
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.imshow(data,interpolation='nearest',aspect='auto',cmap=plt.cm.bone_r)
artist.adjust_spines(ax)
ax.set_ylabel(artist.format('Topics'))
ax.set_xlabel(artist.format('Words'))
ax.set_xticks([])
ax.set_yticks([])
plt.colorbar(cax)
plt.tight_layout()
plt.show()

#--Visualize words
data = np.loadtxt('../data/model/model/k20.pz_d')
data[data>0.01] = 0
print data.shape

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.imshow(data,interpolation='nearest',aspect='auto',cmap=plt.cm.bone_r)
artist.adjust_spines(ax)
ax.set_xlabel(artist.format('Topics'))
ax.set_ylabel(artist.format('Comments'))
ax.set_xticks([])
ax.set_yticks([])
plt.colorbar(cax)
plt.tight_layout()
plt.show()
'''
#-- Sort Topics x Comments with according to ratings 
RATING = 'Physician Comment'
ratings = [u'Superior',u'Reporter', u'Interpreter', 'Manager']
comments = json.load(open('../data/comments.json','rb'))

idx = {rating:[i for i,comment in enumerate(comments) 
			if simplify(comment[RATING]) == rating] for rating in ratings}

comments = [comments[i] for i in itertools.chain.from_iterable(idx.values())]

tps = [1 if simplify(comments[i][RATING]) != simplify(comments[i+1][RATING]) else 0 
		for i in xrange(len(comments)-1)]	

change_idx = [i for i,x in enumerate(tps) if x ==1]
data = np.loadtxt('../data/model/model/k20.pz_d')
data[data>0.01] = 0
'''
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.imshow(data[list(itertools.chain.from_iterable(idx.values())),:],interpolation='nearest',aspect='auto',cmap=plt.cm.bone_r)
for j in change_idx:
	ax.axhline(j)
artist.adjust_spines(ax)
ax.set_xlabel(artist.format('Topics'))
ax.set_ylabel(artist.format('Comments'))
ax.set_xticks([])
ax.set_yticks([])
plt.colorbar(cax)
plt.tight_layout()
plt.show()
'''
#-Relative prevalences of different topics in each category


prevalences = {}
prevalences['Superior'] = np.nanmean(data[:change_idx[0],:],axis=0)
prevalences['Reporter'] = np.nanmean(data[change_idx[0]:change_idx[1],:],axis=0)
prevalences['Interpreter'] = np.nanmean(data[change_idx[1]:change_idx[2],:],axis=0)
prevalences['Manager'] = np.nanmean(data[change_idx[2]:,:],axis=0)


#data = np.array([prevalences[category] for category in ratings])

variances = {}
variances['Superior'] = np.nanvar(data[:change_idx[0],:],axis=0)
variances['Reporter'] = np.nanvar(data[change_idx[0]:change_idx[1],:],axis=0)
variances['Interpreter'] = np.nanvar(data[change_idx[1]:change_idx[2],:],axis=0)
variances['Manager'] = np.nanvar(data[change_idx[2]:,:],axis=0)

lens = {}
lens['Superior'] = 218.
lens['Reporter'] = 218.
lens['Interpreter'] = 114.
lens['Manager'] = 321.


data = np.zeros((data.shape[1],len(ratings),len(ratings)))

for topic in xrange(data.shape[0]):
	for i in xrange(len(ratings)):
		for j in xrange(len(ratings)):
			t = (prevalences[ratings[i]][topic] - prevalences[ratings[j]][topic])/(variances[ratings[i]][topic]/float(lens[ratings[i]])+variances[ratings[j]][topic]/float(lens[ratings[j]]))
			n = max(lens[ratings[i]],lens[ratings[j]])
			data[topic,i,j] = stats.t.sf(np.abs(t), n-1)*2

for topic in xrange(data.shape[0]):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.imshow(np.tril(data[topic,:,:],k=-1),interpolation='nearest',aspect='auto',cmap=plt.cm.bone_r)
	print data[topic,:,:]
	artist.adjust_spines(ax)
	ax.set_yticks(xrange(data.shape[1]))
	ax.set_yticklabels(map(artist.format,ratings))
	ax.set_xticks(xrange(data.shape[2]))
	ax.set_xticklabels(map(artist.format,ratings))
	plt.tight_layout()
	plt.show()

