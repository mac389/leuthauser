import json, itertools, random
import numpy as np 
from scipy.stats import percentileofscore
import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib import rcParams

#Jaccard Similarity 
def jaccard_similarity(one,two):
	one = set(one)
	two = set(two)

	return len(one & two)/float(len(one | two))

corpus = json.load(open('../data/comments.json','rb'))

INTERPRETER = list(itertools.chain.from_iterable([doc['cleansed text'] for doc in corpus if 'Interpreter' in doc['Physician Comment'] 
														or 'Interpreter-' in doc['Physician Comment']]))

MANAGER = list(itertools.chain.from_iterable([doc['cleansed text'] for doc  in corpus if 'Manager' in doc['Physician Comment'] 
														or 'Manager-' in doc['Physician Comment']]))
														
SUPERIOR = list(itertools.chain.from_iterable([doc['cleansed text'] 
		for doc in corpus if 'Superior' in doc['Physician Comment']]))
														
J = np.zeros((3,3))

for i,one in enumerate([INTERPRETER,MANAGER,SUPERIOR]):
	for j,two in enumerate([INTERPRETER,MANAGER,SUPERIOR]):
		J[i,j] = jaccard_similarity(one,two)

print J
k = ['INTERPRETER','MANAGER','SUPERIOR']
all_text = list(itertools.chain.from_iterable([INTERPRETER,MANAGER,SUPERIOR]))
L = {}
L['INTERPRETER'] = len(INTERPRETER)
L['MANAGER'] = len(MANAGER)
L['SUPERIOR'] = len(SUPERIOR)

#-- bootstrap
n_iterations = 10000

'''
bootstrap_js = [jaccard_similarity(random.sample(all_text,i),random.sample(all_text,j))
				for i,j in itertools.combinations(L.values(),2)
				for _ in xrange(n_iterations)]

np.savetxt('../data/bootstrapped-jsims',bootstrap_js, delimiter='\t',fmt='%.04f')
print np.percentile(bootstrap_js,95), np.percentile(bootstrap_js,5)

for i,j in zip(*np.tril_indices_from(J,k=-1)):
	print percentileofscore(bootstrap_js,J[i,j]),'%s-%s'%(k[i],k[j])


'''

p = {}
p['INTERPRETER-MANAGER'] = 0
p['MANAGER-INTERPRETER'] = 0
p['SUPERIOR-INTERPRETER'] = 0
p['INTERPRETER-SUPERIOR'] = 0
p['SUPERIOR-MANAGER'] = 0.053
p['MANAGER-SUPERIOR'] = 0.053

def stars(i,j):
	key = '%s-%s'%(k[i],k[j])
	print key
	if key in p:
		return ' (p<0.0500)' if p[key] < 0.05 else ' (p=%.04f)'%p[key]
	elif '%s-%s'%(k[j],k[i]) in p:
		key = '%s-%s'%(k[j],k[i])
		return ' (p<0.0500)' if p[key] < 0.05 else ' (p=%.04f)'%p[key]
	else:
		return ''

def create_label(i,j):
	return '%.02f'%(J[i,j]) + stars(i,j)

labels = np.array([[create_label(i,j) for i in xrange(J.shape[1])] 
				for j in xrange(J.shape[0])])
print labels
fig = plt.figure()
ax = fig.add_subplot(111)
mask = np.zeros_like(J)
mask[np.triu_indices_from(J,k=0)] = 1
sns.heatmap(pd.DataFrame(J,index=k,columns=k),ax=ax, mask=mask,square=True, 
		annot=labels, fmt='')

plt.savefig("../imgs/improved-jaccard-similarity.png")

