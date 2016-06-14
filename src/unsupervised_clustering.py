import json 

import numpy as np 
import semantic_similarity as semsim
import seaborn as sns

from progress.bar import Bar 

comments = json.load(open('../data/comments.json','rb'))

S = np.zeros((len(comments),len(comments)))

bar = Bar("Filling similarity matrix",max = S.shape[0]*(S.shape[1]-1)/2)
for i,j in zip(*np.tril_indices_from(S,k=1)):
	if comments[i]['Student Comment'] == "NA" or comments[j]["Student Comment"] == "NA" or comments[i]['Student Comment'] == "None" or comments[j]["Student Comment"] == "None":
		S[i,j] = np.nan
	else:	
		S[i,j] = semsim.similarity(comments[i]['Student Comment'],comments[j]['Student Comment'])
	bar.next()
bar.finish()

S += S.T
S[np.diag_indices_from(S)] = 1
np.savetxt('../data/comments-semantic-similarity',S,delimiter='\t',fmt='%.04f')
g = sns.clustermap(S)
g.savefig('../imgs/cluster-comments-by-semantic-similarity.png')