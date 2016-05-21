import nltk, csv, itertools, json

import utils as tech
import Graphics as artist
import matplotlib.pyplot as plt 
import numpy as np 

from awesome_print import ap 
from nltk.util import ngrams
from matplotlib import rcParams

rcParams['text.usetex'] = True

data = list(csv.DictReader(open('comments.csv','rb')))
categories = ['Reporter','Interpreter','Manager','Superior']

data_by_category = {}
for category in categories:
	data_by_category[category] = {}
	comments = ' '.join([student['Student Comment'] for student in data if student['Physician Comment']==category])
	data_by_category[category]['comments'] = tech.cleanse(comments)
	data_by_category[category]['fdist'] = nltk.FreqDist(data_by_category[category]['comments'])
	tech.save_ngrams(data_by_category[category]['fdist'].most_common(50),filename='comments-%s'%category.lower())
	
	data_by_category[category]['bigram.fdist'] = nltk.FreqDist(ngrams(data_by_category[category]['comments'],2))
	tech.save_ngrams(data_by_category[category]['bigram.fdist'].most_common(50),filename='comments-%s'%category.lower())

	data_by_category[category]['count.comments'] = len(comments)
	data_by_category[category]['count.students'] = len([student['Student Comment'] 
					for student in data if student['Physician Comment']==category])

jmat = np.array([[tech.jaccard_similarity(data_by_category[one]['comments'],data_by_category[two]['comments'])
				for one in categories]
				for two in categories])
	
np.savetxt('calculated-jaccard-similarity.tsv',jmat,delimiter='\t',fmt='%.04f')
json.dump({category:data_by_category[category]['count.comments'] for category in categories},open('category-sizes.json','wb'))

nrows=ncols=2
fig,axs = plt.subplots(nrows=2,ncols=2,sharey=True)
for i,row in enumerate(axs):
	for j,col in enumerate(row):
		labels,freqs = zip(*data_by_category[categories[j*nrows+i]]['fdist'].most_common(20))
		labels = [''.join([ch for ch in label if ord(ch)<128]) for label in labels]
		col.plot(freqs,'k--',linewidth = 2)
		artist.adjust_spines(col)
		col.set_xticks(xrange(len(labels)))
		col.set_xticklabels(map(artist.format,labels),rotation='vertical')
		col.set_xlabel(r'$\left(\textrm{Evaluations}=%d,\textrm{Words}=%d\right)$'%(data_by_category[categories[j*nrows+i]]['count.students'],
		data_by_category[categories[j*nrows+i]]['count.comments']))
		
		col.annotate(artist.format(categories[j*nrows+i]), xy=(0.8, 0.8), 
		xycoords='axes fraction', horizontalalignment='right', verticalalignment='top')
		
plt.tight_layout()
fig.text(0.01, 0.5, artist.format('No. of times word occurred'), ha='center', va='center', rotation='vertical')

plt.savefig('frequency-by-category.tiff')