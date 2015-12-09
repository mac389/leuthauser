import csv, nltk, itertools, unicodedata
import matplotlib.pyplot as plt 
import Graphics as artist
import numpy as np

from matplotlib import rcParams
rcParams['text.usetex'] = True
data = list(csv.DictReader(open('comments.csv','rb')))

#text = ' '.join(itertools.chain(record['Student Comment'] for record in data))
stopwords = set(open('stopwords.txt').read().splitlines())

#performance_categories = set([record['Physician Comment'] for record in data])
performance_categories = ['Reporter','Interpreter','Manager','Superior']

cohorts = {category: filter(lambda word: word not in stopwords,' '.join([record['Student Comment'] 
												for record in data if category==record['Physician Comment']]).split()) 
												for category in performance_categories}


with open('words-by-category.txt','wb') as f:
	for category in performance_categories:
		labels,freqs = zip(*nltk.FreqDist(cohorts[category]).most_common(20))
		print>>f,category.upper()
		for label,freq in zip(labels,freqs):
			print>>f,label,freq
		print>>f,'\n-----------\n'


by_categories = True
if by_categories:
	nrows=ncols=2
	fig,axs = plt.subplots(nrows=2,ncols=2,sharey=True)
	for i,row in enumerate(axs):
		for j,col in enumerate(row):
			labels,freqs = zip(*nltk.FreqDist(cohorts[performance_categories[j*nrows+i]]).most_common(20))
			labels = [''.join([ch for ch in label if ord(ch)<128]) for label in labels]
			col.plot(freqs,'k--',linewidth = 2)
			artist.adjust_spines(col)
			col.set_xticks(xrange(len(labels)))
			col.set_xticklabels(map(artist.format,labels),rotation='vertical')
			col.annotate(artist.format(performance_categories[j*nrows+i]), xy=(0.8, 0.8), 
			 	xycoords='axes fraction', horizontalalignment='right', verticalalignment='top')

	plt.tight_layout()
	fig.text(0.01, 0.6, artist.format('No. of times word occurred'), ha='center', va='center', rotation='vertical')
	
	plt.show('by_category_frequency.png')#,dpi=300)

def jaccard_similarity(one,two):
	return len(set(one) & set(two))/float(len(set(one) | set(two)))

jmat = [[jaccard_similarity(cohorts[a],cohorts[b]) for a in performance_categories] for b in performance_categories]
DELIMITER = '\t'
#np.savetxt('jaccard_similarity.tsv',jmat,fmt='%.02f',delimiter = DELIMITER,header = DELIMITER.join(performance_categories))
print jmat