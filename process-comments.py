#from __future__ import unicode_literals
import csv, nltk, itertools, unicodedata, string
import matplotlib.pyplot as plt 
import Graphics as artist
import numpy as np 

from matplotlib import rcParams
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.util import ngrams 
from awesome_print import ap 

rcParams['text.usetex'] = True
data = list(csv.DictReader(open('comments.csv','rb')))

#text = ' '.join(itertools.chain(record['Student Comment'] for record in data))
stopwords = set(open('stopwords.txt').read().splitlines())
punkt = set(string.punctuation)
#BASH is less clunky than this
#print len(text.split())   #2807 
#text = [''.join(filter(lambda ch:ord(ch)<128,word.lower())) for word in text.split() if word not in stopwords]
#print len(text)  #1666, with stopwords removed

#fdist = nltk.FreqDist(text)
#labels, freqs = zip(*fdist.most_common(50))
#Most common words in each performance category
lemmatizer = WordNetLemmatizer()
performance_categories = ['reporter','interpreter','manager','superior']

'''
with open('comments-tabulated.txt','wb') as f:
	for sample, frequency in fdist.most_common(100):
		print>>f,sample,frequency
'''

def to_ascii(word):
	return ''.join([letter for letter in word if ord(letter)<128 and letter not in punkt])

def get_text_from_file(filename,DELIMITER=',',TEXT=0, stopwords=stopwords):
	filename = 'comments-%s'%filename
	#Could use better parser, like NLTK
	return [to_ascii(word.lower()) for word in ' '.join([line.split(DELIMITER)[TEXT] for line in open(filename,'rb')]).split() 
		if not any([word == 'None',word == 'NA',word.lower() in stopwords,word.isdigit()])]


text_by_categories = {category:nltk.FreqDist([lemmatizer.lemmatize(word,pos=get_wordnet_pos(pos)) 
						for word,pos in nltk.pos_tag(get_text_from_file(category))]) 
						for category in performance_categories}


unprocessed = list(itertools.chain.from_iterable(text_by_categories.values()))
unprocessed_fdist = nltk.FreqDist(unprocessed)

#all_text = [lemmatizer.lemmatize(word,pos=get_wordnet_pos(pos)) 
#				for word,pos in  nltk.pos_tag(list(itertools.chain.from_iterable(text_by_categories.values())))]


all_text = itertools.chain.from_iterable(text_by_categories.values())
fdist = nltk.FreqDist(all_text)
labels,freqs = zip(*fdist.most_common(50))

'''
bigrams_fdist = nltk.FreqDist(ngrams(all_text,2))
labels,freqs = zip(*bigrams_fdist.most_common(20))
'''
with open('all-text-unprocessed','wb') as outfile:
	for label,freq in unprocessed_fdist.most_common(50):
		print>>outfile,label,freq

with open('all-text-lemmatized_s','wb') as outfile:
	for label,freq in fdist.most_common(50):
		print>>outfile,label,freq

'''
with open('overall-bigrams','wb') as outfile:
	for bigram in bigrams_fdist.most_common(20):
		print>>outfile, ' '.join(bigram[0]),'('+str(bigram[1])+')'
'''
by_categories = True
if by_categories:
	nrows=ncols=2
	fig,axs = plt.subplots(nrows=2,ncols=2,sharey=True)
	for i,row in enumerate(axs):
		for j,col in enumerate(row):
			labels,freqs = zip(*text_by_categories[performance_categories[j*nrows+i]].most_common(20))
			labels = [''.join([ch for ch in label if ord(ch)<128]) for label in labels]
			col.plot(freqs,'k--',linewidth = 2)
			artist.adjust_spines(col)
			col.set_xticks(xrange(len(labels)))
			col.set_xticklabels(map(artist.format,labels),rotation='vertical')
			col.annotate(artist.format(performance_categories[j*nrows+i]), xy=(0.8, 0.8), 
			 	xycoords='axes fraction', horizontalalignment='right', verticalalignment='top')

	plt.tight_layout()
	fig.text(0.01, 0.5, artist.format('No. of times word occurred'), ha='center', va='center', rotation='vertical')
	
	plt.savefig('by_category_frequency_lemmas.png',dpi=300)

with open('words-by-category.txt','wb') as f:
	for category in text_by_categories:
		print>>f,category.upper()
		for sample,frequency in text_by_categories[category].most_common(50):
			print>>f,sample,frequency
		print>>f,'\n-----------\n'

for category in text_by_categories:
	with open(category,'wb') as f:
		for sample,frequency in text_by_categories[category].most_common(50):
			print>>f,sample

def jaccard_similarity(one,two):
		one,_ = zip(*one)
		two,_ = zip(*two)
		return len(set(one) & set(two))/float(len(set(one) | set(two)))

jmat = [[jaccard_similarity(text_by_categories[category_one].most_common(50),text_by_categories[category_two].most_common(50))  
			for category_one in performance_categories] 
			for category_two in performance_categories]

np.savetxt('jmat.csv',jmat,fmt='%.04f',delimiter=',')
show_jmat=False
if show_jmat:
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.imshow(jmat,interpolation='nearest',aspect='auto')
	plt.grid(True)
	artist.adjust_spines(ax)
	ax.set_xticks(range(len(performance_categories)))
	ax.set_xticklabels(map(artist.format,performance_categories))
	ax.set_yticks(range(len(performance_categories)))
	ax.set_yticklabels(map(artist.format,performance_categories))
	cbar = plt.colorbar(cax)
	cbar.set_label(artist.format('Jaccard similarity'))
	plt.tight_layout()
	plt.savefig('jaccard_similarity-12302014',dpi=300)
else:
	print jmat 
overall = True
if overall:
	fig = plt.figure()
	ax = fig.add_subplot(111)
	labels, freqs = zip(*unprocessed_fdist.most_common(50))
	ax.plot(freqs,'k--',linewidth=2)
	artist.adjust_spines(ax)
	ax.set_xticks(xrange(len(labels)))
	ax.set_xticklabels(map(artist.format,labels),rotation='vertical')
	ax.set_ylabel(artist.format('No. of times word occurred'))
	plt.tight_layout()
	plt.savefig('overall_frequency_unprocessed.png',dpi=300)