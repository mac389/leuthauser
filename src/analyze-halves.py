import json, csv, nltk, string

import matplotlib.pyplot as plt 
import Graphics as artist
import utils as tech 

from awesome_print import ap 

def to_ascii(word):
	return ''.join([letter for letter in word if ord(letter)<128 and letter not in punkt])

stopwords = set(open('stopwords.txt').read().splitlines())
punkt = set(string.punctuation)

upper_half = json.load(open('upper_half.json','rb'))
lower_half = json.load(open('lower_half.json','rb'))

comments = list(csv.DictReader(open('comments.csv','rb')))

comments = [comment for comment in comments 
		if comment['Student Comment'] != 'None' and comment['Student Comment'] !='NA']

upper_half_comments = [comment for comment in comments if comment['Name'] in upper_half.keys()]
lower_half_comments = [comment for comment in comments if comment['Name'] in lower_half.keys()]

ap('Upper len :%d'%len(upper_half_comments))
ap('Lower len :%d'%len(lower_half_comments))

upper_half_vocabulary  =' '.join(tech.cleanse(' '.join([comment['Student Comment'] for comment in upper_half_comments])))
lower_half_vocabulary = ' '.join(tech.cleanse(' '.join([comment['Student Comment'] for comment in lower_half_comments])))

upper_half_words = [word.lower() for word in nltk.word_tokenize(to_ascii(upper_half_vocabulary)) 
					if word not in punkt and word not in stopwords]
lower_half_words = [word.lower() for word in nltk.word_tokenize(to_ascii(lower_half_vocabulary)) 
					if word not in punkt and word not in stopwords]

upper_freqs = nltk.FreqDist(upper_half_words)
lower_freqs = nltk.FreqDist(lower_half_words)

print tech.weighted_jaccard_similarity(upper_freqs,lower_freqs)

fig,axs = plt.subplots(ncols=2)
for ax,data,label in zip(axs,[upper_freqs,lower_freqs],['Completers','Non-completers']):
	words,freqs = zip(*data.most_common(20))
	ax.plot(freqs,'k--',linewidth=2)
	ax.set_xticks(range(len(freqs)))
	artist.adjust_spines(ax)
	ax.set_xticklabels(map(artist.format,words),rotation='vertical')

	ax.annotate(artist.format(label), xy=(0.8, 0.8), 
		xycoords='axes fraction', horizontalalignment='right', verticalalignment='top')

plt.tight_layout()
plt.savefig('compare-havled-word-freqs.tiff')

