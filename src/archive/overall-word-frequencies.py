import string, csv, itertools, nltk

import Graphics as artist 
import utils as tech

from nltk.util import ngrams
from awesome_print import ap 

data = list(csv.DictReader(open('comments.csv','rb')))
text = tech.cleanse(' '.join(itertools.chain(record['Student Comment'] for record in data)))

tech.savelines(text,filename='all-words-cleansed')

fdist = nltk.FreqDist(text)
tech.savelines(zip(*fdist.most_common(100)),filename='overall-frequencies-cleansed')

bigram_fdist = nltk.FreqDist(ngrams(text,2))
tech.savelines(zip(*bigram_fdist.most_common(100)),filename='bigram-frequencies-cleansed')

artist.frequency_plot(fdist,filename='overall-frequency-distribution')