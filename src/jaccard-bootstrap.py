import itertools, random, json

import numpy as np 
import utils as tech
import Graphics as artist
import matplotlib.pyplot as plt

from awesome_print import ap
from nltk.util import ngrams 
from matplotlib import rcParams
from awesome_print import ap

rcParams['text.usetex'] = True 
bootstrap_iterations = 10000

categories = ['Reporter','Interpreter','Manager','Superior']

sample_sizes = 50
text = open('all-words-cleansed').read().splitlines()
pdf = np.zeros((bootstrap_iterations,(len(categories)**2-len(categories))/2))
#Second argument is area of equilateral triangle, less diagonal
#that is the number of categories in lower left triangle not counting the diagonal

for i in xrange(bootstrap_iterations):
	pdf[i] = tech.calculate_jmat({category:random.sample(text,50) #Magic constant is number of words in each category, fixed at 50  
				for category in categories})

np.savetxt('bootstrap_values.tsv',pdf,fmt='%.02f',delimiter=',')

jmat = np.loadtxt('calculated-jaccard-similarity.tsv',delimiter='\t')
from_text =np.tril(jmat,k=-1).flatten()
from_text = from_text[from_text.nonzero()]

labels = zip(from_text,['Reporter-Interpreter','Reporter-Manager','Reporter-Educator','Interpreter-Manager',
	'Interpreter-Educator','Manager-Educator'])


#text portion
ind_array = np.arange(0,4,1)
x, y = np.meshgrid(ind_array, ind_array)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.imshow(jmat,interpolation='nearest',aspect='auto',alpha=0.8)
for xval in x.flatten():
	for yval in y.flatten():
		ax.text(xval,yval,r'\Large $\mathbf{%.04f}$'%tech.calculate_pvalue(pdf.flatten(),jmat[xval,yval]) if tech.calculate_pvalue(pdf.flatten(),jmat[xval,yval]) < 0.05 and xval != yval else r'$%.04f$'%tech.calculate_pvalue(pdf.flatten(),jmat[xval,yval]) ,
			va='center',ha='center')

plt.grid(True)
artist.adjust_spines(ax)
ax.set_xticks(range(len(categories)))
ax.set_xticklabels(map(artist.format,categories))
ax.set_yticks(range(len(categories)))
ax.set_yticklabels(map(artist.format,categories))
cbar = plt.colorbar(cax)
cbar.set_label(artist.format('Jaccard similarity'))
plt.tight_layout()
plt.savefig('jaccard-similarity-matrix.tiff')

del fig,ax

fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(pdf.flatten(),color='k',bins=40,alpha=0.7)
for label in labels:
	ymax = pdf.flatten().max()
	ax.axvline(label[0],color='r',linewidth=2,linestyle='--',ymax=ymax)
	ax.annotate(artist.format(label[1]),xy=(label[0],1000),xycoords='data',rotation='vertical')
artist.adjust_spines(ax)
ax.set_xlabel(artist.format('Jaccard Similarity'))
ax.set_ylabel(artist.format('Frequency'))
plt.tight_layout()
plt.savefig('sampled-jaccard-pdf.tiff')
