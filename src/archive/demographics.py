import csv 

import numpy as np 
import matplotlib.pyplot as plt 
import Graphics as artist

from awesome_print import ap 
from matplotlib import rcParams

rcParams['text.usetex'] = True

params = {
   'axes.labelsize': 8,
   'text.fontsize': 8,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'figure.figsize': [4.5, 4.5]
   }
rcParams.update(params)

data = list(csv.DictReader(open('comments.csv','rb')))
categories = list(set([student['Physician Comment'] for student in data])-set(['Cedar','X']))

frequencies = {category:len([student for student in data 
					if student['Physician Comment'] == category])
					for category in categories}

meaningful_category_order = ['Inadequate','Reporter','Reporter-Interpreter','Interpreter',
								'Interpreter-Manager','Interpreter','Interpreter-Superior','Superior',
								'NA']

ratings = [float(student['exam']) for student in list(csv.DictReader(open('data.csv','rb'))) 
	if student['grade'].isdigit()]
data = [frequencies[category] if category in frequencies else 0 
			for category in meaningful_category_order]

fig = plt.figure()
ax = fig.add_subplot(111)
width=0.35
ax.bar(np.arange(len(data))+0.35,data,color='k')
artist.adjust_spines(ax)
ax.set_xticks(np.arange(len(data))+0.75)
ax.set_xticklabels(map(artist.format,meaningful_category_order), rotation='vertical')
ax.set_ylabel(artist.format('No. of Comments'))
plt.tight_layout()
plt.savefig('distribution-of-ratings.tiff')

del fig,ax

fig = plt.figure()
ax = fig.add_subplot(111)
width=0.35
ax.hist(ratings,color='k')
artist.adjust_spines(ax)
ax.set_ylabel(artist.format('No. of Students'))
ax.set_xlabel(artist.format('Final Grade'))
plt.tight_layout()
plt.savefig('distribution-of-scores.tiff')
