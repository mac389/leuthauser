import csv 

import numpy as np 
import matplotlib.pyplot as plt 
import Graphics as artist

from awesome_print import ap 
from matplotlib import rcParams
from scipy.stats import linregress

rcParams['text.usetex'] = True 
data = list(csv.DictReader(open('data.csv','rb')))
comments = list(csv.DictReader(open('comments.csv','rb')))

names = set(student['Name'] for student in comments)

iqr = lambda x: np.subtract(*np.percentile(x, [75, 25]))

def zscore(arr):
	arr = np.array(arr)
	return (arr-np.median(arr))/iqr(arr)

def pop(lst):
	if len(lst) > 0:
		ans = lst.pop()
		if ans != '':
			return ans
		else:
			return None
	else:
		return None

#Condense all comments into a dictionary with student name at keys
condensed_comments = {name:{'comments':[student['Physician Comment'] for student in comments if student['Name']==name]}
						for name in names}

for name in condensed_comments:
	comments = condensed_comments[name]['comments']
	condensed_comments[name]['attending_rating'] = comments.count('Manager')

unified_database = {name:(condensed_comments[name]['attending_rating'], 
						pop([student['exam']  for student in data if name.lower().split()[-1] in student['name']])) 
						for name in names}

attending_rating, exam = zip(*[(x,y) for x,y in unified_database.values() if x is not None and y is not None])
attending_rating = np.array(attending_rating).astype(int)
exam = np.array(exam).astype(float)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(attending_rating,exam, c='.1',s=30)
slope, intercept, r_value, p_value, std_err = linregress(attending_rating,exam)
ax.plot(attending_rating,slope*attending_rating+intercept,'k--')
ax.annotate('$y=%.02f \cdot x + %.02f$'%(slope,intercept), xy=(.9, .4),  xycoords='axes fraction',
        horizontalalignment='right', verticalalignment='top')

ax.annotate('$r^2=%.02f,p=%.02f$'%(r_value**2,p_value), xy=(.9, .35),  xycoords='axes fraction',
        horizontalalignment='right', verticalalignment='top')

artist.adjust_spines(ax)
ax.set_xlabel(artist.format('No. of Superior Ratings'))
ax.set_ylabel(artist.format('Score on Final Exam'))
plt.tight_layout()
plt.savefig('attending-vs-exam.tiff')